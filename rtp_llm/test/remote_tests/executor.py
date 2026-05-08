"""Remote Execution client wrapping the REAPI Execute RPC."""

import atexit
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import grpc
from google.protobuf import duration_pb2

from . import bytestream_pb2 as bs_pb2
from . import remote_execution_pb2 as re_pb2
from . import remote_execution_pb2_grpc as re_grpc
from .action_cache_client import _encode_varint
from .cas_client import CASClient
from .endpoint_info import (
    ExecutorEndpointPool,
    combine_reapi_endpoints,
    describe_reapi_endpoint,
    extract_remote_worker_ip,
)

log = logging.getLogger(__name__)

StageCallback = Optional[Callable[[str, str], None]]


def _byte_stream_tail_loop(
    cas: CASClient,
    resource_name: str,
    out_path: Path,
    metadata: List[tuple],
    stop: threading.Event,
) -> None:
    """Poll ByteStream.Read until stop; append chunks to out_path (live remote stdout/stderr)."""
    stub = cas.new_bytestream_stub()
    offset = 0
    with open(out_path, "ab", buffering=0) as f:
        while not stop.is_set():
            req = bs_pb2.ReadRequest(
                resource_name=resource_name, read_offset=offset, read_limit=0
            )
            try:
                for resp in stub.Read(req, metadata=metadata, timeout=300):
                    if stop.is_set():
                        return
                    if resp.data:
                        f.write(resp.data)
                        f.flush()
                        offset += len(resp.data)
            except grpc.RpcError as e:
                if stop.is_set():
                    return
                if e.code() == grpc.StatusCode.OUT_OF_RANGE:
                    return
                time.sleep(0.5)
                continue
            if stop.is_set():
                return
            time.sleep(0.25)


@dataclass
class ExecutionResult:
    exit_code: int
    stdout_raw: bytes = b""
    stderr_raw: bytes = b""
    stdout_digest: Optional[re_pb2.Digest] = None
    stderr_digest: Optional[re_pb2.Digest] = None
    output_files: Dict[str, re_pb2.Digest] = field(default_factory=dict)
    # Filled when remote pytest prints >>>RTP_REMOTE_HOST_IP (actual worker NIC)
    worker_host_ip: Optional[str] = None
    # REAPI ExecutedActionMetadata.worker when the server populates partial_execution_metadata
    metadata_worker: Optional[str] = None
    cached_result: Optional[bool] = None
    response_status_code: Optional[int] = None
    response_status_message: Optional[str] = None
    # Local paths for live-tailed stream logs (ByteStream); same as logged at execute() start
    stream_stdout_path: Optional[str] = None
    stream_stderr_path: Optional[str] = None
    executor_endpoint: Optional[str] = None
    operation_name: Optional[str] = None
    last_stage: Optional[str] = None
    infra_category: Optional[str] = None
    failover_attempts: int = 0


class RemoteExecutor:
    def __init__(
        self,
        executor_endpoint: str,
        cas: CASClient,
        metadata: Optional[List[tuple]] = None,
    ):
        self.grpc_uri = executor_endpoint
        self.reapi_peer_line = describe_reapi_endpoint("executor", executor_endpoint)
        self.reapi_targets_combined = combine_reapi_endpoints(
            cas.grpc_uri, executor_endpoint
        )
        log.info("REAPI %s", self.reapi_targets_combined)

        addr = executor_endpoint.replace("grpc://", "")
        self.channel = grpc.insecure_channel(
            addr,
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.max_pings_without_data", 0),
            ],
        )
        self.stub = re_grpc.ExecutionStub(self.channel)
        self.cas = cas
        self.metadata = metadata or []
        self.instance_name = ""

    def close(self) -> None:
        try:
            self.channel.close()
        except Exception:
            pass

    def cancel_operation(self, operation_name: Optional[str], timeout: int = 5) -> bool:
        if not operation_name:
            return False
        try:
            self.channel.unary_unary(
                "/google.longrunning.Operations/CancelOperation",
                request_serializer=lambda req: req,
                response_deserializer=lambda resp: resp,
            )(
                b"\x0a"
                + _encode_varint(len(operation_name.encode("utf-8")))
                + operation_name.encode("utf-8"),
                metadata=self.metadata,
                timeout=timeout,
            )
            log.info("Cancelled remote operation %s", operation_name)
            return True
        except Exception:
            log.warning("Failed to cancel remote operation %s", operation_name)
            return False

    @staticmethod
    def _try_unpack_execute_metadata(
        op: re_pb2.Operation,
    ) -> Optional[re_pb2.ExecuteOperationMetadata]:
        if not op.metadata or not op.metadata.type_url:
            return None
        meta = re_pb2.ExecuteOperationMetadata()
        try:
            if op.metadata.Unpack(meta):
                return meta
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_stage(op: re_pb2.Operation) -> str:
        meta = RemoteExecutor._try_unpack_execute_metadata(op)
        if meta is None:
            return "UNKNOWN"
        try:
            return re_pb2.ExecutionStage.Value.Name(meta.stage)
        except ValueError:
            return "UNKNOWN"

    def execute(
        self,
        command: List[str],
        input_root_digest: re_pb2.Digest,
        env_vars: Optional[Dict[str, str]] = None,
        platform_properties: Optional[Dict[str, str]] = None,
        timeout: int = 7200,
        output_files: Optional[List[str]] = None,
        on_stage: StageCallback = None,
        stream_stdout_file: Optional[Path] = None,
        stream_stderr_file: Optional[Path] = None,
        no_cache: bool = False,
    ) -> ExecutionResult:
        # Build Command proto — platform.properties are the REAPI equivalent of Bazel
        # exec_properties / remote_default_exec_properties (gpu, gpu_count, …).
        cmd = re_pb2.Command(
            arguments=command,
            environment_variables=[
                re_pb2.Command.EnvironmentVariable(name=k, value=v)
                for k, v in (env_vars or {}).items()
            ],
            output_files=output_files or [],
            platform=re_pb2.Platform(
                properties=[
                    re_pb2.Platform.Property(name=k, value=v)
                    for k, v in (platform_properties or {}).items()
                ]
            ),
        )
        cmd_digest = self.cas.upload_blob(cmd.SerializeToString())

        # Build Action proto
        action = re_pb2.Action(
            command_digest=cmd_digest,
            input_root_digest=input_root_digest,
            timeout=duration_pb2.Duration(seconds=timeout),
            do_not_cache=no_cache,
        )
        action_digest = self.cas.upload_blob(action.SerializeToString())

        log.info(
            "[REMOTE_SUBMIT] executor=%s action=%s timeout=%ds",
            self.grpc_uri,
            action_digest.hash[:12],
            timeout,
        )

        abs_stdout: Optional[str] = None
        abs_stderr: Optional[str] = None
        if stream_stdout_file is not None:
            stream_stdout_file.parent.mkdir(parents=True, exist_ok=True)
            stream_stdout_file.write_bytes(b"")
            abs_stdout = str(stream_stdout_file.resolve())
        if stream_stderr_file is not None:
            stream_stderr_file.parent.mkdir(parents=True, exist_ok=True)
            stream_stderr_file.write_bytes(b"")
            abs_stderr = str(stream_stderr_file.resolve())
        if abs_stdout is not None or abs_stderr is not None:
            log.info(
                "Remote stream logs (tail -f): stdout=%s stderr=%s",
                abs_stdout or "n/a",
                abs_stderr or "n/a",
            )

        # Execute
        request = re_pb2.ExecuteRequest(
            instance_name=self.instance_name,
            action_digest=action_digest,
            skip_cache_lookup=no_cache,
        )

        stop_event = threading.Event()
        stream_threads: List[threading.Thread] = []
        started_stdout = False
        started_stderr = False
        logged_metadata_worker: Optional[str] = None
        last_stage = "SUBMITTED"

        # --- atexit / SIGTERM cancel: abort remote action if local process dies ---
        _op_name_holder: List[Optional[str]] = [None]  # mutable for closure
        _last_op_name_holder: List[Optional[str]] = [None]
        _original_sigterm = signal.getsignal(signal.SIGTERM)

        def _cancel_remote():
            name = _op_name_holder[0]
            if not name:
                return
            _op_name_holder[0] = None  # prevent double cancel
            self.cancel_operation(name)

        def _sigterm_handler(signum, frame):
            _cancel_remote()
            if callable(_original_sigterm) and _original_sigterm not in (
                signal.SIG_DFL,
                signal.SIG_IGN,
            ):
                _original_sigterm(signum, frame)
            else:
                raise SystemExit(128 + signum)

        atexit.register(_cancel_remote)
        # signal.signal() can only be called from the main thread.
        # In per-test mode, execute() runs in a ThreadPoolExecutor worker thread.
        _is_main_thread = threading.current_thread() is threading.main_thread()
        if _is_main_thread:
            signal.signal(signal.SIGTERM, _sigterm_handler)

        # Wall-clock watchdog. The gRPC `timeout=...` on Execute() is supposed
        # to deadline the entire stream, but in practice when a remote worker
        # pool is congested the server holds the stream open indefinitely
        # without sending any stage updates — gRPC's per-message deadline
        # behavior is not enforced and the call hangs forever (no QUEUED→
        # EXECUTING transition, no error). Fire a Timer that cancels the
        # remote operation after `timeout + 180s`; the cancellation closes
        # the stream server-side, raising RpcError here that breaks the
        # for-loop. +180s buffer over the action timeout (vs the gRPC
        # +120s) so the wall-clock fallback fires AFTER the server-side
        # action timeout has had a chance.
        _watchdog_fired = [False]

        def _watchdog():
            _watchdog_fired[0] = True
            log.error(
                "[REMOTE_WATCHDOG] action exceeded %ds wall-clock — cancelling "
                "remote op (server didn't honor stream deadline)",
                timeout + 180,
            )
            _cancel_remote()

        watchdog_timer = threading.Timer(timeout + 180, _watchdog)
        watchdog_timer.daemon = True
        watchdog_timer.start()

        try:
            for op in self.stub.Execute(
                request, metadata=self.metadata, timeout=timeout + 120
            ):
                if _op_name_holder[0] is None and op.name:
                    _op_name_holder[0] = op.name
                if op.name:
                    _last_op_name_holder[0] = op.name
                meta = self._try_unpack_execute_metadata(op)
                if meta is not None:
                    w = (meta.partial_execution_metadata.worker or "").strip()
                    if w and w != logged_metadata_worker:
                        logged_metadata_worker = w
                        st = self._extract_stage(op)
                        log.info(
                            "Execute REAPI worker=%s stage=%s stdout_stream=%s stderr_stream=%s",
                            w,
                            st,
                            bool(meta.stdout_stream_name),
                            bool(meta.stderr_stream_name),
                        )

                    if (
                        stream_stdout_file is not None
                        and meta.stdout_stream_name
                        and not started_stdout
                    ):
                        started_stdout = True
                        t = threading.Thread(
                            target=_byte_stream_tail_loop,
                            args=(
                                self.cas,
                                meta.stdout_stream_name,
                                stream_stdout_file,
                                self.metadata,
                                stop_event,
                            ),
                            name="reapi-stdout-tail",
                            daemon=True,
                        )
                        t.start()
                        stream_threads.append(t)

                    if (
                        stream_stderr_file is not None
                        and meta.stderr_stream_name
                        and not started_stderr
                    ):
                        started_stderr = True
                        t = threading.Thread(
                            target=_byte_stream_tail_loop,
                            args=(
                                self.cas,
                                meta.stderr_stream_name,
                                stream_stderr_file,
                                self.metadata,
                                stop_event,
                            ),
                            name="reapi-stderr-tail",
                            daemon=True,
                        )
                        t.start()
                        stream_threads.append(t)

                if op.done:
                    if on_stage:
                        on_stage("COMPLETED", op.name)
                    last_stage = "COMPLETED"
                    stop_event.set()
                    for t in stream_threads:
                        t.join(timeout=60)
                    result = self._parse(op)
                    # Stream metadata worker (if any) or ActionResult.execution_metadata.worker
                    result.metadata_worker = (
                        (logged_metadata_worker or "").strip()
                        or (result.metadata_worker or "").strip()
                        or None
                    )
                    result.stream_stdout_path = abs_stdout
                    result.stream_stderr_path = abs_stderr
                    result.executor_endpoint = self.grpc_uri
                    result.operation_name = op.name
                    result.last_stage = last_stage
                    self._write_final_stream_files(
                        stream_stdout_file,
                        stream_stderr_file,
                        result,
                        started_stdout,
                        started_stderr,
                    )
                    if result.worker_host_ip:
                        log.info(
                            "Execute worker host_ip=%s operation=%s",
                            result.worker_host_ip,
                            op.name,
                        )
                    return result

                stage = self._extract_stage(op)
                last_stage = stage
                if on_stage:
                    on_stage(stage, op.name)
                log.info("[REMOTE_STAGE] stage=%s op=%s", stage, (op.name or "")[:48])
                log.debug("Operation %s stage=%s", op.name, stage)
        except grpc.RpcError as e:
            log.error("Execute RPC failed: %s", e)
            category = "watchdog_timeout" if _watchdog_fired[0] else "executor_rpc"
            log.error(
                "[RESULT] status=blocked category=%s detail=%s",
                category,
                e.code().name,
            )
            stop_event.set()
            for t in stream_threads:
                t.join(timeout=5)
            detail = f"{e.code().name}: {e.details()}"
            if _watchdog_fired[0]:
                detail = (
                    f"WATCHDOG_TIMEOUT after {timeout + 180}s wall-clock — "
                    f"server didn't honor stream deadline. {detail}"
                )
            tail = f"{detail}\n[reapi-targets] {self.reapi_targets_combined}"
            return ExecutionResult(
                exit_code=-1,
                stderr_raw=tail.encode(),
                metadata_worker=logged_metadata_worker,
                stream_stdout_path=abs_stdout,
                stream_stderr_path=abs_stderr,
                executor_endpoint=self.grpc_uri,
                operation_name=_last_op_name_holder[0],
                last_stage=last_stage,
                infra_category=category,
            )
        finally:
            # Unregister cancel handlers — action completed or errored
            _op_name_holder[0] = None
            atexit.unregister(_cancel_remote)
            watchdog_timer.cancel()
            if _is_main_thread:
                signal.signal(signal.SIGTERM, _original_sigterm)

        stop_event.set()
        for t in stream_threads:
            t.join(timeout=5)
        return ExecutionResult(
            exit_code=-1,
            stderr_raw=(
                "Execute stream ended without result\n"
                f"[reapi-targets] {self.reapi_targets_combined}"
            ).encode(),
            metadata_worker=logged_metadata_worker,
            stream_stdout_path=abs_stdout,
            stream_stderr_path=abs_stderr,
            executor_endpoint=self.grpc_uri,
            operation_name=_last_op_name_holder[0],
            last_stage=last_stage,
            infra_category="executor_stream_ended",
        )

    def _write_final_stream_files(
        self,
        stdout_file: Optional[Path],
        stderr_file: Optional[Path],
        result: ExecutionResult,
        started_byte_stream_stdout: bool,
        started_byte_stream_stderr: bool,
    ) -> None:
        """Always materialize stream log paths from ActionResult (CAS digest or inline).

        Many schedulers never set ExecuteOperationMetadata.stdout_stream_name; ByteStream
        then stays idle. After completion we have full stdout/stderr in the response.
        """
        if stdout_file is not None:
            stdout_file.write_bytes(result.stdout_raw or b"")
        if stderr_file is not None:
            stderr_file.write_bytes(result.stderr_raw or b"")
        if stdout_file is None and stderr_file is None:
            return
        if not started_byte_stream_stdout and not started_byte_stream_stderr:
            if result.stdout_raw or result.stderr_raw:
                log.info(
                    "REAPI did not expose ByteStream log names; wrote final stdout/stderr "
                    "(%d / %d bytes) to stream log paths",
                    len(result.stdout_raw or b""),
                    len(result.stderr_raw or b""),
                )
            else:
                log.info(
                    "REAPI returned empty stdout/stderr (no stream names, no inline/digest data)."
                )
        elif not started_byte_stream_stdout and result.stdout_raw:
            log.info(
                "REAPI had no stdout_stream_name; filled stdout log from ActionResult (%d bytes)",
                len(result.stdout_raw),
            )
        elif not started_byte_stream_stderr and result.stderr_raw:
            log.info(
                "REAPI had no stderr_stream_name; filled stderr log from ActionResult (%d bytes)",
                len(result.stderr_raw),
            )

    def _parse(self, op) -> ExecutionResult:
        resp = re_pb2.ExecuteResponse()
        try:
            # Try Unpack first (handles type_url matching)
            op.response.Unpack(resp)
        except Exception:
            try:
                # Fallback: parse raw value bytes
                resp.ParseFromString(op.response.value)
            except Exception:
                return ExecutionResult(
                    exit_code=-1,
                    stderr_raw=(
                        b"Failed to unpack response\n[reapi-targets] "
                        + self.reapi_targets_combined.encode()
                    ),
                )

        r = resp.result
        log.info(
            "Remote result: exit_code=%d cached=%s status_code=%s status_message=%r stdout_digest=%s stderr_digest=%s",
            r.exit_code,
            resp.cached_result,
            resp.status.code if resp.HasField("status") else None,
            resp.status.message if resp.HasField("status") else "",
            r.stdout_digest.hash[:12] if r.stdout_digest.hash else "none",
            r.stderr_digest.hash[:12] if r.stderr_digest.hash else "none",
        )

        output_files = {f.path: f.digest for f in r.output_files}

        out_raw = r.stdout_raw or b""
        err_raw = r.stderr_raw or b""
        if not out_raw and r.stdout_digest and r.stdout_digest.hash:
            try:
                out_raw = self.cas.download_blob(r.stdout_digest)
            except Exception as e:
                log.warning("Failed to download stdout from digest: %s", e)
        if not err_raw and r.stderr_digest and r.stderr_digest.hash:
            try:
                err_raw = self.cas.download_blob(r.stderr_digest)
            except Exception as e:
                log.warning("Failed to download stderr from digest: %s", e)

        meta_worker = ""
        if r.execution_metadata and r.execution_metadata.worker:
            meta_worker = (r.execution_metadata.worker or "").strip()
            if meta_worker:
                log.info("ActionResult.execution_metadata.worker=%s", meta_worker)
        else:
            log.debug(
                "ActionResult.execution_metadata missing or worker empty "
                "(scheduler may omit REAPI field 9; use >>>RTP_REMOTE_HOST_IP in stdout)"
            )

        out_txt = out_raw.decode("utf-8", errors="replace")
        worker_ip = extract_remote_worker_ip(out_txt)

        return ExecutionResult(
            exit_code=r.exit_code,
            stdout_raw=out_raw,
            stderr_raw=err_raw,
            stdout_digest=r.stdout_digest if r.stdout_digest.hash else None,
            stderr_digest=r.stderr_digest if r.stderr_digest.hash else None,
            output_files=output_files,
            worker_host_ip=worker_ip,
            metadata_worker=meta_worker or None,
            cached_result=resp.cached_result,
            response_status_code=resp.status.code if resp.HasField("status") else None,
            response_status_message=(
                resp.status.message if resp.HasField("status") else None
            ),
        )

    def download_output(self, digest: re_pb2.Digest) -> str:
        data = self.cas.download_blob(digest)
        return data.decode("utf-8", errors="replace")


class FailoverRemoteExecutor:
    """RemoteExecutor wrapper that rotates executor IPs on infra failures."""

    _FAILOVER_CATEGORIES = {
        "executor_rpc",
        "executor_stream_ended",
        "watchdog_timeout",
    }

    def __init__(
        self,
        executor_endpoint: str,
        cas: CASClient,
        metadata: Optional[List[tuple]] = None,
        *,
        enabled: bool = True,
        max_failovers: int = 3,
        dns_refresh_seconds: int = 60,
        fallback_executor_endpoint: Optional[str] = None,
        executor_factory=RemoteExecutor,
    ):
        self.pool = ExecutorEndpointPool(
            executor_endpoint,
            fallback_uri=fallback_executor_endpoint,
            refresh_seconds=dns_refresh_seconds,
        )
        self.cas = cas
        self.metadata = metadata or []
        self.enabled = enabled
        self.max_failovers = max(0, int(max_failovers))
        self.executor_factory = executor_factory
        self._executor: Optional[RemoteExecutor] = None
        self.reapi_targets_combined = combine_reapi_endpoints(
            cas.grpc_uri,
            self.pool.current_endpoint(),
        )

    def _new_executor(self, endpoint: str) -> RemoteExecutor:
        if self._executor is not None:
            self._executor.close()
        self._executor = self.executor_factory(endpoint, self.cas, self.metadata)
        self.reapi_targets_combined = self._executor.reapi_targets_combined
        return self._executor

    @staticmethod
    def _is_failoverable(result: ExecutionResult) -> bool:
        if result.exit_code != -1:
            return False
        return result.infra_category in FailoverRemoteExecutor._FAILOVER_CATEGORIES

    def execute(self, **kwargs) -> ExecutionResult:
        attempts = 0
        endpoint = self.pool.current_endpoint()
        tried = []

        while True:
            executor = self._new_executor(endpoint)
            tried.append(endpoint)
            result = executor.execute(**kwargs)
            result.failover_attempts = attempts
            if not self.enabled or not self._is_failoverable(result):
                return result
            self.pool.refresh(force=True)
            if attempts >= self.max_failovers or len(self.pool.endpoints()) <= 1:
                log.warning(
                    "[EXECUTOR_FAILOVER] exhausted endpoint=%s category=%s "
                    "operation=%s last_stage=%s tried=[%s]",
                    endpoint,
                    result.infra_category,
                    result.operation_name or "n/a",
                    result.last_stage or "n/a",
                    ",".join(tried),
                )
                return result

            old_endpoint = endpoint
            old_operation = result.operation_name
            if old_operation:
                executor.cancel_operation(old_operation)
            endpoint = self.pool.advance(refresh=True)
            attempts += 1
            log.warning(
                "[EXECUTOR_FAILOVER] old=%s new=%s category=%s operation=%s "
                "last_stage=%s will_rerun=true attempt=%d/%d",
                old_endpoint,
                endpoint,
                result.infra_category,
                old_operation or "n/a",
                result.last_stage or "n/a",
                attempts,
                self.max_failovers,
            )

    def download_output(self, digest: re_pb2.Digest) -> str:
        if self._executor is None:
            self._new_executor(self.pool.current_endpoint())
        return self._executor.download_output(digest)

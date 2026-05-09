import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rtp_llm.test.remote_tests import endpoint_info, remote_exec_rtp
from rtp_llm.test.remote_tests.executor import ExecutionResult, FailoverRemoteExecutor


class _FakeCAS:
    grpc_uri = "grpc://cas.service:50051"

    def download_blob(self, digest):
        return b""


class _FakeExecutor:
    results = []
    endpoints = []
    cancelled = []
    closed = []

    def __init__(self, endpoint, cas, metadata):
        self.grpc_uri = endpoint
        self.reapi_targets_combined = f"cas={cas.grpc_uri} | executor={endpoint}"
        self.endpoints.append(endpoint)

    def execute(self, **kwargs):
        return self.results.pop(0)

    def cancel_operation(self, operation_name, timeout=5):
        self.cancelled.append(operation_name)
        return True

    def close(self):
        self.closed.append(self.grpc_uri)

    def download_output(self, digest):
        return ""


def _reset_fake_executor():
    _FakeExecutor.results = []
    _FakeExecutor.endpoints = []
    _FakeExecutor.cancelled = []
    _FakeExecutor.closed = []


def test_executor_pool_resolves_hostname_inside_remote_framework(monkeypatch):
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1", "10.0.0.2"],
    )

    pool = endpoint_info.ExecutorEndpointPool("grpc://scheduler.vipserver:50052")

    assert pool.source_uri == "grpc://scheduler.vipserver:50052"
    assert pool.current_endpoint() == "grpc://10.0.0.1:50052"
    assert pool.advance() == "grpc://10.0.0.2:50052"


def test_executor_pool_samples_single_ip_vipserver_answers(monkeypatch):
    answers = iter([["10.0.0.1"], ["10.0.0.2"]])

    monkeypatch.setattr(endpoint_info, "_FORCED_RESOLVE_SLEEP_SECONDS", 0)
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: next(answers, ["10.0.0.2"]),
    )

    pool = endpoint_info.ExecutorEndpointPool("grpc://scheduler.vipserver:50052")

    assert pool.endpoints() == [
        "grpc://10.0.0.1:50052",
        "grpc://10.0.0.2:50052",
    ]
    assert pool.advance() == "grpc://10.0.0.2:50052"


def test_executor_pool_falls_back_when_primary_hostname_unresolved(monkeypatch):
    def fake_resolve(host, port):
        if host == "scheduler.vipserver":
            return []
        if host == "scheduler.daily":
            return ["10.0.0.9"]
        return []

    monkeypatch.setattr(endpoint_info, "resolve_ipv4_addresses", fake_resolve)

    pool = endpoint_info.ExecutorEndpointPool(
        "grpc://scheduler.vipserver:50052",
        fallback_uri="grpc://scheduler.daily:50052",
    )

    assert pool.source_uri == "grpc://scheduler.vipserver:50052"
    assert pool.active_source_uri == "grpc://scheduler.daily:50052"
    assert pool.current_endpoint() == "grpc://10.0.0.9:50052"


def test_executor_pool_does_not_fallback_for_literal_ip(monkeypatch):
    calls = []

    def fake_resolve(host, port):
        calls.append((host, port))
        return ["10.0.0.9"]

    monkeypatch.setattr(endpoint_info, "resolve_ipv4_addresses", fake_resolve)

    pool = endpoint_info.ExecutorEndpointPool(
        "grpc://127.0.0.1:50052",
        fallback_uri="grpc://scheduler.daily:50052",
    )

    assert pool.current_endpoint() == "grpc://127.0.0.1:50052"
    assert pool.active_source_uri == "grpc://127.0.0.1:50052"
    assert calls == []


def test_default_reapi_endpoints_keep_hostnames(monkeypatch):
    monkeypatch.setattr(
        remote_exec_rtp,
        "_load_pyproject",
        lambda root: {
            "tool": {
                "rtp-llm": {
                    "remote": {
                        "executor-daily": "scheduler.example",
                        "cas-daily": "cas.example",
                        "executor-port": 50052,
                        "cas-port": 50051,
                    }
                }
            }
        },
    )

    executor_ep, cas_ep = remote_exec_rtp.resolve_default_reapi_endpoints(
        rootdir=Path("."),
        env="daily",
    )

    assert executor_ep == "grpc://scheduler.example:50052"
    assert cas_ep == "grpc://cas.example:50051"


def test_online_default_endpoint_config_adds_daily_executor_fallback(monkeypatch):
    monkeypatch.setattr(
        remote_exec_rtp,
        "_load_pyproject",
        lambda root: {
            "tool": {
                "rtp-llm": {
                    "remote": {
                        "executor-online": "scheduler.vipserver",
                        "cas-online": "cas.vipserver",
                        "executor-daily": "scheduler.daily",
                        "cas-daily": "cas.daily",
                        "executor-port": 50052,
                        "cas-port": 50051,
                    }
                }
            }
        },
    )

    endpoints = remote_exec_rtp.resolve_default_reapi_endpoint_config(
        rootdir=Path("."),
        env="online",
    )

    assert endpoints.executor == "grpc://scheduler.vipserver:50052"
    assert endpoints.cas == "grpc://cas.vipserver:50051"
    assert endpoints.fallback_executor == "grpc://scheduler.daily:50052"


def test_failover_retries_same_action_on_next_executor_ip(monkeypatch):
    _reset_fake_executor()
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1", "10.0.0.2"],
    )
    _FakeExecutor.results = [
        ExecutionResult(
            exit_code=-1,
            infra_category="executor_rpc",
            operation_name="operations/1",
            last_stage="QUEUED",
        ),
        ExecutionResult(exit_code=0),
    ]

    executor = FailoverRemoteExecutor(
        "grpc://scheduler.vipserver:50052",
        _FakeCAS(),
        enabled=True,
        max_failovers=1,
        executor_factory=_FakeExecutor,
    )

    result = executor.execute(command=["bash", "-c", "true"])

    assert result.exit_code == 0
    assert result.failover_attempts == 1
    assert _FakeExecutor.endpoints == [
        "grpc://10.0.0.1:50052",
        "grpc://10.0.0.2:50052",
    ]
    assert _FakeExecutor.cancelled == ["operations/1"]


def test_failover_does_not_retry_test_failures(monkeypatch):
    _reset_fake_executor()
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1", "10.0.0.2"],
    )
    _FakeExecutor.results = [ExecutionResult(exit_code=1)]

    executor = FailoverRemoteExecutor(
        "grpc://scheduler.vipserver:50052",
        _FakeCAS(),
        enabled=True,
        max_failovers=1,
        executor_factory=_FakeExecutor,
    )

    result = executor.execute(command=["bash", "-c", "false"])

    assert result.exit_code == 1
    assert _FakeExecutor.endpoints == ["grpc://10.0.0.1:50052"]
    assert _FakeExecutor.cancelled == []


def test_failover_executor_does_not_close_concurrent_actions(monkeypatch):
    monkeypatch.setattr(
        endpoint_info,
        "resolve_ipv4_addresses",
        lambda host, port: ["10.0.0.1"],
    )
    started = []
    all_started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()

    class _BlockingExecutor:
        closed = []

        def __init__(self, endpoint, cas, metadata):
            self.grpc_uri = endpoint
            self.reapi_targets_combined = f"cas={cas.grpc_uri} | executor={endpoint}"

        def execute(self, **kwargs):
            with lock:
                started.append(self.grpc_uri)
                if len(started) == 2:
                    all_started.set()
            assert release.wait(timeout=5)
            return ExecutionResult(exit_code=0)

        def close(self):
            self.closed.append(self.grpc_uri)

    executor = FailoverRemoteExecutor(
        "grpc://scheduler.vipserver:50052",
        _FakeCAS(),
        enabled=True,
        max_failovers=1,
        executor_factory=_BlockingExecutor,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.execute, command=["bash", "-c", "true"])
        second = pool.submit(executor.execute, command=["bash", "-c", "true"])
        assert all_started.wait(timeout=5)
        assert _BlockingExecutor.closed == []
        release.set()
        assert first.result(timeout=5).exit_code == 0
        assert second.result(timeout=5).exit_code == 0

    assert _BlockingExecutor.closed == [
        "grpc://10.0.0.1:50052",
        "grpc://10.0.0.1:50052",
    ]

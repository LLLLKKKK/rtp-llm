# ============================================================================
# xdist Per-Worker GPU Pinning (MUST be first — before any torch import)
# ============================================================================
import os as _os
import sys as _sys

_SESSION_CUDA_VISIBLE_DEVICES = _os.environ.get("CUDA_VISIBLE_DEVICES")

_xdist_worker = _os.environ.get("PYTEST_XDIST_WORKER")
if _xdist_worker:
    if "torch" in _sys.modules:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "torch imported before conftest GPU pinning — "
            "CUDA_VISIBLE_DEVICES will still take effect if cuInit() hasn't fired yet"
        )
    _wn = int(_xdist_worker.replace("gw", ""))

    if _SESSION_CUDA_VISIBLE_DEVICES:
        _gpus = [g.strip() for g in _SESSION_CUDA_VISIBLE_DEVICES.split(",") if g.strip()]
    else:
        _gc = int(_os.environ.get("GPU_COUNT", "0"))
        _gpus = [str(i) for i in range(_gc)] if _gc > 0 else []

    if len(_gpus) > 1:
        _os.environ["CUDA_VISIBLE_DEVICES"] = _gpus[_wn % len(_gpus)]

# ============================================================================
import logging
import os
import pytest
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Marker Registration & Rewriting
# ============================================================================

def _register_synthetic_gpu_marker(config, count: int) -> str:
    synthetic_name = f"gpu_count_{count}"
    registered = getattr(config, "_synthetic_gpu_markers", set())
    if synthetic_name not in registered:
        config.addinivalue_line(
            "markers",
            f"{synthetic_name}: synthetic marker for gpu(count={count}) filtering",
        )
        registered.add(synthetic_name)
        config._synthetic_gpu_markers = registered
    return synthetic_name


def pytest_configure(config):
    """Rewrite gpu(count=N) in -m expressions and register needed synthetic markers."""
    config.addinivalue_line("markers", "manual: test requires manual execution (deselected by default)")
    config._synthetic_gpu_markers = set()

    marker_expr = config.option.markexpr
    if marker_expr:
        rewritten = re.sub(
            r'(?<!\w)gpu\(count\s*=\s*(\d+)\)',
            r'gpu_count_\1',
            marker_expr,
        )
        config.option.markexpr = rewritten
        for count in re.findall(r'(?<!\w)gpu_count_(\d+)\b', rewritten):
            _register_synthetic_gpu_marker(config, int(count))
        logger.debug(f"Modified marker expression: {marker_expr} -> {rewritten}")


# ============================================================================
# GPU Lock — delegates to DeviceResource (file-lock based, cross-process safe)
# ============================================================================

def _get_gpu_count_from_markers(node) -> int:
    """Get required GPU count from @pytest.mark.gpu(count=N), GPU_COUNT env, or default 1."""
    gpu_marker = node.get_closest_marker("gpu")
    if gpu_marker:
        if "count" in gpu_marker.kwargs:
            return int(gpu_marker.kwargs["count"])
        return 1

    gpu_count_env = os.environ.get("GPU_COUNT")
    if gpu_count_env:
        try:
            return int(gpu_count_env)
        except ValueError:
            logger.warning(f"Invalid GPU_COUNT env: {gpu_count_env}, using default 1")

    return 1


@pytest.fixture(scope="function", autouse=True)
def _gpu_isolation(request):
    """Auto GPU isolation for multi-GPU py-ut tests (count>1).

    For count<=1: no-op (worker is already pinned to 1 GPU at module level).
    For count>1: acquires file locks via DeviceResource for the needed GPUs.

    IMPORTANT: Under xdist, each worker is pinned to a single GPU at module
    level. CUDA_VISIBLE_DEVICES cannot be expanded after cuInit() — the CUDA
    runtime reads it once during initialization. Multi-GPU tests that spawn
    subprocesses (mp.spawn/mp.Process) will fail because children inherit the
    pinned single-GPU view. These tests must be skipped under xdist.
    """
    gpu_marker = request.node.get_closest_marker("gpu")
    if not gpu_marker:
        yield
        return

    gpu_count = int(gpu_marker.kwargs.get("count", 1))
    if gpu_count <= 1:
        yield
        return

    if _xdist_worker and _SESSION_CUDA_VISIBLE_DEVICES:
        session_gpus = [g.strip() for g in _SESSION_CUDA_VISIBLE_DEVICES.split(",") if g.strip()]
        if len(session_gpus) < gpu_count:
            pytest.skip(
                f"Skipping multi-GPU test (need {gpu_count} GPUs) under xdist: "
                f"worker pinned to 1 GPU, CUDA_VISIBLE_DEVICES cannot be expanded after cuInit()"
            )

    from rtp_llm.test.utils.device_resource import (
        DeviceResource,
        GpuLockError,
        GPU_LOCK_DEFAULT_TIMEOUT,
        GPU_LOCK_TIMEOUT_ENV,
        get_device_info,
        _get_visible_devices_env,
    )

    device_info = get_device_info()
    if not device_info:
        yield
        return

    device_name, _ = device_info
    env_name = _get_visible_devices_env(device_name)
    saved_cvd = os.environ.get(env_name)

    if _SESSION_CUDA_VISIBLE_DEVICES:
        os.environ[env_name] = _SESSION_CUDA_VISIBLE_DEVICES
    elif env_name in os.environ:
        del os.environ[env_name]

    lock_timeout = int(os.environ.get(GPU_LOCK_TIMEOUT_ENV, GPU_LOCK_DEFAULT_TIMEOUT))
    try:
        with DeviceResource(required_gpu_count=gpu_count, timeout=lock_timeout) as gpu_resource:
            os.environ[env_name] = ",".join(gpu_resource.gpu_ids)
            logger.info(
                "multi-GPU isolation: %s=%s (count=%d)",
                env_name, os.environ[env_name], gpu_count,
            )
            yield gpu_resource
    except GpuLockError as exc:
        pytest.fail(f"GPU lock failed: {exc}", pytrace=False)
    finally:
        if saved_cvd is not None:
            os.environ[env_name] = saved_cvd
        elif env_name in os.environ:
            del os.environ[env_name]


@pytest.fixture(scope="function")
def gpu_lock(request):
    """Function-scoped GPU lock for smoke tests.

    Acquires N GPUs via DeviceResource file locks and sets CUDA_VISIBLE_DEVICES.
    This only affects SUBPROCESSES spawned after the fixture (e.g., server
    processes in smoke tests).  For py-ut in-process CUDA, the autouse
    _gpu_isolation fixture handles GPU assignment instead.
    """
    if request.node.get_closest_marker("no_gpu_lock"):
        yield None
        return

    gpu_count = _get_gpu_count_from_markers(request.node)
    if gpu_count < 1:
        yield None
        return

    from rtp_llm.test.utils.device_resource import (
        DeviceResource,
        GpuLockError,
        GPU_LOCK_DEFAULT_TIMEOUT,
        GPU_LOCK_TIMEOUT_ENV,
        get_device_info,
        _get_visible_devices_env,
    )

    device_info = get_device_info()
    if not device_info:
        yield None
        return

    device_name, _ = device_info
    env_name = _get_visible_devices_env(device_name)

    lock_timeout = int(os.environ.get(GPU_LOCK_TIMEOUT_ENV, GPU_LOCK_DEFAULT_TIMEOUT))
    try:
        with DeviceResource(required_gpu_count=gpu_count, timeout=lock_timeout) as gpu_resource:
            os.environ[env_name] = ",".join(gpu_resource.gpu_ids)
            logger.info(f"gpu_lock: {env_name}={os.environ[env_name]} (count={gpu_count})")
            yield gpu_resource
    except GpuLockError as exc:
        pytest.fail(f"GPU lock failed: {exc}", pytrace=False)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """
    - Deselect tests marked @pytest.mark.manual (require manual execution).
    - Add synthetic gpu_count_N markers before pytest applies -m selection.
    """
    marker_expr = getattr(config.option, "markexpr", "") or ""
    if "manual" not in marker_expr:
        manual_items = []
        remaining = []
        for item in items:
            if item.get_closest_marker("manual"):
                manual_items.append(item)
            else:
                remaining.append(item)
        if manual_items:
            config.hook.pytest_deselected(items=manual_items)
            items[:] = remaining

    for item in items:
        gpu_marker = item.get_closest_marker("gpu")
        if not gpu_marker:
            continue

        gpu_type = gpu_marker.kwargs.get("type")
        if gpu_type:
            item.add_marker(getattr(pytest.mark, gpu_type))

        count = gpu_marker.kwargs.get("count", 1)
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = 1

        synthetic_name = _register_synthetic_gpu_marker(config, count)
        item.add_marker(getattr(pytest.mark, synthetic_name))

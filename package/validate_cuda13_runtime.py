#!/usr/bin/env python3
import argparse
import importlib.metadata
import os
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path

PACKAGE_ROOTS = (
    "rtp_llm",
    "torch",
    "deep_ep",
    "deep_gemm",
    "flashinfer",
    "flashinfer_jit_cache",
    "flashinfer_cubin",
    "flash_attn",
    "flash_attn_3",
    "flash_mla",
    "fast_hadamard_transform",
    "fast_safetensors",
    "fastsafetensors",
    "rtp_kernel",
)
TOP_LEVEL_PATTERNS = (
    "flash_attn_2_cuda*.so*",
    "fast_hadamard_transform_cuda*.so*",
)
HOST_DRIVER_LIBRARIES = {"libcuda.so.1", "libnvidia-ml.so.1"}
FORBIDDEN_NEEDED = {"visibility=hidden"}
DEPENDENCY_RE = re.compile(
    r"\((?:NEEDED|AUXILIARY|FILTER)\).*library: \[([^]]+)\]", re.IGNORECASE
)
SEARCH_PATH_RE = re.compile(
    r"\((?:RPATH|RUNPATH)\).*Library (?:rpath|runpath): \[([^]]*)\]"
)
CUDA_LIBRARY_RE = re.compile(
    r"^lib(?:cudart|cupti|cublas(?:Lt)?|cudnn(?:_[^.]+)?|nccl|nvrtc|"
    r"nvJitLink|cusolver|cusparse|curand|cufft|nvshmem(?:_[^.]+)?)\.so"
)
UNAMBIGUOUS_CUDA12_RE = re.compile(r"^lib(?:cudart|cupti)\.so\.12(?:\.|$)")
CUDA12_PATH_RE = re.compile(
    r"(?:^|[/+_.-])cu(?:da)?[-_.]?12(?:[0-9]*|[/+_.-]|$)", re.IGNORECASE
)
CUDA12_DIST_RE = re.compile(r"(?:^|[-_.+])cu12(?:[0-9]*|[-_.+]|$)", re.IGNORECASE)


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def is_dynamic_elf(path):
    try:
        with path.open("rb") as stream:
            header = stream.read(18)
    except OSError:
        return False
    if len(header) < 18 or header[:4] != b"\x7fELF":
        return False
    endian = {1: "<", 2: ">"}.get(header[5])
    return endian is not None and struct.unpack(f"{endian}H", header[16:18])[0] in (
        2,
        3,
    )


def iter_files(root, seen_directories, seen_files):
    try:
        root_stat = root.stat()
    except OSError as error:
        fail(f"cannot inspect runtime path {root}: {error}")
    if root.is_file():
        identity = (root_stat.st_dev, root_stat.st_ino)
        if identity not in seen_files:
            seen_files.add(identity)
            yield root.resolve()
        return
    for directory, child_directories, files in os.walk(root, followlinks=True):
        directory_path = Path(directory)
        try:
            stat_result = directory_path.stat()
        except OSError as error:
            fail(f"cannot inspect runtime directory {directory_path}: {error}")
        identity = (stat_result.st_dev, stat_result.st_ino)
        if identity in seen_directories:
            child_directories[:] = []
            continue
        seen_directories.add(identity)
        retained = []
        for child in child_directories:
            child_path = directory_path / child
            try:
                child_stat = child_path.stat()
            except OSError as error:
                fail(f"cannot inspect runtime directory {child_path}: {error}")
            child_identity = (child_stat.st_dev, child_stat.st_ino)
            if child_identity not in seen_directories:
                retained.append(child)
        child_directories[:] = retained
        for name in files:
            path = directory_path / name
            try:
                file_stat = path.stat()
            except OSError as error:
                fail(f"cannot inspect runtime file {path}: {error}")
            identity = (file_stat.st_dev, file_stat.st_ino)
            if identity in seen_files:
                continue
            seen_files.add(identity)
            yield path.resolve()


def bad_distribution_files(platlib):
    bad_files = {}
    for distribution in importlib.metadata.distributions(path=[str(platlib)]):
        name = str(distribution.metadata.get("Name") or "")
        version = str(distribution.version or "")
        if not CUDA12_DIST_RE.search(name) and not CUDA12_DIST_RE.search(version):
            continue
        owner = f"{name}=={version}"
        for relative_path in distribution.files or ():
            path = Path(distribution.locate_file(relative_path))
            try:
                bad_files[path.resolve()] = owner
            except OSError:
                continue
    return bad_files


def under(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def find_readelf():
    candidates = (
        "/usr/bin/readelf",
        "/bin/readelf",
        "/usr/bin/eu-readelf",
        "/usr/local/PPU_SDK/bin/llvm-readelf",
    )
    for candidate in candidates:
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    resolved = shutil.which("readelf")
    if not resolved:
        fail("required ELF inspection tool is missing: readelf")
    return resolved


def dynamic_metadata(readelf, inspection_environment, path):
    result = subprocess.run(
        [readelf, "-d", str(path)],
        env=inspection_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout.strip())
    needed = DEPENDENCY_RE.findall(result.stdout)
    search_paths = []
    for value in SEARCH_PATH_RE.findall(result.stdout):
        for item in value.split(":"):
            item = item.strip().strip("'")
            item = item.replace("${ORIGIN}", str(path.parent))
            item = item.replace("$ORIGIN", str(path.parent))
            item = item.replace("${LIB}", "lib64").replace("$LIB", "lib64")
            item = item.replace("${PLATFORM}", os.uname().machine)
            item = item.replace("$PLATFORM", os.uname().machine)
            if item:
                search_paths.append(Path(item))
    return needed, search_paths


def resolve_library(name, search_paths):
    if "/" in name:
        candidate = Path(name)
        return candidate.resolve() if candidate.is_file() else None
    for directory in search_paths:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-cuda-major", type=int, required=True)
    parser.add_argument(
        "--platlib", type=Path, default=Path(sysconfig.get_path("platlib"))
    )
    parser.add_argument("--root", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.expected_cuda_major != 13:
        fail(f"this validator only supports CUDA 13, got {args.expected_cuda_major}")

    readelf = find_readelf()
    inspection_environment = os.environ.copy()
    inspection_environment.pop("LD_LIBRARY_PATH", None)
    platlib = args.platlib.resolve()
    roots = [path.resolve() for path in args.root]
    if not roots:
        roots = [
            (platlib / name).resolve()
            for name in PACKAGE_ROOTS
            if (platlib / name).exists()
        ]
        for pattern in TOP_LEVEL_PATTERNS:
            roots.extend(path.resolve() for path in platlib.glob(pattern))
    if not roots:
        fail("no runtime package roots found")

    library_directories = [
        platlib / "rtp_llm/libs",
        platlib / "torch/lib",
        platlib / "nvidia/cu13/lib",
        platlib / "nvidia/nvshmem/lib",
        Path("/opt/conda310/lib"),
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda/targets/aarch64-linux/lib"),
        Path("/usr/local/cuda/targets/sbsa-linux/lib"),
        Path("/usr/local/PPU_SDK/lib"),
        Path("/usr/local/PPU_SDK/CUDA_SDK/lib64"),
        Path("/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib"),
        Path("/usr/local/PPU_SDK/sailSHMEM/lib"),
        Path("/usr/local/lib64"),
        Path("/usr/local/lib"),
        Path("/lib64"),
        Path("/usr/lib64"),
        Path("/lib"),
        Path("/usr/lib"),
    ]
    library_directories.extend(platlib.glob("nvidia/*/lib"))
    library_directories.extend(
        Path(path) for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path
    )
    for base in (Path("/lib"), Path("/usr/lib")):
        if base.is_dir():
            library_directories.extend(base.glob("*-linux-gnu"))
    library_directories = [path for path in library_directories if path.is_dir()]

    certified_roots = [platlib / "nvidia", platlib / "torch/lib"]
    for path in (Path("/usr/local/cuda"), Path("/usr/local/PPU_SDK")):
        if path.exists():
            certified_roots.append(path.resolve())
    certified_roots.extend(
        path.resolve() for path in Path("/usr/local").glob("cuda-13*")
    )
    bad_owned_files = bad_distribution_files(platlib)

    seen_directories = set()
    seen_files = set()
    seed_elfs = []
    for root in roots:
        seed_elfs.extend(iter_files(root, seen_directories, seen_files))
    seed_elfs = sorted(path for path in seed_elfs if is_dynamic_elf(path))
    if not seed_elfs:
        fail("runtime package roots contain no dynamic ELF files")

    failures = []
    inspected = set()
    pending = list(seed_elfs)
    while pending:
        elf = pending.pop()
        if elf in inspected:
            continue
        inspected.add(elf)
        try:
            needed, object_search_paths = dynamic_metadata(
                readelf, inspection_environment, elf
            )
        except RuntimeError as error:
            failures.append(f"{elf}: readelf failed: {error}")
            continue
        search_paths = object_search_paths + library_directories
        for library_name in needed:
            if library_name in FORBIDDEN_NEEDED:
                failures.append(f"{elf}: forbidden dependency {library_name}")
                continue
            provider = resolve_library(library_name, search_paths)
            if provider is None:
                if library_name not in HOST_DRIVER_LIBRARIES:
                    failures.append(f"{elf}: unresolved dependency {library_name}")
                continue
            owner = bad_owned_files.get(provider)
            if (
                UNAMBIGUOUS_CUDA12_RE.match(library_name)
                or CUDA12_PATH_RE.search(str(provider))
                or owner is not None
            ):
                owner_suffix = f" ({owner})" if owner else ""
                failures.append(
                    f"{elf}: CUDA 12 dependency {library_name} => {provider}{owner_suffix}"
                )
                continue
            if CUDA_LIBRARY_RE.match(library_name) and not any(
                under(provider, root.resolve()) for root in certified_roots
            ):
                failures.append(
                    f"{elf}: CUDA provider is outside certified CUDA 13 roots: "
                    f"{library_name} => {provider}"
                )
            if is_dynamic_elf(provider) and provider not in inspected:
                pending.append(provider)

    if failures:
        print("ERROR: runtime ELF closure validation failed:", file=sys.stderr)
        for failure in sorted(set(failures)):
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"validated runtime ELF closure for {len(inspected)} dynamic object(s) "
        f"from {len(seed_elfs)} seed(s)"
    )


if __name__ == "__main__":
    main()

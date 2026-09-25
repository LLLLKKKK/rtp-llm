#!/usr/bin/env python3
import argparse
import importlib.metadata
import os
import re
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
UNRESOLVED_RE = re.compile(r"^\s*(\S+)\s+=>\s+not found(?:\s|$)")
RESOLVED_RE = re.compile(r"^\s*(\S+)\s+=>\s+(/\S+)(?:\s|$)")
DIRECT_RE = re.compile(r"^\s*(/\S+)(?:\s|$)")
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
    ]
    library_directories.extend(platlib.glob("nvidia/*/lib"))
    library_directories.extend(
        Path(path) for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path
    )
    loader_environment = os.environ.copy()
    loader_environment["LC_ALL"] = "C"
    loader_environment["LD_LIBRARY_PATH"] = ":".join(
        str(path) for path in library_directories if path.is_dir()
    )

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
    elf_files = []
    for root in roots:
        elf_files.extend(iter_files(root, seen_directories, seen_files))
    elf_files = sorted(path for path in elf_files if is_dynamic_elf(path))
    if not elf_files:
        fail("runtime package roots contain no dynamic ELF files")

    failures = []
    checked = 0
    for elf in elf_files:
        dynamic = subprocess.run(
            ["readelf", "-d", str(elf)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if dynamic.returncode:
            failures.append(f"{elf}: readelf failed: {dynamic.stdout.strip()}")
            continue
        if "(NEEDED)" not in dynamic.stdout:
            continue
        inspected = subprocess.run(
            ["ldd", str(elf)],
            env=loader_environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        checked += 1
        output = inspected.stdout.strip()
        if inspected.returncode:
            failures.append(f"{elf}: ldd failed ({inspected.returncode}): {output}")
            continue
        for line in output.splitlines():
            unresolved = UNRESOLVED_RE.match(line)
            if unresolved:
                failures.append(f"{elf}: unresolved dependency {unresolved.group(1)}")
                continue
            resolved = RESOLVED_RE.match(line)
            if resolved:
                library_name, resolved_name = resolved.groups()
                provider = Path(resolved_name).resolve()
            else:
                direct = DIRECT_RE.match(line)
                if not direct:
                    continue
                provider = Path(direct.group(1)).resolve()
                library_name = provider.name
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

    if failures:
        print("ERROR: runtime ELF closure validation failed:", file=sys.stderr)
        for failure in sorted(set(failures)):
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)
    print(f"validated runtime ELF closure for {checked} dynamic object(s)")


if __name__ == "__main__":
    main()

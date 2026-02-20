import argparse
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve


MMG_VERSION_DEFAULT = "5.8.0"
MMG_BASENAMES = ("mmg2d_O3", "mmg3d_O3", "mmgs_O3")


def _norm_os(os_name: str) -> str:
    if os_name.lower() in {"linux"}:
        return "Linux"
    if os_name.lower() in {"darwin", "mac", "macos"}:
        return "Mac"
    if os_name.lower() in {"windows", "win"}:
        return "Windows"
    raise ValueError(f"Unsupported OS: {os_name}")


def _norm_arch(arch: str) -> str:
    a = arch.strip().lower()
    if a in {"x86_64", "amd64"}:
        return "x86_64"
    if a in {"aarch64", "arm64"}:
        return "aarch64"
    if a in {"universal2"}:
        return "universal2"
    raise ValueError(f"Unsupported arch: {arch}")


def _tar_safe_extract(t: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in t.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest) + os.sep):
            raise RuntimeError(f"Unsafe tar path: {member.name}")
    t.extractall(dest)


def _chmod_x(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def _run(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _find_built_mmg_exes(prefix: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for p in prefix.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if os.name == "nt":
            if not name.lower().endswith(".exe"):
                continue
            stem = p.stem
            if stem in MMG_BASENAMES:
                found[stem] = p
        else:
            if name in MMG_BASENAMES:
                found[name] = p
    return found


def build_mmg(*, version: str, os_name: str, arch: str, out_dir: Path, jobs: int) -> None:
    os_name = _norm_os(os_name)
    arch = _norm_arch(arch)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://github.com/MmgTools/mmg/archive/refs/tags/v{version}.tar.gz"

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tar_path = tmp / "mmg.tar.gz"
        src_root = tmp / "src"
        build_dir = tmp / "build"
        install_dir = tmp / "install"
        src_root.mkdir(parents=True, exist_ok=True)
        build_dir.mkdir(parents=True, exist_ok=True)
        install_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading MMG v{version}...", flush=True)
        urlretrieve(url, tar_path)  # nosec - trusted upstream in CI/release flows

        print("Extracting MMG...", flush=True)
        with tarfile.open(tar_path, "r:gz") as t:
            _tar_safe_extract(t, src_root)

        # Archive extracts into mmg-<ver>/
        subdirs = [p for p in src_root.iterdir() if p.is_dir()]
        if not subdirs:
            raise RuntimeError("MMG source extraction produced no subdirectory")
        mmg_src = subdirs[0]

        cmake_cmd = [
            "cmake",
            "-S",
            str(mmg_src),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG",
        ]

        if os_name == "Mac" and arch == "universal2":
            # Build fat binaries.
            deployment = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0").strip()
            cmake_cmd += [
                "-DCMAKE_OSX_ARCHITECTURES=x86_64;arm64",
                f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment}",
            ]

        if os_name == "Windows":
            # Prefer a 64-bit build when using Visual Studio generators.
            cmake_cmd += ["-A", "x64"]

        _run(cmake_cmd)

        build_cmd = ["cmake", "--build", str(build_dir), "--parallel", str(max(1, jobs))]
        if os_name == "Windows":
            build_cmd += ["--config", "Release"]
        _run(build_cmd)

        install_cmd = ["cmake", "--install", str(build_dir)]
        if os_name == "Windows":
            install_cmd += ["--config", "Release"]
        _run(install_cmd)

        found = _find_built_mmg_exes(install_dir)
        missing = sorted(set(MMG_BASENAMES) - set(found.keys()))
        if missing:
            raise RuntimeError(f"MMG build succeeded but executables not found: {missing}")

        # Copy into the repo/package layout
        for stem in MMG_BASENAMES:
            src = found[stem]
            dest = out_dir / src.name
            if dest.exists():
                dest.unlink()
            shutil.copy2(src, dest)
            _chmod_x(dest)

        print("MMG installed to:", out_dir, flush=True)
        for stem in MMG_BASENAMES:
            exe = out_dir / (stem + (".exe" if os.name == "nt" else ""))
            if exe.exists():
                print(" -", exe, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and stage MMG executables for svv packaging.")
    parser.add_argument("--version", default=MMG_VERSION_DEFAULT, help="MMG version (default: 5.8.0)")
    parser.add_argument("--os", dest="os_name", default=platform.system(), help="Target OS (Linux/Mac/Windows)")
    parser.add_argument("--arch", default=platform.machine(), help="Target arch (x86_64/aarch64/universal2)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for executables (default: svv/utils/remeshing/<OS>/<arch>)",
    )
    parser.add_argument("--jobs", type=int, default=(os.cpu_count() or 1), help="Parallel build jobs")
    args = parser.parse_args()

    os_name = _norm_os(args.os_name)
    arch = _norm_arch(args.arch)

    repo_root = Path(__file__).resolve().parents[2]
    default_out = repo_root / "svv" / "utils" / "remeshing" / os_name / arch
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out

    build_mmg(version=args.version, os_name=os_name, arch=arch, out_dir=out_dir, jobs=args.jobs)


if __name__ == "__main__":
    main()

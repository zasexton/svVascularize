"""
Platform/architecture-aware MMG executable selection.

This module intentionally avoids importing heavy dependencies (PyVista, VTK, etc.)
so it can be used in lightweight CI checks and during wheel build validation.
"""

from __future__ import annotations

import errno
import os
import platform
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


MMG_TOOL_TO_EXE = {
    "mmg2d": "mmg2d_O3",
    "mmg3d": "mmg3d_O3",
    "mmgs": "mmgs_O3",
}


def _norm_os_dir(system: Optional[str] = None) -> str:
    sysname = (system or platform.system()).strip()
    if sysname == "Linux":
        return "Linux"
    if sysname == "Windows":
        return "Windows"
    if sysname == "Darwin":
        return "Mac"
    raise RuntimeError(f"Unsupported OS: {sysname}")


def _norm_arch(machine: Optional[str] = None) -> str:
    m = (machine or platform.machine()).strip().lower()
    if m in {"x86_64", "amd64"}:
        return "x86_64"
    if m in {"aarch64", "arm64"}:
        return "aarch64"
    return m or "unknown"


def _mmg_exe_filename(tool: str) -> str:
    if tool not in MMG_TOOL_TO_EXE:
        raise ValueError(f"Unknown MMG tool: {tool!r} (expected one of {sorted(MMG_TOOL_TO_EXE)})")
    base = MMG_TOOL_TO_EXE[tool]
    if os.name == "nt":
        return base + ".exe"
    return base


def _ensure_executable(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        mode = path.stat().st_mode
        if mode & stat.S_IXUSR:
            return
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        # Best-effort; actual execution will raise if this fails.
        pass


def _is_exec_format_error(e: BaseException) -> bool:
    if isinstance(e, OSError):
        if getattr(e, "errno", None) == errno.ENOEXEC:
            return True
        # Windows: ERROR_BAD_EXE_FORMAT (193)
        if getattr(e, "winerror", None) == 193:
            return True
    return False


@dataclass(frozen=True)
class MMGSelection:
    tool: str
    os_dir: str
    arch: str
    candidates: Tuple[Path, ...]


def get_mmg_candidates(
    tool: str,
    *,
    os_dir: Optional[str] = None,
    arch: Optional[str] = None,
) -> MMGSelection:
    """
    Return an ordered list of candidate executable paths for the requested MMG tool.

    Search order:
    1) SVV_MMG_PATH (explicit executable)
    2) SVV_MMG_DIR/<exe>
    3) svv/bin/<exe> (locally built override)
    4) bundled: svv/utils/remeshing/<OS>/<arch>/<exe>
    5) bundled fallbacks (mac): universal2 -> arch -> legacy
    6) legacy flat paths: svv/utils/remeshing/<OS>/<exe>
    """
    os_dir = os_dir or _norm_os_dir()
    arch_override = os.environ.get("SVV_MMG_ARCH")
    arch = (arch_override or arch or "").strip()
    if not arch:
        arch = _norm_arch()
    else:
        a = arch.lower()
        if a in {"amd64", "x86_64"}:
            arch = "x86_64"
        elif a in {"arm64", "aarch64"}:
            arch = "aarch64"
        elif a in {"universal2"}:
            arch = "universal2"

    exe_name = _mmg_exe_filename(tool)

    # Base package dir: .../svv/utils/remeshing
    remesh_dir = Path(__file__).resolve().parent

    candidates: List[Path] = []

    mmg_path = os.environ.get("SVV_MMG_PATH")
    if mmg_path:
        candidates.append(Path(mmg_path).expanduser())

    mmg_dir = os.environ.get("SVV_MMG_DIR")
    if mmg_dir:
        candidates.append(Path(mmg_dir).expanduser() / exe_name)

    # svv/bin override if present
    try:
        import svv  # imported lazily to keep this module lightweight

        svv_root = Path(svv.__file__).resolve().parent
        candidates.append(svv_root / "bin" / exe_name)
    except Exception:
        pass

    # Bundled per-platform layout.
    os_base = remesh_dir / os_dir
    if os_dir == "Mac":
        # Prefer universal2 if present, then machine arch fallbacks.
        candidates.append(os_base / "universal2" / exe_name)
        # Fallback: some trees may use x86_64/arm64 or x86_64/aarch64 naming.
        candidates.append(os_base / "x86_64" / exe_name)
        candidates.append(os_base / "arm64" / exe_name)
        candidates.append(os_base / "aarch64" / exe_name)
    else:
        candidates.append(os_base / arch / exe_name)

    # Legacy flat layout (pre-arch split)
    candidates.append(os_base / exe_name)

    # De-duplicate while preserving order
    uniq: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    return MMGSelection(tool=tool, os_dir=os_dir, arch=arch, candidates=tuple(uniq))


def get_mmg_exe(tool: str) -> Path:
    sel = get_mmg_candidates(tool)
    for p in sel.candidates:
        if p.is_file():
            return p
    tried = "\n  - ".join(str(p) for p in sel.candidates)
    raise FileNotFoundError(
        f"MMG executable not found for tool={tool!r} (os={sel.os_dir}, arch={sel.arch}). Tried:\n  - {tried}"
    )


def run_mmg(
    tool: str,
    args: Sequence[str],
    *,
    stdout=None,
    stderr=None,
) -> Path:
    """
    Execute the requested MMG tool, trying platform/arch candidates as needed.

    Returns the Path of the executable that successfully ran.
    """
    sel = get_mmg_candidates(tool)
    last_exec_error: Optional[BaseException] = None

    for exe in sel.candidates:
        if not exe.is_file():
            continue
        try:
            _ensure_executable(exe)
            subprocess.check_call([str(exe), *map(str, args)], stdout=stdout, stderr=stderr)
            return exe
        except subprocess.CalledProcessError:
            # Tool ran but failed; this is not an architecture-selection issue.
            raise
        except BaseException as e:
            # Exec-format mismatch / permissions: try another candidate.
            if _is_exec_format_error(e) or isinstance(e, PermissionError):
                last_exec_error = e
                continue
            raise

    tried = "\n  - ".join(str(p) for p in sel.candidates)
    if last_exec_error is not None:
        raise RuntimeError(
            f"No compatible MMG executable could be executed for tool={tool!r} "
            f"(os={sel.os_dir}, arch={sel.arch}). Tried:\n  - {tried}\n"
            f"Last error: {last_exec_error}"
        ) from last_exec_error
    raise FileNotFoundError(
        f"MMG executable not found for tool={tool!r} (os={sel.os_dir}, arch={sel.arch}). Tried:\n  - {tried}"
    )


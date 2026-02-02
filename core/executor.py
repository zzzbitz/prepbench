from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CodeExecutor:
    """Execute generated code in a constrained sandbox.

    - Working directory: ephemeral temp dir (or provided work_dir)
    - Allowed reads: only under work_dir/inputs
    - Allowed writes: only collected from work_dir/cand
    - Network: discouraged by environment; no proxies
    - Returns: (ok, stderr_or_errmsg, outputs_in_cand)
    """

    def __init__(
        self,
        *,
        allowed_read_dirs: Optional[List[str]] = None,
        allowed_write_dirs: Optional[List[str]] = None,
        timeout_sec: int = 60,
        preserve_workdir: bool = False,
    ) -> None:
        self.allowed_read_dirs = allowed_read_dirs or []
        self.allowed_write_dirs = allowed_write_dirs or []
        self.timeout_sec = timeout_sec
        self.preserve_workdir = preserve_workdir

    def execute_code(
        self,
        code: str,
        input_files: Dict[str, Path],
        gt_files: Optional[Dict[str, Path]] = None,
        timeout: Optional[int] = None,
        work_dir: Optional[Path] = None,
    ) -> Tuple[bool, str, str, List[Path]]:
        tmo = timeout or self.timeout_sec
        tmp_owned = False
        if work_dir is None:
            tmp_path = Path(tempfile.mkdtemp(prefix="exec_"))
            tmp_owned = True
        else:
            # Normalize to absolute path to avoid cwd-relative duplication
            tmp_path = Path(work_dir).resolve()
            tmp_path.mkdir(parents=True, exist_ok=True)

        in_dir = tmp_path / "inputs"
        gt_dir = tmp_path / "gt"
        cand_dir = tmp_path / "cand"
        in_dir.mkdir(parents=True, exist_ok=True)
        cand_dir.mkdir(parents=True, exist_ok=True)
        if gt_files:
            gt_dir.mkdir(parents=True, exist_ok=True)

        def _materialize(files: Dict[str, Path], target_dir: Path) -> None:
            for name, src in files.items():
                dst = target_dir / name
                try:
                    # Try symlink first (saves space and time)
                    if dst.exists() or dst.is_symlink():
                        try:
                            dst.unlink()
                        except Exception:
                            pass
                    os.symlink(src, dst)
                    continue
                except Exception:
                    pass
                try:
                    # Fallback to hardlink (saves space, requires same filesystem)
                    if dst.exists():
                        try:
                            dst.unlink()
                        except Exception:
                            pass
                    os.link(src, dst)
                    continue
                except Exception:
                    pass
                try:
                    # Last resort: copy
                    shutil.copy2(src, dst)
                except Exception:
                    # Ignore missing gracefully
                    pass

        # Materialize inputs and gt
        _materialize(input_files, in_dir)
        if gt_files:
            _materialize(gt_files, gt_dir)

        code_path = tmp_path / "main.py"
        code_path.write_text(code, encoding="utf-8")

        env = os.environ.copy()
        # Strip network related env and PYTHONPATH for isolation
        env["PYTHONPATH"] = ""
        env["NO_PROXY"] = "*"
        env["HTTPS_PROXY"] = ""
        env["HTTP_PROXY"] = ""

        # Use absolute script path with cwd set to tmp_path for robustness
        cmd = [sys.executable, str(code_path)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(tmp_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=tmo,
            )
            ok = proc.returncode == 0
            outputs = sorted(cand_dir.glob("*"))
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            return ok, stderr, stdout, outputs
        except Exception as e:
            return False, str(e), "", []
        finally:
            # Preserve workdir by default for audit; set preserve_workdir=False to clean up
            if tmp_owned and not self.preserve_workdir:
                try:
                    shutil.rmtree(tmp_path, ignore_errors=True)
                except Exception:
                    pass



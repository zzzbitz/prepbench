from __future__ import annotations

import json
from pathlib import Path
from typing import List


def list_outputs_from_dir(gt_dir: Path) -> List[str]:
    return [p.name for p in sorted(gt_dir.glob("*.csv"))]


def get_allowed_outputs(gt_dir: Path, *, strict: bool = True) -> List[str]:
    if not strict:
        return list_outputs_from_dir(gt_dir)

    cfg_path = gt_dir / "config.json"
    outputs_from_cfg: list[str] = []

    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in GT config file {cfg_path}: {e}")
        files = (cfg.get("files") or {})
        if isinstance(files, dict) and files:
            outputs_from_cfg = list(files.keys())

    outputs_from_dir: list[str] = list_outputs_from_dir(gt_dir)

    if outputs_from_cfg:
        if len(outputs_from_cfg) != len(outputs_from_dir):
            raise ValueError(
                "GT outputs count mismatch between config and GT directory: "
                f"config_count={len(outputs_from_cfg)}, dir_count={len(outputs_from_dir)}"
            )
        return outputs_from_cfg

    raise FileNotFoundError(f"GT config file with 'files' missing or empty: {cfg_path}")


from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from invest_ai_core.manifest import (
    attach_output_files,
    normalize_for_json,
    save_manifest,
    sha256_file,
    sha256_payload,
)
from us_invest_ai.config import RunConfig


def sidecar_manifest_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    file_path = Path(path)
    if file_path.suffix:
        return file_path.with_suffix(".manifest.json")
    return file_path.with_name(f"{file_path.name}.manifest.json")


def build_git_snapshot(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(project_root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}


def build_run_manifest(
    config: RunConfig,
    experiment_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_snapshot = normalize_for_json(config)
    llm_signal_hash = sha256_file(config.llm.signal_path) if config.llm.enabled else None
    universe_snapshot_manifest_path = sidecar_manifest_path(config.data.universe_snapshots_file)

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "project_root": str(config.project_root),
        "config_path": str(config.config_path),
        "config_file_sha256": sha256_file(config.config_path),
        "config_snapshot_sha256": sha256_payload(config_snapshot),
        "git": build_git_snapshot(config.project_root),
        "config": config_snapshot,
        "inputs": {
            "llm_signal_path": str(config.llm.signal_path) if config.llm.enabled else None,
            "llm_signal_sha256": llm_signal_hash,
            "tickers_file_path": str(config.data.tickers_file) if config.data.tickers_file else None,
            "tickers_file_sha256": sha256_file(config.data.tickers_file),
            "metadata_file_path": str(config.data.metadata_file) if config.data.metadata_file else None,
            "metadata_file_sha256": sha256_file(config.data.metadata_file),
            "universe_snapshots_file_path": (
                str(config.data.universe_snapshots_file)
                if config.data.universe_snapshots_file
                else None
            ),
            "universe_snapshots_file_sha256": sha256_file(config.data.universe_snapshots_file),
            "universe_snapshots_manifest_path": (
                str(universe_snapshot_manifest_path)
                if universe_snapshot_manifest_path is not None and universe_snapshot_manifest_path.exists()
                else None
            ),
            "universe_snapshots_manifest_sha256": (
                sha256_file(universe_snapshot_manifest_path)
                if universe_snapshot_manifest_path is not None and universe_snapshot_manifest_path.exists()
                else None
            ),
        },
    }
    if extra:
        manifest["extra"] = normalize_for_json(extra)
    return manifest

from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from us_invest_ai.config import RunConfig


def _normalize_for_json(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_for_json(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(item) for item in value]
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except TypeError:
            return str(value)
    return value


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(_normalize_for_json(payload), indent=2, sort_keys=True, ensure_ascii=True)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    config_snapshot = _normalize_for_json(config)
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
        manifest["extra"] = _normalize_for_json(extra)
    return manifest


def attach_output_files(manifest: dict[str, Any], output_files: dict[str, str | Path | None]) -> dict[str, Any]:
    enriched = deepcopy(manifest)
    enriched["output_files"] = {}
    for name, path in output_files.items():
        if path is None:
            continue
        file_path = Path(path)
        enriched["output_files"][name] = {
            "path": str(file_path),
            "exists": file_path.exists(),
            "sha256": sha256_file(file_path),
        }
    return enriched


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{stable_json_dumps(manifest)}\n", encoding="utf-8")

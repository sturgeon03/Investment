from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def normalize_for_json(value: Any) -> Any:
    if is_dataclass(value):
        return normalize_for_json(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_for_json(item) for item in value]
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except TypeError:
            return str(value)
    return value


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(normalize_for_json(payload), indent=2, sort_keys=True, ensure_ascii=True)


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

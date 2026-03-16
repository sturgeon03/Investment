from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}

    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded

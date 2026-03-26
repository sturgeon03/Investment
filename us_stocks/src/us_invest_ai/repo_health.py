from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any


def write_repo_health_snapshot(repo_root: Path, output_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_path = output_path.resolve()
    command = [
        "git",
        "-c",
        f"safe.directory={repo_root}",
        "-C",
        str(repo_root),
        "status",
        "--short",
        "--branch",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    stderr = result.stderr.decode("utf-8", errors="replace").strip()

    if result.returncode == 0:
        text = stdout or "git status returned no output."
        status = "succeeded"
    else:
        warning = stderr or stdout or "git status failed with no additional output."
        text = f"[repo-health warning]\n{warning}\n"
        status = "warning"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    return {
        "status": status,
        "exit_code": int(result.returncode),
        "output_path": str(output_path),
        "last_output": text.strip().splitlines()[-1] if text.strip() else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a best-effort git status snapshot for automation runs.")
    parser.add_argument("--repo-root", required=True, help="Repository root to inspect.")
    parser.add_argument("--output-path", required=True, help="Text file that receives the status snapshot.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = write_repo_health_snapshot(Path(args.repo_root), Path(args.output_path))
    return 0 if payload["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())

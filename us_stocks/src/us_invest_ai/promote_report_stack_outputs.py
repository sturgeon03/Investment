from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from invest_ai_core.manifest import attach_output_files, save_manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_tree_files(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for child in destination_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for child in source_dir.iterdir():
        destination = destination_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


def _rewrite_manifest_output_paths(manifest: dict[str, Any], destination_dir: Path) -> dict[str, Any]:
    output_files = manifest.get("output_files", {})
    rewritten_paths: dict[str, Path] = {}
    for name, metadata in output_files.items():
        original_path = metadata.get("path")
        if not original_path:
            continue
        rewritten_paths[name] = destination_dir / Path(str(original_path)).name
    return attach_output_files(manifest, rewritten_paths)


def promote_report_directory(
    *,
    source_dir: Path,
    destination_dir: Path,
    manifest_filename: str,
) -> Path:
    source_manifest_path = source_dir / manifest_filename
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest to promote: {source_manifest_path}")

    _copy_tree_files(source_dir, destination_dir)
    manifest = _load_manifest(source_manifest_path)
    promoted_manifest = _rewrite_manifest_output_paths(manifest, destination_dir)
    destination_manifest_path = destination_dir / manifest_filename
    save_manifest(destination_manifest_path, promoted_manifest)
    return destination_manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a timestamped report-stack run into the canonical tracked artifact directories."
    )
    parser.add_argument("--last-year-src", required=True, help="Source directory for the last-year report.")
    parser.add_argument("--last-year-dest", required=True, help="Canonical destination for the last-year report.")
    parser.add_argument("--stability-src", required=True, help="Source directory for the repeated-window report.")
    parser.add_argument("--stability-dest", required=True, help="Canonical destination for the repeated-window report.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    last_year_manifest = promote_report_directory(
        source_dir=Path(args.last_year_src).resolve(),
        destination_dir=Path(args.last_year_dest).resolve(),
        manifest_filename="report_manifest.json",
    )
    stability_manifest = promote_report_directory(
        source_dir=Path(args.stability_src).resolve(),
        destination_dir=Path(args.stability_dest).resolve(),
        manifest_filename="stability_manifest.json",
    )

    print(f"Promoted last-year report to: {last_year_manifest.parent}")
    print(f"Promoted stability report to: {stability_manifest.parent}")


if __name__ == "__main__":
    main()

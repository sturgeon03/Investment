from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from invest_ai_core.manifest import attach_output_files, save_manifest


@dataclass(frozen=True, slots=True)
class DataFrameArtifact:
    name: str
    frame: pd.DataFrame
    filename: str
    index: bool = False
    index_label: str | None = None


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_dataframe_artifacts(
    output_dir: str | Path,
    artifacts: Iterable[DataFrameArtifact],
) -> dict[str, Path]:
    resolved_output_dir = ensure_output_dir(output_dir)
    output_files: dict[str, Path] = {}
    for artifact in artifacts:
        output_path = resolved_output_dir / artifact.filename
        artifact.frame.to_csv(
            output_path,
            index=artifact.index,
            index_label=artifact.index_label,
        )
        output_files[artifact.name] = output_path
    return output_files


def write_manifest_with_outputs(
    output_dir: str | Path,
    manifest_filename: str,
    manifest: dict[str, object],
    output_files: dict[str, str | Path | None],
) -> Path:
    resolved_output_dir = ensure_output_dir(output_dir)
    manifest_path = resolved_output_dir / manifest_filename
    save_manifest(manifest_path, attach_output_files(manifest, output_files))
    return manifest_path

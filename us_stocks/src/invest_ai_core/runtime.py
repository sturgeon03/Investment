from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from invest_ai_core.artifacts import DataFrameArtifact, ensure_output_dir, write_dataframe_artifacts


@dataclass(frozen=True, slots=True)
class ResearchArtifactBundle:
    features: pd.DataFrame
    ranking_history: pd.DataFrame
    target_weights: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: pd.DataFrame


def write_research_artifact_bundle(
    output_dir: str | Path,
    bundle: ResearchArtifactBundle,
    *,
    extra_artifacts: Iterable[DataFrameArtifact] | None = None,
) -> dict[str, Path]:
    resolved_output_dir = ensure_output_dir(output_dir)
    artifacts = [
        DataFrameArtifact("features", bundle.features, "features.csv"),
        DataFrameArtifact("ranking_history", bundle.ranking_history, "ranking_history.csv"),
        DataFrameArtifact("target_weights", bundle.target_weights, "target_weights.csv", index=True, index_label="date"),
        DataFrameArtifact("equity_curve", bundle.equity_curve, "equity_curve.csv", index=True, index_label="date"),
        DataFrameArtifact("summary", bundle.summary, "summary.csv"),
    ]
    if extra_artifacts is not None:
        artifacts.extend(extra_artifacts)
    return write_dataframe_artifacts(resolved_output_dir, artifacts)

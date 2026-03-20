from __future__ import annotations

import json
from pathlib import Path

from us_invest_ai.promote_report_stack_outputs import promote_report_directory


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_promote_report_directory_copies_files_and_rewrites_manifest(tmp_path) -> None:
    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "destination"
    _write_text(source_dir / "deep_learning_summary_last_year.csv", "col\n1\n")
    _write_text(source_dir / "deep_learning_values_last_year.svg", "<svg />\n")
    source_manifest = {
        "experiment_name": "deep_learning_report",
        "output_files": {
            "summary": {
                "path": str(source_dir / "deep_learning_summary_last_year.csv"),
                "exists": True,
                "sha256": "ignore-me",
            },
            "chart": {
                "path": str(source_dir / "deep_learning_values_last_year.svg"),
                "exists": True,
                "sha256": "ignore-me-too",
            },
        },
    }
    _write_text(source_dir / "report_manifest.json", json.dumps(source_manifest))

    promoted_manifest_path = promote_report_directory(
        source_dir=source_dir,
        destination_dir=destination_dir,
        manifest_filename="report_manifest.json",
    )

    promoted_manifest = json.loads(promoted_manifest_path.read_text(encoding="utf-8"))
    assert (destination_dir / "deep_learning_summary_last_year.csv").exists()
    assert (destination_dir / "deep_learning_values_last_year.svg").exists()
    assert promoted_manifest["output_files"]["summary"]["path"] == str(
        destination_dir / "deep_learning_summary_last_year.csv"
    )
    assert promoted_manifest["output_files"]["chart"]["path"] == str(
        destination_dir / "deep_learning_values_last_year.svg"
    )
    assert promoted_manifest["output_files"]["summary"]["sha256"] != "ignore-me"

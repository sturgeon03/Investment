from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from us_invest_ai.config import load_config
from us_invest_ai.experiment_manifest import attach_output_files, build_run_manifest, save_manifest


class ExperimentManifestTests(unittest.TestCase):
    def test_build_run_manifest_tracks_config_and_signal_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            signal_dir = root / "signals"
            config_dir.mkdir(parents=True, exist_ok=True)
            signal_dir.mkdir(parents=True, exist_ok=True)
            signal_path = signal_dir / "llm_scores.generated.csv"
            signal_path.write_text(
                "date,ticker,horizon_bucket,llm_score\n2025-01-31,AAA,swing,0.1\n",
                encoding="utf-8",
            )
            config_path = config_dir / "test.yaml"
            config_path.write_text(
                """
data:
  tickers: [AAA]
  benchmark: SPY
  start: "2024-01-01"
  end:
strategy:
  rebalance: monthly
  top_n: 1
  min_history_days: 20
  trend_filter_mode: soft
  trend_penalty: 0.2
  momentum_20_weight: 0.45
  momentum_60_weight: 0.35
  volatility_weight: -0.20
  llm_weight: 0.30
backtest:
  transaction_cost_bps: 10.0
llm:
  enabled: true
  signal_path: signals/llm_scores.generated.csv
  horizon_bucket: swing
output:
  data_dir: data
  artifacts_dir: artifacts
""".strip(),
                encoding="utf-8",
            )
            config = load_config(config_path)

            manifest = build_run_manifest(config, experiment_name="unit_test", extra={"foo": "bar"})

        self.assertEqual(manifest["experiment_name"], "unit_test")
        self.assertEqual(manifest["config_path"], str(config_path.resolve()))
        self.assertEqual(manifest["extra"]["foo"], "bar")
        self.assertIsNotNone(manifest["inputs"]["llm_signal_sha256"])
        self.assertIsNone(manifest["inputs"]["tickers_file_sha256"])
        self.assertIsNone(manifest["inputs"]["metadata_file_sha256"])
        self.assertIsNone(manifest["inputs"]["universe_snapshots_file_sha256"])

    def test_save_manifest_includes_output_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "result.csv"
            output_path.write_text("a,b\n1,2\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = attach_output_files({"experiment_name": "x"}, {"result": output_path})

            save_manifest(manifest_path, manifest)
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["output_files"]["result"]["path"], str(output_path))
        self.assertTrue(saved["output_files"]["result"]["exists"])
        self.assertIsNotNone(saved["output_files"]["result"]["sha256"])

    def test_build_run_manifest_tracks_snapshot_builder_manifest_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            universe_dir = root / "universes"
            config_dir.mkdir(parents=True, exist_ok=True)
            universe_dir.mkdir(parents=True, exist_ok=True)

            snapshots_path = universe_dir / "dynamic_snapshots.csv"
            snapshots_path.write_text(
                "effective_date,ticker\n2025-01-31,AAA\n",
                encoding="utf-8",
            )
            snapshot_manifest_path = universe_dir / "dynamic_snapshots.manifest.json"
            snapshot_manifest_path.write_text('{"ok": true}\n', encoding="utf-8")
            metadata_path = universe_dir / "metadata.csv"
            metadata_path.write_text("ticker,sector\nAAA,Technology\n", encoding="utf-8")
            tickers_path = universe_dir / "tickers.txt"
            tickers_path.write_text("AAA\n", encoding="utf-8")

            config_path = config_dir / "test.yaml"
            config_path.write_text(
                """
data:
  tickers_file: universes/tickers.txt
  universe_snapshots_file: universes/dynamic_snapshots.csv
  metadata_file: universes/metadata.csv
  benchmark: SPY
  start: "2024-01-01"
  end:
strategy:
  rebalance: monthly
  top_n: 1
  min_history_days: 20
  trend_filter_mode: soft
  trend_penalty: 0.2
  momentum_20_weight: 0.45
  momentum_60_weight: 0.35
  volatility_weight: -0.20
  llm_weight: 0.00
backtest:
  transaction_cost_bps: 10.0
llm:
  enabled: false
  signal_path: signals/llm_scores.generated.csv
  horizon_bucket: swing
output:
  data_dir: data
  artifacts_dir: artifacts
""".strip(),
                encoding="utf-8",
            )
            config = load_config(config_path)

            manifest = build_run_manifest(config, experiment_name="snapshot_test")

        self.assertEqual(
            manifest["inputs"]["universe_snapshots_manifest_path"],
            str(snapshot_manifest_path),
        )
        self.assertIsNotNone(manifest["inputs"]["universe_snapshots_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from kr_invest_ai.data_bundle import KRResearchDataBundle
from kr_invest_ai.main import main
from kr_invest_ai.pipeline import KRPipelineRun


class KRMainCLITests(unittest.TestCase):
    def test_main_writes_run_manifest_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            prices_path = raw_dir / "prices.csv"
            benchmark_path = raw_dir / "benchmark.csv"
            filings_path = raw_dir / "filings.csv"
            raw_manifest_path = raw_dir / "kr_market_data_manifest.json"
            pd.DataFrame([{"date": "2026-03-19", "ticker": "005930.KS", "close": 70000.0}]).to_csv(
                prices_path,
                index=False,
            )
            pd.DataFrame([{"date": "2026-03-19", "ticker": "069500.KS", "close": 30000.0}]).to_csv(
                benchmark_path,
                index=False,
            )
            pd.DataFrame([{"receipt_no": "20260319000123", "ticker": "005930.KS"}]).to_csv(
                filings_path,
                index=False,
            )
            raw_manifest_path.write_text(json.dumps({"request_signature": "abc123"}), encoding="utf-8")

            run = KRPipelineRun(
                bundle=KRResearchDataBundle(
                    prices=pd.read_csv(prices_path),
                    benchmark_prices=pd.read_csv(benchmark_path),
                    filings=pd.read_csv(filings_path),
                    price_bar_count=1,
                    benchmark_bar_count=1,
                    filing_count=1,
                    provenance={"market_data_provider": "stub_market", "dart_enabled": False},
                ),
                raw_dir=raw_dir,
                manifest_path=raw_manifest_path,
                prices_path=prices_path,
                benchmark_path=benchmark_path,
                filings_path=filings_path,
                from_cache=False,
                manifest={"request_signature": "abc123"},
            )

            artifacts_dir = Path(temp_dir) / "artifacts"
            argv = [
                "kr_invest_ai.main",
                "--tickers",
                "005930",
                "--price-start-date",
                "2026-03-01",
                "--price-end-date",
                "2026-03-19",
                "--data-dir",
                str(Path(temp_dir) / "data"),
                "--artifacts-dir",
                str(artifacts_dir),
            ]

            with patch("kr_invest_ai.main.run_kr_data_pipeline", return_value=run):
                with patch("sys.argv", argv):
                    main()

            run_manifest = json.loads((artifacts_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["pipeline"], "kr_research_data_pipeline")
            self.assertFalse(run_manifest["from_cache"])
            self.assertEqual(run_manifest["output_files"]["prices"]["path"], str(prices_path))
            self.assertEqual(run_manifest["output_files"]["benchmark"]["path"], str(benchmark_path))
            self.assertEqual(run_manifest["output_files"]["raw_manifest"]["path"], str(raw_manifest_path))


if __name__ == "__main__":
    unittest.main()

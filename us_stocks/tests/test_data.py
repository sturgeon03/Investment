from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import invest_ai_core.market_data as shared_market_data
from us_invest_ai.data import download_ohlcv, prepare_market_data_bundle
from us_invest_ai.experiment_manifest import sha256_file


class DataBundleTests(unittest.TestCase):
    def test_shared_market_data_module_is_compatible_with_us_wrapper(self) -> None:
        self.assertIs(shared_market_data.download_ohlcv, download_ohlcv)
        self.assertIs(shared_market_data.prepare_market_data_bundle, prepare_market_data_bundle)

    def test_download_ohlcv_retries_after_transient_failure(self) -> None:
        downloaded = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.0, 101.0],
                "Volume": [1000, 1100],
            },
            index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
        )
        downloaded.index.name = "Date"

        with patch("us_invest_ai.data._sleep_backoff") as mocked_sleep:
            with patch(
                "us_invest_ai.data.yf.download",
                side_effect=[RuntimeError("temporary failure"), downloaded],
            ) as mocked_download:
                prices = download_ohlcv(["AAA"], start="2025-01-01", end=None)

        self.assertEqual(mocked_download.call_count, 2)
        mocked_sleep.assert_called_once()
        self.assertEqual(prices["ticker"].unique().tolist(), ["AAA"])
        self.assertEqual(len(prices), 2)

    def test_prepare_market_data_bundle_uses_cache_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            prices_path = raw_dir / "prices.csv"
            benchmark_path = raw_dir / "benchmark.csv"
            manifest_path = raw_dir / "market_data_manifest.json"
            metadata_path = root / "universes" / "metadata.csv"
            snapshots_path = root / "universes" / "snapshots.csv"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            prices_path.write_text(
                (
                    "date,ticker,open,high,low,close,volume\n"
                    "2025-01-02,AAA,100,101,99,100,1000\n"
                    "2025-01-03,AAA,101,102,100,101,1100\n"
                ),
                encoding="utf-8",
            )
            benchmark_path.write_text(
                (
                    "date,ticker,open,high,low,close,volume\n"
                    "2025-01-02,SPY,500,501,499,500,1000000\n"
                    "2025-01-03,SPY,501,502,500,501,1100000\n"
                ),
                encoding="utf-8",
            )
            metadata_path.write_text("ticker,sector\nAAA,Technology\n", encoding="utf-8")
            snapshots_path.write_text(
                "effective_date,ticker\n2025-01-01,AAA\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "query": {
                            "tickers": ["AAA"],
                            "benchmark": "SPY",
                            "start": "2025-01-01",
                            "end": None,
                        },
                        "inputs": {
                            "tickers_file": None,
                            "metadata_file": {
                                "path": str(metadata_path),
                                "exists": True,
                                "sha256": sha256_file(metadata_path),
                            },
                            "universe_snapshots_file": {
                                "path": str(snapshots_path),
                                "exists": True,
                                "sha256": sha256_file(snapshots_path),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("us_invest_ai.data.download_ohlcv") as mocked_download:
                bundle = prepare_market_data_bundle(
                    data_dir=root / "data",
                    tickers=["AAA"],
                    benchmark="SPY",
                    start="2025-01-01",
                    end=None,
                    metadata_file=metadata_path,
                    universe_snapshots_file=snapshots_path,
                )

            mocked_download.assert_not_called()
            self.assertEqual(bundle.provenance["source"], "cache")
            self.assertEqual(int(bundle.prices["close"].iloc[-1]), 101)
            self.assertEqual(int(bundle.benchmark_prices["close"].iloc[-1]), 501)
            self.assertEqual(bundle.provenance["inputs"]["metadata_file"]["sha256"] is not None, True)
            self.assertEqual(bundle.provenance["inputs"]["universe_snapshots_file"]["sha256"] is not None, True)

            manifest_path = raw_dir / "market_data_manifest.json"
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["source"], "cache")
            self.assertEqual(saved["prices_summary"]["ticker_count"], 1)
            self.assertEqual(saved["benchmark_summary"]["ticker_count"], 1)

    def test_prepare_market_data_bundle_reuses_cache_when_paths_change_but_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            prices_path = raw_dir / "prices.csv"
            benchmark_path = raw_dir / "benchmark.csv"
            manifest_path = raw_dir / "market_data_manifest.json"
            old_inputs_root = root / "old_inputs"
            new_inputs_root = root / "new_inputs"
            old_inputs_root.mkdir(parents=True, exist_ok=True)
            new_inputs_root.mkdir(parents=True, exist_ok=True)
            old_metadata_path = old_inputs_root / "metadata.csv"
            new_metadata_path = new_inputs_root / "metadata.csv"
            old_snapshots_path = old_inputs_root / "snapshots.csv"
            new_snapshots_path = new_inputs_root / "snapshots.csv"

            prices_path.write_text(
                (
                    "date,ticker,open,high,low,close,volume\n"
                    "2025-01-02,AAA,100,101,99,100,1000\n"
                    "2025-01-03,AAA,101,102,100,101,1100\n"
                ),
                encoding="utf-8",
            )
            benchmark_path.write_text(
                (
                    "date,ticker,open,high,low,close,volume\n"
                    "2025-01-02,SPY,500,501,499,500,1000000\n"
                    "2025-01-03,SPY,501,502,500,501,1100000\n"
                ),
                encoding="utf-8",
            )
            old_metadata_path.write_text("ticker,sector\nAAA,Technology\n", encoding="utf-8")
            new_metadata_path.write_text("ticker,sector\nAAA,Technology\n", encoding="utf-8")
            old_snapshots_path.write_text("effective_date,ticker\n2025-01-01,AAA\n", encoding="utf-8")
            new_snapshots_path.write_text("effective_date,ticker\n2025-01-01,AAA\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "query": {
                            "tickers": ["AAA"],
                            "benchmark": "SPY",
                            "start": "2025-01-01",
                            "end": None,
                        },
                        "inputs": {
                            "tickers_file": None,
                            "metadata_file": {
                                "path": str(old_metadata_path),
                                "exists": True,
                                "sha256": sha256_file(old_metadata_path),
                            },
                            "universe_snapshots_file": {
                                "path": str(old_snapshots_path),
                                "exists": True,
                                "sha256": sha256_file(old_snapshots_path),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("us_invest_ai.data.download_ohlcv") as mocked_download:
                bundle = prepare_market_data_bundle(
                    data_dir=root / "data",
                    tickers=["AAA"],
                    benchmark="SPY",
                    start="2025-01-01",
                    end=None,
                    metadata_file=new_metadata_path,
                    universe_snapshots_file=new_snapshots_path,
                )

            mocked_download.assert_not_called()
            self.assertEqual(bundle.provenance["source"], "cache")
            self.assertTrue(bundle.provenance["yfinance_cache_dir"].endswith("yfinance_cache"))

    def test_prepare_market_data_bundle_refreshes_when_request_signature_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            prices_path = raw_dir / "prices.csv"
            benchmark_path = raw_dir / "benchmark.csv"
            manifest_path = raw_dir / "market_data_manifest.json"

            prices_path.write_text(
                "date,ticker,open,high,low,close,volume\n2025-01-02,AAA,100,101,99,100,1000\n",
                encoding="utf-8",
            )
            benchmark_path.write_text(
                "date,ticker,open,high,low,close,volume\n2025-01-02,SPY,500,501,499,500,1000000\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "query": {
                            "tickers": ["AAA"],
                            "benchmark": "SPY",
                            "start": "2024-01-01",
                            "end": None,
                        },
                        "inputs": {
                            "tickers_file": None,
                            "metadata_file": None,
                            "universe_snapshots_file": None,
                        },
                    }
                ),
                encoding="utf-8",
            )

            fresh_prices = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                    "ticker": ["AAA", "AAA"],
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.0, 101.0],
                    "volume": [1000, 1100],
                }
            )
            fresh_benchmark = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                    "ticker": ["SPY", "SPY"],
                    "open": [500.0, 501.0],
                    "high": [501.0, 502.0],
                    "low": [499.0, 500.0],
                    "close": [500.0, 501.0],
                    "volume": [1000000, 1100000],
                }
            )

            with patch("us_invest_ai.data.download_ohlcv", side_effect=[fresh_prices, fresh_benchmark]) as mocked_download:
                bundle = prepare_market_data_bundle(
                    data_dir=root / "data",
                    tickers=["AAA"],
                    benchmark="SPY",
                    start="2025-01-01",
                    end=None,
                )

            self.assertEqual(mocked_download.call_count, 2)
            self.assertEqual(bundle.provenance["source"], "download")
            self.assertEqual(int(bundle.prices["close"].iloc[-1]), 101)


if __name__ == "__main__":
    unittest.main()

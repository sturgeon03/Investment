from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from us_invest_ai.build_universe_snapshots import build_dynamic_universe_snapshots
from us_invest_ai.data import load_universe_snapshots


def _make_price_frame(
    *,
    start: str = "2024-01-01",
    periods: int = 140,
    tickers: dict[str, dict[str, float]],
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    records: list[dict[str, object]] = []
    for ticker, params in tickers.items():
        price = params.get("price", 100.0)
        volume = params.get("volume", 1_000_000.0)
        drift = params.get("drift", 0.0)
        start_offset = int(params.get("start_offset", 0))
        late_volume = params.get("late_volume")
        late_volume_start = int(params.get("late_volume_start", periods + 1))
        for index, date in enumerate(dates[start_offset:], start=start_offset):
            active_volume = volume
            if late_volume is not None and index >= late_volume_start:
                active_volume = late_volume
            close = price + drift * index
            records.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": active_volume,
                }
            )
    return pd.DataFrame(records)


class DynamicUniverseSnapshotTests(unittest.TestCase):
    def test_snapshot_builder_never_selects_before_enough_age_history(self) -> None:
        prices = _make_price_frame(
            periods=120,
            tickers={
                "AAA": {"price": 100.0, "volume": 1_000_000.0, "start_offset": 0},
                "BBB": {"price": 110.0, "volume": 1_200_000.0, "start_offset": 60},
            },
        )

        snapshots = build_dynamic_universe_snapshots(
            prices,
            snapshot_size=2,
            min_close_price=5.0,
            min_dollar_volume_60=10_000.0,
            min_universe_age_days=120,
        )

        early_bbb = snapshots.loc[
            (snapshots["ticker"] == "BBB")
            & (pd.to_datetime(snapshots["effective_date"]) < pd.Timestamp("2024-08-01"))
        ]
        self.assertTrue(early_bbb.empty)

    def test_snapshot_builder_uses_only_history_available_at_snapshot_date(self) -> None:
        prices = _make_price_frame(
            periods=140,
            tickers={
                "AAA": {
                    "price": 100.0,
                    "volume": 1_000_000.0,
                    "late_volume": 10_000_000.0,
                    "late_volume_start": 90,
                },
                "BBB": {"price": 120.0, "volume": 900_000.0},
            },
        )

        snapshots = build_dynamic_universe_snapshots(
            prices,
            snapshot_size=2,
            min_close_price=5.0,
            min_dollar_volume_60=10_000.0,
            min_universe_age_days=0,
        )

        march_snapshot = snapshots.loc[
            (pd.to_datetime(snapshots["effective_date"]) == pd.Timestamp("2024-03-29"))
            & (snapshots["ticker"] == "AAA")
        ]
        self.assertEqual(len(march_snapshot), 1)
        self.assertAlmostEqual(float(march_snapshot["avg_dollar_volume_60"].iloc[0]), 100_000_000.0, places=3)

    def test_snapshot_builder_respects_filters_and_snapshot_size(self) -> None:
        prices = _make_price_frame(
            periods=140,
            tickers={
                "AAA": {"price": 100.0, "volume": 2_000_000.0},
                "BBB": {"price": 4.0, "volume": 5_000_000.0},
                "CCC": {"price": 90.0, "volume": 100_000.0},
                "DDD": {"price": 95.0, "volume": 1_500_000.0},
            },
        )

        snapshots = build_dynamic_universe_snapshots(
            prices,
            snapshot_size=1,
            min_close_price=5.0,
            min_dollar_volume_60=50_000_000.0,
            min_universe_age_days=0,
        )

        latest_snapshot = snapshots.loc[
            pd.to_datetime(snapshots["effective_date"]) == pd.to_datetime(snapshots["effective_date"]).max()
        ]
        self.assertEqual(len(latest_snapshot), 1)
        self.assertEqual(latest_snapshot["ticker"].iloc[0], "AAA")
        self.assertEqual(int(latest_snapshot["selection_rank"].iloc[0]), 1)

    def test_generated_snapshot_csv_is_consumable_by_existing_loader(self) -> None:
        prices = _make_price_frame(
            periods=140,
            tickers={
                "AAA": {"price": 100.0, "volume": 2_000_000.0},
                "DDD": {"price": 95.0, "volume": 1_500_000.0},
            },
        )
        snapshots = build_dynamic_universe_snapshots(
            prices,
            snapshot_size=2,
            min_close_price=5.0,
            min_dollar_volume_60=50_000_000.0,
            min_universe_age_days=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dynamic_snapshots.csv"
            snapshots.to_csv(path, index=False)
            loaded = load_universe_snapshots(path)

        self.assertListEqual(list(loaded.columns), ["effective_date", "ticker"])
        self.assertGreaterEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()

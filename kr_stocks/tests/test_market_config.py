from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kr_invest_ai.market_config import (
    default_config_dir,
    load_fee_tax_config,
    load_kr_market_adapter_config,
    load_market_assumptions,
    load_ticker_conventions,
    load_trading_calendar_config,
)


class KRMarketConfigTests(unittest.TestCase):
    def test_default_config_dir_points_to_repo_config(self) -> None:
        self.assertEqual(default_config_dir(), Path(__file__).resolve().parents[1] / "config")

    def test_load_default_configs(self) -> None:
        adapter = load_kr_market_adapter_config()

        self.assertEqual(adapter.market_assumptions.market.name, "KRX")
        self.assertEqual(adapter.market_assumptions.market.base_currency, "KRW")
        self.assertEqual(adapter.trading_calendar.calendar.timezone, "Asia/Seoul")
        self.assertEqual(adapter.ticker_conventions.listing_code_digits, 6)
        self.assertIn("KOSPI", adapter.market_assumptions.universe.primary_exchanges)
        self.assertIn("common_stock", adapter.market_assumptions.universe.security_types)

    def test_individual_loaders_accept_custom_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "market_assumptions.yaml").write_text(
                """
market:
  name: TEST
  country: KR
  base_currency: KRW
  timezone: Asia/Seoul
sessions:
  regular_open: "09:00"
  regular_close: "15:30"
  call_auction_open: "08:30"
  call_auction_close: "09:00"
  closing_auction_open: "15:20"
  closing_auction_close: "15:30"
eligibility:
  min_close_price: 500
  min_average_daily_value_20_krw: 1000000
  min_listing_age_days: 100
universe:
  primary_exchanges: [KOSPI]
  security_types: [common_stock]
notes: ["x"]
""".strip(),
                encoding="utf-8",
            )
            (root / "fees_tax.yaml").write_text(
                """
commission:
  broker_name: test
  per_side_bps: 1.0
  min_fee_krw: 100
  max_fee_krw:
tax:
  securities_transaction_tax_bps:
    kospi_common: 15
  local_surtax_bps: 1.5
slippage:
  model: fixed_bps
  fixed_bps: 2.0
funding:
  currency: KRW
  cash_drag_bps_per_year: 10
notes: ["fee"]
""".strip(),
                encoding="utf-8",
            )
            (root / "trading_calendar.yaml").write_text(
                """
calendar:
  timezone: Asia/Seoul
  regular_trading_days: Monday-Friday
  holiday_source: custom
session_rules:
  use_end_of_day_rebalance: true
  allow_half_day_sessions: false
  allow_auction_rebalance: true
holiday_assumptions:
  include_national_holidays: true
  include_market_closures: false
  include_ad_hoc_closures: true
notes: ["calendar"]
""".strip(),
                encoding="utf-8",
            )
            (root / "ticker_conventions.yaml").write_text(
                """
formats:
  listing_code_digits: 6
  exchange_suffix_examples: ["005930.KS"]
venues: [KOSPI]
security_labels:
  common_stock: Common equity
mapping_notes: ["note"]
""".strip(),
                encoding="utf-8",
            )

            market = load_market_assumptions(root / "market_assumptions.yaml")
            fee_tax = load_fee_tax_config(root / "fees_tax.yaml")
            calendar = load_trading_calendar_config(root / "trading_calendar.yaml")
            ticker_conventions = load_ticker_conventions(root / "ticker_conventions.yaml")

        self.assertEqual(market.market.name, "TEST")
        self.assertEqual(fee_tax.commission.per_side_bps, 1.0)
        self.assertEqual(calendar.calendar.holiday_source, "custom")
        self.assertEqual(ticker_conventions.exchange_suffix_examples, ("005930.KS",))


if __name__ == "__main__":
    unittest.main()

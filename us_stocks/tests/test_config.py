from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import invest_ai_core.config as shared_config
from us_invest_ai.config import load_config


class ConfigTests(unittest.TestCase):
    def test_shared_config_module_is_compatible_with_us_wrapper(self) -> None:
        self.assertIs(shared_config.load_config, load_config)

    def test_load_config_parses_scoring_and_workflow_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "test.yaml"
            env_path = root / ".env.test"
            env_path.write_text("DEEPSEEK_API_KEY=x\n", encoding="utf-8")
            config_path.write_text(
                """
data:
  tickers: [AAPL]
  benchmark: SPY
  start: "2024-01-01"
  end:
strategy:
  rebalance: monthly
  top_n: 2
  min_history_days: 20
  trend_filter_mode: soft
  trend_penalty: 0.25
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
scoring:
  provider: openai-compatible
  base_url: https://api.deepseek.com
  model: deepseek-chat
  api_key_env: DEEPSEEK_API_KEY
  timeout_seconds: 45
  temperature: 0.1
  env_file: .env.test
risk:
  capital_base: 50000
  cash_buffer: 0.1
  max_position_weight: 0.2
  min_trade_notional: 150
  allow_fractional_shares: false
eligibility:
  min_close_price: 10
  min_dollar_volume_20: 1000000
  min_universe_age_days: 60
workflow:
  forms: [10-K, 8-K]
  start_date_lookback_days: 90
  limit_per_ticker: 1
  pause_seconds: 0.5
  max_chars: 10000
  min_section_chars: 100
  positions_path: paper/current_positions.csv
  output_root: runs
  apply_paper_orders: true
output:
  data_dir: data
  artifacts_dir: artifacts
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.config_path, config_path.resolve())
        self.assertEqual(config.scoring.provider, "openai-compatible")
        self.assertEqual(config.scoring.timeout_seconds, 45)
        self.assertEqual(config.scoring.env_file, env_path)
        self.assertEqual(config.eligibility.min_close_price, 10.0)
        self.assertEqual(config.eligibility.min_dollar_volume_20, 1_000_000.0)
        self.assertEqual(config.eligibility.min_universe_age_days, 60)
        self.assertEqual(config.workflow.forms, ["10-K", "8-K"])
        self.assertEqual(config.workflow.start_date_lookback_days, 90)
        self.assertTrue(config.workflow.apply_paper_orders)

    def test_load_config_supports_tickers_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            universe_dir = root / "universes"
            config_dir.mkdir(parents=True, exist_ok=True)
            universe_dir.mkdir(parents=True, exist_ok=True)
            tickers_file = universe_dir / "large_cap.txt"
            metadata_file = universe_dir / "large_cap_metadata.csv"
            tickers_file.write_text("AAPL\nMSFT\nNVDA\n", encoding="utf-8")
            metadata_file.write_text("ticker,sector\nAAPL,Technology\nMSFT,Technology\nNVDA,Technology\n", encoding="utf-8")
            config_path = config_dir / "test.yaml"
            config_path.write_text(
                """
data:
  tickers_file: universes/large_cap.txt
  metadata_file: universes/large_cap_metadata.csv
  benchmark: spy
  start: "2024-01-01"
  end:
strategy:
  rebalance: monthly
  top_n: 2
  min_history_days: 20
  trend_filter_mode: soft
  trend_penalty: 0.25
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
eligibility:
  min_close_price: 5
  min_dollar_volume_20: 25000000
  min_universe_age_days: 120
output:
  data_dir: data
  artifacts_dir: artifacts
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.data.tickers, ["AAPL", "MSFT", "NVDA"])
        self.assertEqual(config.data.tickers_file, tickers_file)
        self.assertEqual(config.data.metadata_file, metadata_file)
        self.assertEqual(config.data.benchmark, "SPY")
        self.assertEqual(config.eligibility.min_universe_age_days, 120)

    def test_load_config_supports_universe_snapshots_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            universe_dir = root / "universes"
            config_dir.mkdir(parents=True, exist_ok=True)
            universe_dir.mkdir(parents=True, exist_ok=True)
            snapshots_file = universe_dir / "large_cap_snapshots.csv"
            metadata_file = universe_dir / "large_cap_metadata.csv"
            snapshots_file.write_text(
                (
                    "effective_date,ticker\n"
                    "2024-01-01,AAPL\n"
                    "2024-01-01,MSFT\n"
                    "2025-01-01,AAPL\n"
                    "2025-01-01,MSFT\n"
                    "2025-01-01,NVDA\n"
                ),
                encoding="utf-8",
            )
            metadata_file.write_text(
                "ticker,sector\nAAPL,Technology\nMSFT,Technology\nNVDA,Technology\n",
                encoding="utf-8",
            )
            config_path = config_dir / "test.yaml"
            config_path.write_text(
                """
data:
  universe_snapshots_file: universes/large_cap_snapshots.csv
  metadata_file: universes/large_cap_metadata.csv
  benchmark: spy
  start: "2024-01-01"
  end:
strategy:
  rebalance: monthly
  top_n: 2
  min_history_days: 20
  trend_filter_mode: soft
  trend_penalty: 0.25
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
eligibility:
  min_close_price: 5
  min_dollar_volume_20: 25000000
  min_universe_age_days: 180
output:
  data_dir: data
  artifacts_dir: artifacts
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.data.tickers, ["AAPL", "MSFT", "NVDA"])
        self.assertEqual(config.data.metadata_file, metadata_file)
        self.assertEqual(config.data.universe_snapshots_file, snapshots_file)
        self.assertEqual(config.data.benchmark, "SPY")
        self.assertEqual(config.eligibility.min_dollar_volume_20, 25_000_000.0)


if __name__ == "__main__":
    unittest.main()

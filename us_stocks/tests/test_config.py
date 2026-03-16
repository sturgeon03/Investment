from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from us_invest_ai.config import load_config


class ConfigTests(unittest.TestCase):
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

        self.assertEqual(config.scoring.provider, "openai-compatible")
        self.assertEqual(config.scoring.timeout_seconds, 45)
        self.assertEqual(config.scoring.env_file, env_path)
        self.assertEqual(config.workflow.forms, ["10-K", "8-K"])
        self.assertEqual(config.workflow.start_date_lookback_days, 90)
        self.assertTrue(config.workflow.apply_paper_orders)


if __name__ == "__main__":
    unittest.main()

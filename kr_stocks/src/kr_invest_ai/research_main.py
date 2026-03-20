from __future__ import annotations

import argparse
from datetime import date

from invest_ai_core.config import BacktestConfig
from kr_invest_ai.data_bundle import KRResearchDataRequest
from kr_invest_ai.research import run_kr_research_pipeline, save_kr_research_outputs
from kr_invest_ai.strategy import KRStrategyConfig


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first Korea-market research/backtest lane.")
    parser.add_argument("--tickers", nargs="+", required=True, help="One or more KR tickers or listing codes.")
    parser.add_argument("--price-start-date", required=True, help="Price window start date in YYYY-MM-DD.")
    parser.add_argument("--price-end-date", required=True, help="Price window end date in YYYY-MM-DD.")
    parser.add_argument("--benchmark-ticker", default=None, help="Optional KR benchmark ticker or listing code.")
    parser.add_argument("--filings-start-date", default=None, help="Optional DART window start date in YYYY-MM-DD.")
    parser.add_argument("--filings-end-date", default=None, help="Optional DART window end date in YYYY-MM-DD.")
    parser.add_argument("--corp-code-map-csv", default=None, help="Optional CSV with columns ticker,corp_code.")
    parser.add_argument("--data-dir", default="kr_stocks/data", help="Directory for cached raw data.")
    parser.add_argument("--artifacts-dir", default="kr_stocks/artifacts/research", help="Directory for research artifacts.")
    parser.add_argument("--use-dart", action="store_true", help="Enable DART fetch when corp codes and dates are provided.")
    parser.add_argument("--top-n", type=int, default=3, help="Number of names held per rebalance.")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0, help="Transaction cost in basis points.")
    parser.add_argument("--spread-cost-bps", type=float, default=0.0, help="Half-spread cost in basis points.")
    parser.add_argument("--market-impact-bps", type=float, default=0.0, help="Participation-scaled market impact in basis points.")
    parser.add_argument("--market-impact-exponent", type=float, default=0.5, help="Exponent applied to participation when scaling market impact.")
    parser.add_argument("--liquidity-lookback-days", type=int, default=20, help="Rolling lookback used to estimate tradable dollar volume.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    request = KRResearchDataRequest(
        tickers=tuple(args.tickers),
        price_start_date=date.fromisoformat(args.price_start_date),
        price_end_date=date.fromisoformat(args.price_end_date),
        benchmark_ticker=args.benchmark_ticker,
        filings_start_date=_parse_date(args.filings_start_date),
        filings_end_date=_parse_date(args.filings_end_date),
    )
    run = run_kr_research_pipeline(
        request,
        data_dir=args.data_dir,
        corp_code_map_csv=args.corp_code_map_csv,
        use_dart=args.use_dart,
        strategy_config=KRStrategyConfig(top_n=args.top_n),
        backtest_config=BacktestConfig(
            transaction_cost_bps=args.transaction_cost_bps,
            spread_cost_bps=args.spread_cost_bps,
            market_impact_bps=args.market_impact_bps,
            market_impact_exponent=args.market_impact_exponent,
            liquidity_lookback_days=args.liquidity_lookback_days,
        ),
    )
    output_dir = save_kr_research_outputs(run, output_dir=args.artifacts_dir)

    print(run.backtest_result.summary.to_string(index=False))
    print(f"Saved research artifacts to: {output_dir}")


if __name__ == "__main__":
    main()

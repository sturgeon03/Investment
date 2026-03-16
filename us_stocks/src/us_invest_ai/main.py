from __future__ import annotations

import argparse

from us_invest_ai.config import load_config
from us_invest_ai.pipeline import run_research_pipeline, save_research_outputs
from us_invest_ai.portfolio import (
    build_next_positions,
    build_rebalance_orders,
    build_target_portfolio,
    latest_prices_by_ticker,
    load_current_positions,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the US stocks research pipeline.")
    parser.add_argument(
        "--config",
        default="us_stocks/config/base.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--portfolio-date",
        default=None,
        help="Optional YYYY-MM-DD date used for the target portfolio snapshot.",
    )
    parser.add_argument(
        "--current-positions-csv",
        default=None,
        help="Optional current positions CSV with columns ticker,shares for order generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    run = run_research_pipeline(config)
    target_portfolio = build_target_portfolio(
        target_weights=run.target_weights,
        prices=run.prices,
        risk=config.risk,
        as_of_date=args.portfolio_date,
    )
    recommended_orders = None
    next_positions = None

    if args.current_positions_csv:
        current_positions = load_current_positions(args.current_positions_csv)
        latest_prices = latest_prices_by_ticker(run.prices, as_of_date=args.portfolio_date)
        recommended_orders = build_rebalance_orders(
            target_portfolio,
            current_positions,
            config.risk,
            latest_prices=latest_prices,
        )
        next_positions = build_next_positions(target_portfolio)

    save_research_outputs(
        run=run,
        output_dir=config.output.artifacts_dir,
        data_dir=config.output.data_dir,
        target_portfolio=target_portfolio,
        recommended_orders=recommended_orders,
        next_positions=next_positions,
    )

    print(run.backtest_result.summary.to_string(index=False))
    if not target_portfolio.empty:
        print(f"Target portfolio date: {target_portfolio['date'].iloc[0].date().isoformat()}")
    else:
        print("Target portfolio is empty.")
    print(f"Saved outputs to: {config.output.artifacts_dir}")


if __name__ == "__main__":
    main()

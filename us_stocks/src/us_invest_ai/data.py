from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)

    renamed = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    missing = [column for column in PRICE_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"Missing expected price columns: {missing}")
    return renamed


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        downloaded = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if downloaded.empty:
            raise ValueError(f"No price data returned for {ticker}")

        normalized = _normalize_columns(downloaded)
        normalized = normalized.reset_index().rename(columns={"Date": "date"})
        normalized["ticker"] = ticker
        frames.append(normalized[["date", "ticker", *PRICE_COLUMNS]])

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    return prices.sort_values(["date", "ticker"]).reset_index(drop=True)


def save_prices(prices: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index=False)

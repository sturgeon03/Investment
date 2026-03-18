from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from us_invest_ai.experiment_manifest import sha256_file


PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
YFINANCE_MAX_ATTEMPTS = 3
YFINANCE_BACKOFF_SECONDS = 1.0


@dataclass(slots=True)
class MarketDataBundle:
    prices: pd.DataFrame
    benchmark_prices: pd.DataFrame
    ticker_metadata: pd.DataFrame | None
    universe_snapshots: pd.DataFrame | None
    provenance: dict[str, Any]


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


def _sleep_backoff(base_delay_seconds: float, attempt_number: int) -> None:
    time_to_sleep = max(base_delay_seconds, 0.0) * (2 ** max(attempt_number - 1, 0))
    if time_to_sleep > 0:
        import time

        time.sleep(time_to_sleep)


def _download_single_ticker_with_retry(
    ticker: str,
    start: str,
    end: str | None,
    max_attempts: int = YFINANCE_MAX_ATTEMPTS,
    backoff_seconds: float = YFINANCE_BACKOFF_SECONDS,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
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
            return downloaded
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            _sleep_backoff(backoff_seconds, attempt)

    if last_error is None:
        raise ValueError(f"No price data returned for {ticker}")
    raise last_error


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        downloaded = _download_single_ticker_with_retry(ticker=ticker, start=start, end=end)
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


def load_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_ticker_metadata(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None

    metadata = pd.read_csv(path)
    required = {"ticker", "sector"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Ticker metadata file is missing columns: {sorted(missing)}")

    metadata = metadata.copy()
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    metadata["sector"] = metadata["sector"].astype(str).str.strip().replace("", "UNKNOWN").fillna("UNKNOWN")
    keep_columns = ["ticker", "sector"]
    if "industry" in metadata.columns:
        metadata["industry"] = metadata["industry"].astype(str).str.strip().replace("", "UNKNOWN").fillna("UNKNOWN")
        keep_columns.append("industry")
    return metadata[keep_columns].drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def load_universe_snapshots(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None

    snapshots = pd.read_csv(path)
    required = {"effective_date", "ticker"}
    missing = required.difference(snapshots.columns)
    if missing:
        raise ValueError(f"Universe snapshots file is missing columns: {sorted(missing)}")

    snapshots = snapshots.copy()
    snapshots["effective_date"] = pd.to_datetime(snapshots["effective_date"]).dt.normalize()
    snapshots["ticker"] = snapshots["ticker"].astype(str).str.upper()
    snapshots = snapshots.loc[snapshots["ticker"] != ""].copy()
    if snapshots.empty:
        raise ValueError(f"Universe snapshots file is empty after normalization: {path}")

    return (
        snapshots[["effective_date", "ticker"]]
        .drop_duplicates(subset=["effective_date", "ticker"])
        .sort_values(["effective_date", "ticker"])
        .reset_index(drop=True)
    )


def _path_info(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path),
    }


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    normalized_dates = pd.to_datetime(frame["date"]).dt.normalize()
    return {
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()),
        "start_date": normalized_dates.min().date().isoformat(),
        "end_date": normalized_dates.max().date().isoformat(),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def prepare_market_data_bundle(
    *,
    data_dir: Path,
    tickers: list[str],
    benchmark: str,
    start: str,
    end: str | None,
    tickers_file: Path | None = None,
    metadata_file: Path | None = None,
    universe_snapshots_file: Path | None = None,
    prefer_cache: bool = True,
) -> MarketDataBundle:
    raw_dir = data_dir / "raw"
    prices_path = raw_dir / "prices.csv"
    benchmark_path = raw_dir / "benchmark.csv"
    manifest_path = raw_dir / "market_data_manifest.json"

    requested_inputs = {
        "tickers_file": _path_info(tickers_file),
        "metadata_file": _path_info(metadata_file),
        "universe_snapshots_file": _path_info(universe_snapshots_file),
    }
    requested_query = {
        "tickers": tickers,
        "benchmark": benchmark,
        "start": start,
        "end": end,
    }
    cached_manifest = _load_json(manifest_path)
    used_cache = (
        prefer_cache
        and prices_path.exists()
        and benchmark_path.exists()
        and cached_manifest is not None
        and cached_manifest.get("query") == requested_query
        and cached_manifest.get("inputs") == requested_inputs
    )
    if used_cache:
        prices = load_prices(prices_path)
        benchmark_prices = load_prices(benchmark_path)
        source = "cache"
    else:
        prices = download_ohlcv(
            tickers=tickers,
            start=start,
            end=end,
        )
        benchmark_prices = download_ohlcv(
            tickers=[benchmark],
            start=start,
            end=end,
        )
        save_prices(prices, prices_path)
        save_prices(benchmark_prices, benchmark_path)
        source = "download"

    ticker_metadata = load_ticker_metadata(metadata_file)
    universe_snapshots = load_universe_snapshots(universe_snapshots_file)

    provenance: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "query": requested_query,
        "inputs": requested_inputs,
        "prices_file": _path_info(prices_path),
        "benchmark_file": _path_info(benchmark_path),
        "prices_summary": _frame_summary(prices),
        "benchmark_summary": _frame_summary(benchmark_prices),
        "metadata_rows": int(len(ticker_metadata)) if ticker_metadata is not None else 0,
        "snapshot_rows": int(len(universe_snapshots)) if universe_snapshots is not None else 0,
    }
    _save_json(manifest_path, provenance)
    provenance["manifest_path"] = str(manifest_path)
    provenance["manifest_sha256"] = sha256_file(manifest_path)

    return MarketDataBundle(
        prices=prices,
        benchmark_prices=benchmark_prices,
        ticker_metadata=ticker_metadata,
        universe_snapshots=universe_snapshots,
        provenance=provenance,
    )

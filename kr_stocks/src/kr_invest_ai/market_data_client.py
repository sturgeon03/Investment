from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import pandas as pd

from kr_invest_ai.market_config import KRMarketAdapterConfig, load_kr_market_adapter_config
from kr_invest_ai.tickers import CanonicalKRTicker, canonicalize_kr_ticker


YAHOO_CHART_ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


class KRMarketDataError(RuntimeError):
    """Raised when the KR market-data adapter cannot parse or fetch a response."""


@dataclass(frozen=True, slots=True)
class KRDailyOHLCVRequest:
    ticker: str
    start_date: date
    end_date: date
    interval: str = "1d"
    include_adjusted_close: bool = True
    default_vendor_suffix: str | None = None

    def canonical_ticker(self, adapter_config: KRMarketAdapterConfig | None = None) -> CanonicalKRTicker:
        conventions = adapter_config.ticker_conventions if adapter_config else None
        canonical = canonicalize_kr_ticker(self.ticker, conventions=conventions)
        if canonical.vendor_suffix or not self.default_vendor_suffix:
            return canonical
        return CanonicalKRTicker(
            listing_code=canonical.listing_code,
            vendor_suffix=self.default_vendor_suffix,
        )

    def vendor_symbol(self, adapter_config: KRMarketAdapterConfig | None = None) -> str:
        return self.canonical_ticker(adapter_config=adapter_config).vendor_ticker

    def to_query_params(self) -> dict[str, str]:
        period_start = datetime.combine(self.start_date, datetime.min.time(), tzinfo=UTC)
        # Yahoo period2 is exclusive; move one day forward to include the end date.
        period_end = datetime.combine(self.end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
        return {
            "period1": str(int(period_start.timestamp())),
            "period2": str(int(period_end.timestamp())),
            "interval": self.interval,
            "includeAdjustedClose": str(self.include_adjusted_close).lower(),
            "events": "div,splits",
        }


@dataclass(frozen=True, slots=True)
class NormalizedDailyOHLCVBar:
    date: date
    canonical_ticker: CanonicalKRTicker
    provider_symbol: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None
    volume: float
    provider: str
    currency: str
    exchange_timezone: str


class KRHistoricalMarketDataClient:
    def __init__(
        self,
        *,
        base_url_template: str = YAHOO_CHART_ENDPOINT,
        provider_name: str = "yahoo_chart",
        adapter_config: KRMarketAdapterConfig | None = None,
    ) -> None:
        self.base_url_template = base_url_template
        self.provider_name = provider_name
        self.adapter_config = adapter_config or load_kr_market_adapter_config()

    def fetch_daily_bars(self, request: KRDailyOHLCVRequest) -> list[NormalizedDailyOHLCVBar]:
        canonical = request.canonical_ticker(adapter_config=self.adapter_config)
        payload = self._request_json(
            request.vendor_symbol(adapter_config=self.adapter_config),
            request.to_query_params(),
        )
        return self._parse_chart_payload(payload, canonical, request)

    def fetch_daily_frame(self, request: KRDailyOHLCVRequest) -> pd.DataFrame:
        return bars_to_frame(self.fetch_daily_bars(request))

    def _request_json(self, provider_symbol: str, params: dict[str, str]) -> dict[str, Any]:
        url = self.base_url_template.format(symbol=provider_symbol)
        query = urlencode(params)
        with urlopen(f"{url}?{query}") as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_chart_payload(
        self,
        payload: dict[str, Any],
        canonical_ticker: CanonicalKRTicker,
        request: KRDailyOHLCVRequest,
    ) -> list[NormalizedDailyOHLCVBar]:
        chart = payload.get("chart")
        if not isinstance(chart, dict):
            raise KRMarketDataError("Missing chart payload.")

        error = chart.get("error")
        if error:
            raise KRMarketDataError(f"Provider returned an error payload: {error}")

        results = chart.get("result")
        if not isinstance(results, list) or not results:
            raise KRMarketDataError("Provider returned no result rows.")

        result = results[0]
        timestamps = result.get("timestamp")
        indicators = result.get("indicators")
        if not isinstance(timestamps, list) or not isinstance(indicators, dict):
            raise KRMarketDataError("Provider response is missing timestamps or indicators.")

        quotes = indicators.get("quote")
        if not isinstance(quotes, list) or not quotes or not isinstance(quotes[0], dict):
            raise KRMarketDataError("Provider response is missing quote rows.")

        quote = quotes[0]
        adjclose_block = indicators.get("adjclose")
        adjclose_values: list[Any] | None = None
        if isinstance(adjclose_block, list) and adjclose_block and isinstance(adjclose_block[0], dict):
            raw_adj = adjclose_block[0].get("adjclose")
            if isinstance(raw_adj, list):
                adjclose_values = raw_adj

        timezone_name = str(
            result.get("meta", {}).get("exchangeTimezoneName")
            or self.adapter_config.market_assumptions.market.timezone
        )
        exchange_timezone = ZoneInfo(timezone_name)
        currency = str(
            result.get("meta", {}).get("currency")
            or self.adapter_config.market_assumptions.market.base_currency
        )

        bars: list[NormalizedDailyOHLCVBar] = []
        for index, timestamp in enumerate(timestamps):
            row = {
                "open": _value_at(quote.get("open"), index),
                "high": _value_at(quote.get("high"), index),
                "low": _value_at(quote.get("low"), index),
                "close": _value_at(quote.get("close"), index),
                "volume": _value_at(quote.get("volume"), index),
            }
            if any(value is None for value in row.values()):
                continue

            adj_close = _value_at(adjclose_values, index)
            trade_date = datetime.fromtimestamp(int(timestamp), tz=UTC).astimezone(exchange_timezone).date()
            bars.append(
                NormalizedDailyOHLCVBar(
                    date=trade_date,
                    canonical_ticker=canonical_ticker,
                    provider_symbol=request.vendor_symbol(adapter_config=self.adapter_config),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    adj_close=None if adj_close is None else float(adj_close),
                    volume=float(row["volume"]),
                    provider=self.provider_name,
                    currency=currency,
                    exchange_timezone=timezone_name,
                )
            )
        return bars


def bars_to_frame(bars: list[NormalizedDailyOHLCVBar]) -> pd.DataFrame:
    rows = []
    for bar in bars:
        row = asdict(bar)
        row["ticker"] = bar.canonical_ticker.vendor_ticker
        row["listing_code"] = bar.canonical_ticker.listing_code
        row["vendor_suffix"] = bar.canonical_ticker.vendor_suffix
        row.pop("canonical_ticker")
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "listing_code",
                "vendor_suffix",
                "provider_symbol",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "provider",
                "currency",
                "exchange_timezone",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.sort_values("date").reset_index(drop=True)


def _value_at(values: list[Any] | None, index: int) -> Any:
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]

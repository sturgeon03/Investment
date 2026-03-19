from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "config"


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping at {path}")
    return payload


@dataclass(frozen=True, slots=True)
class MarketIdentity:
    name: str
    country: str
    base_currency: str
    timezone: str


@dataclass(frozen=True, slots=True)
class SessionConfig:
    regular_open: str
    regular_close: str
    call_auction_open: str
    call_auction_close: str
    closing_auction_open: str
    closing_auction_close: str


@dataclass(frozen=True, slots=True)
class EligibilityConfig:
    min_close_price: int
    min_average_daily_value_20_krw: int
    min_listing_age_days: int


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    primary_exchanges: tuple[str, ...]
    security_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MarketAssumptions:
    market: MarketIdentity
    sessions: SessionConfig
    eligibility: EligibilityConfig
    universe: UniverseConfig
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommissionConfig:
    broker_name: str
    per_side_bps: float
    min_fee_krw: int
    max_fee_krw: int | None


@dataclass(frozen=True, slots=True)
class TaxConfig:
    securities_transaction_tax_bps: dict[str, float]
    local_surtax_bps: float


@dataclass(frozen=True, slots=True)
class SlippageConfig:
    model: str
    fixed_bps: float


@dataclass(frozen=True, slots=True)
class FundingConfig:
    currency: str
    cash_drag_bps_per_year: float


@dataclass(frozen=True, slots=True)
class FeeTaxConfig:
    commission: CommissionConfig
    tax: TaxConfig
    slippage: SlippageConfig
    funding: FundingConfig
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CalendarConfig:
    timezone: str
    regular_trading_days: str
    holiday_source: str


@dataclass(frozen=True, slots=True)
class SessionRuleConfig:
    use_end_of_day_rebalance: bool
    allow_half_day_sessions: bool
    allow_auction_rebalance: bool


@dataclass(frozen=True, slots=True)
class HolidayAssumptionConfig:
    include_national_holidays: bool
    include_market_closures: bool
    include_ad_hoc_closures: bool


@dataclass(frozen=True, slots=True)
class TradingCalendarConfig:
    calendar: CalendarConfig
    session_rules: SessionRuleConfig
    holiday_assumptions: HolidayAssumptionConfig
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TickerConventions:
    listing_code_digits: int
    exchange_suffix_examples: tuple[str, ...]
    venues: tuple[str, ...]
    security_labels: dict[str, str]
    mapping_notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KRMarketAdapterConfig:
    market_assumptions: MarketAssumptions
    fee_tax: FeeTaxConfig
    trading_calendar: TradingCalendarConfig
    ticker_conventions: TickerConventions


def load_market_assumptions(path: Path | None = None) -> MarketAssumptions:
    config_path = path or (default_config_dir() / "market_assumptions.yaml")
    payload = _load_yaml_dict(config_path)
    market = payload["market"]
    sessions = payload["sessions"]
    eligibility = payload["eligibility"]
    universe = payload["universe"]
    return MarketAssumptions(
        market=MarketIdentity(
            name=str(market["name"]),
            country=str(market["country"]),
            base_currency=str(market["base_currency"]),
            timezone=str(market["timezone"]),
        ),
        sessions=SessionConfig(
            regular_open=str(sessions["regular_open"]),
            regular_close=str(sessions["regular_close"]),
            call_auction_open=str(sessions["call_auction_open"]),
            call_auction_close=str(sessions["call_auction_close"]),
            closing_auction_open=str(sessions["closing_auction_open"]),
            closing_auction_close=str(sessions["closing_auction_close"]),
        ),
        eligibility=EligibilityConfig(
            min_close_price=int(eligibility["min_close_price"]),
            min_average_daily_value_20_krw=int(eligibility["min_average_daily_value_20_krw"]),
            min_listing_age_days=int(eligibility["min_listing_age_days"]),
        ),
        universe=UniverseConfig(
            primary_exchanges=tuple(str(value) for value in universe["primary_exchanges"]),
            security_types=tuple(str(value) for value in universe["security_types"]),
        ),
        notes=tuple(str(value) for value in payload.get("notes", [])),
    )


def load_fee_tax_config(path: Path | None = None) -> FeeTaxConfig:
    config_path = path or (default_config_dir() / "fees_tax.yaml")
    payload = _load_yaml_dict(config_path)
    commission = payload["commission"]
    tax = payload["tax"]
    slippage = payload["slippage"]
    funding = payload["funding"]
    securities_transaction_tax_bps = {
        str(key): float(value)
        for key, value in dict(tax["securities_transaction_tax_bps"]).items()
    }
    return FeeTaxConfig(
        commission=CommissionConfig(
            broker_name=str(commission["broker_name"]),
            per_side_bps=float(commission["per_side_bps"]),
            min_fee_krw=int(commission["min_fee_krw"]),
            max_fee_krw=None if commission["max_fee_krw"] is None else int(commission["max_fee_krw"]),
        ),
        tax=TaxConfig(
            securities_transaction_tax_bps=securities_transaction_tax_bps,
            local_surtax_bps=float(tax["local_surtax_bps"]),
        ),
        slippage=SlippageConfig(
            model=str(slippage["model"]),
            fixed_bps=float(slippage["fixed_bps"]),
        ),
        funding=FundingConfig(
            currency=str(funding["currency"]),
            cash_drag_bps_per_year=float(funding["cash_drag_bps_per_year"]),
        ),
        notes=tuple(str(value) for value in payload.get("notes", [])),
    )


def load_trading_calendar_config(path: Path | None = None) -> TradingCalendarConfig:
    config_path = path or (default_config_dir() / "trading_calendar.yaml")
    payload = _load_yaml_dict(config_path)
    calendar = payload["calendar"]
    session_rules = payload["session_rules"]
    holiday_assumptions = payload["holiday_assumptions"]
    return TradingCalendarConfig(
        calendar=CalendarConfig(
            timezone=str(calendar["timezone"]),
            regular_trading_days=str(calendar["regular_trading_days"]),
            holiday_source=str(calendar["holiday_source"]),
        ),
        session_rules=SessionRuleConfig(
            use_end_of_day_rebalance=bool(session_rules["use_end_of_day_rebalance"]),
            allow_half_day_sessions=bool(session_rules["allow_half_day_sessions"]),
            allow_auction_rebalance=bool(session_rules["allow_auction_rebalance"]),
        ),
        holiday_assumptions=HolidayAssumptionConfig(
            include_national_holidays=bool(holiday_assumptions["include_national_holidays"]),
            include_market_closures=bool(holiday_assumptions["include_market_closures"]),
            include_ad_hoc_closures=bool(holiday_assumptions["include_ad_hoc_closures"]),
        ),
        notes=tuple(str(value) for value in payload.get("notes", [])),
    )


def load_ticker_conventions(path: Path | None = None) -> TickerConventions:
    config_path = path or (default_config_dir() / "ticker_conventions.yaml")
    payload = _load_yaml_dict(config_path)
    formats = payload["formats"]
    return TickerConventions(
        listing_code_digits=int(formats["listing_code_digits"]),
        exchange_suffix_examples=tuple(str(value) for value in formats["exchange_suffix_examples"]),
        venues=tuple(str(value) for value in payload["venues"]),
        security_labels={str(key): str(value) for key, value in dict(payload["security_labels"]).items()},
        mapping_notes=tuple(str(value) for value in payload.get("mapping_notes", [])),
    )


def load_kr_market_adapter_config(config_dir: Path | None = None) -> KRMarketAdapterConfig:
    root = config_dir or default_config_dir()
    return KRMarketAdapterConfig(
        market_assumptions=load_market_assumptions(root / "market_assumptions.yaml"),
        fee_tax=load_fee_tax_config(root / "fees_tax.yaml"),
        trading_calendar=load_trading_calendar_config(root / "trading_calendar.yaml"),
        ticker_conventions=load_ticker_conventions(root / "ticker_conventions.yaml"),
    )

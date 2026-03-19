from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from kr_invest_ai.market_config import MarketAssumptions, TradingCalendarConfig


WEEKDAY_NAME_TO_INDEX = {
    "MONDAY": 0,
    "TUESDAY": 1,
    "WEDNESDAY": 2,
    "THURSDAY": 3,
    "FRIDAY": 4,
    "SATURDAY": 5,
    "SUNDAY": 6,
}


@dataclass(frozen=True, slots=True)
class SessionWindow:
    opens_at: datetime
    closes_at: datetime


def _parse_trading_day_rule(value: str) -> set[int]:
    normalized = str(value).strip().upper()
    if normalized == "MONDAY-FRIDAY":
        return {0, 1, 2, 3, 4}
    if normalized == "MONDAY-SATURDAY":
        return {0, 1, 2, 3, 4, 5}

    parts = [part.strip() for part in normalized.replace("/", ",").split(",") if part.strip()]
    weekdays: set[int] = set()
    for part in parts:
        if part not in WEEKDAY_NAME_TO_INDEX:
            raise ValueError(f"Unsupported trading day rule token: {part}")
        weekdays.add(WEEKDAY_NAME_TO_INDEX[part])
    if not weekdays:
        raise ValueError(f"Unsupported trading day rule: {value}")
    return weekdays


def _parse_clock(value: str) -> time:
    hour, minute = str(value).split(":", 1)
    return time(hour=int(hour), minute=int(minute))


def is_regular_trading_day(
    day: date,
    trading_calendar: TradingCalendarConfig,
    holiday_dates: set[date] | None = None,
) -> bool:
    holiday_dates = holiday_dates or set()
    if day in holiday_dates:
        return False
    allowed_weekdays = _parse_trading_day_rule(trading_calendar.calendar.regular_trading_days)
    return day.weekday() in allowed_weekdays


def build_regular_session_window(
    day: date,
    market_assumptions: MarketAssumptions,
    trading_calendar: TradingCalendarConfig,
    holiday_dates: set[date] | None = None,
) -> SessionWindow:
    if not is_regular_trading_day(day, trading_calendar, holiday_dates=holiday_dates):
        raise ValueError(f"{day.isoformat()} is not a regular trading day under the current calendar.")

    timezone = ZoneInfo(market_assumptions.market.timezone)
    open_time = _parse_clock(market_assumptions.sessions.regular_open)
    close_time = _parse_clock(market_assumptions.sessions.regular_close)
    opens_at = datetime.combine(day, open_time, tzinfo=timezone)
    closes_at = datetime.combine(day, close_time, tzinfo=timezone)
    return SessionWindow(opens_at=opens_at, closes_at=closes_at)


def align_filing_timestamp_to_session_date(
    filing_timestamp: datetime,
    market_assumptions: MarketAssumptions,
    trading_calendar: TradingCalendarConfig | None = None,
    holiday_dates: set[date] | None = None,
) -> date:
    timezone = ZoneInfo(market_assumptions.market.timezone)
    localized = filing_timestamp.astimezone(timezone)
    close_time = _parse_clock(market_assumptions.sessions.regular_close)
    session_date = localized.date()
    if localized.timetz().replace(tzinfo=None) > close_time:
        session_date = session_date + timedelta(days=1)
    if trading_calendar is None:
        return session_date

    while not is_regular_trading_day(session_date, trading_calendar, holiday_dates=holiday_dates):
        session_date = session_date + timedelta(days=1)
    return session_date

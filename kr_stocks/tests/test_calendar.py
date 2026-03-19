from __future__ import annotations

import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from kr_invest_ai.calendar import (
    align_filing_timestamp_to_session_date,
    build_regular_session_window,
    is_regular_trading_day,
)
from kr_invest_ai.market_config import load_kr_market_adapter_config


class KRCalendarTests(unittest.TestCase):
    def test_regular_trading_day_uses_weekday_rule(self) -> None:
        adapter = load_kr_market_adapter_config()

        self.assertTrue(is_regular_trading_day(date(2026, 3, 20), adapter.trading_calendar))
        self.assertFalse(is_regular_trading_day(date(2026, 3, 21), adapter.trading_calendar))

    def test_regular_trading_day_respects_holiday_override(self) -> None:
        adapter = load_kr_market_adapter_config()
        holiday_dates = {date(2026, 3, 20)}

        self.assertFalse(
            is_regular_trading_day(
                date(2026, 3, 20),
                adapter.trading_calendar,
                holiday_dates=holiday_dates,
            )
        )

    def test_build_regular_session_window_uses_market_timezone_and_hours(self) -> None:
        adapter = load_kr_market_adapter_config()

        session = build_regular_session_window(
            date(2026, 3, 20),
            adapter.market_assumptions,
            adapter.trading_calendar,
        )

        self.assertEqual(session.opens_at.tzinfo, ZoneInfo("Asia/Seoul"))
        self.assertEqual(session.opens_at.hour, 9)
        self.assertEqual(session.closes_at.hour, 15)
        self.assertEqual(session.closes_at.minute, 30)

    def test_after_close_filing_rolls_to_next_session_date(self) -> None:
        adapter = load_kr_market_adapter_config()
        filing_time = datetime(2026, 3, 20, 16, 5, tzinfo=ZoneInfo("Asia/Seoul"))

        aligned = align_filing_timestamp_to_session_date(
            filing_time,
            adapter.market_assumptions,
            trading_calendar=adapter.trading_calendar,
        )

        self.assertEqual(aligned, date(2026, 3, 23))

    def test_before_close_filing_stays_same_session_date(self) -> None:
        adapter = load_kr_market_adapter_config()
        filing_time = datetime(2026, 3, 20, 14, 0, tzinfo=ZoneInfo("Asia/Seoul"))

        aligned = align_filing_timestamp_to_session_date(
            filing_time,
            adapter.market_assumptions,
            trading_calendar=adapter.trading_calendar,
        )

        self.assertEqual(aligned, date(2026, 3, 20))


if __name__ == "__main__":
    unittest.main()

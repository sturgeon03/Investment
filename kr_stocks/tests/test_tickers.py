from __future__ import annotations

import unittest

from kr_invest_ai.market_config import load_ticker_conventions
from kr_invest_ai.tickers import canonicalize_kr_ticker, normalize_listing_code


class KRTickerTests(unittest.TestCase):
    def test_normalize_listing_code_zero_pads_numeric_value(self) -> None:
        self.assertEqual(normalize_listing_code("5930"), "005930")

    def test_canonicalize_vendor_ticker_preserves_suffix(self) -> None:
        conventions = load_ticker_conventions()
        ticker = canonicalize_kr_ticker("005930.KS", conventions=conventions)

        self.assertEqual(ticker.listing_code, "005930")
        self.assertEqual(ticker.vendor_suffix, "KS")
        self.assertEqual(ticker.vendor_ticker, "005930.KS")

    def test_canonicalize_plain_listing_code_uses_convention_width(self) -> None:
        conventions = load_ticker_conventions()
        ticker = canonicalize_kr_ticker("35420", conventions=conventions)
        self.assertEqual(ticker.listing_code, "035420")
        self.assertIsNone(ticker.vendor_suffix)

    def test_invalid_listing_code_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_listing_code("ABC123")


if __name__ == "__main__":
    unittest.main()

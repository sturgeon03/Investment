from __future__ import annotations

from dataclasses import dataclass

from kr_invest_ai.market_config import TickerConventions


@dataclass(frozen=True, slots=True)
class CanonicalKRTicker:
    listing_code: str
    vendor_suffix: str | None = None

    @property
    def vendor_ticker(self) -> str:
        return (
            f"{self.listing_code}.{self.vendor_suffix}"
            if self.vendor_suffix
            else self.listing_code
        )


def normalize_listing_code(value: str, listing_code_digits: int = 6) -> str:
    stripped = str(value).strip().upper()
    if not stripped:
        raise ValueError("Listing code cannot be empty.")

    if "." in stripped:
        stripped = stripped.split(".", 1)[0]

    if not stripped.isdigit():
        raise ValueError(f"Listing code must be numeric: {value}")
    if len(stripped) > listing_code_digits:
        raise ValueError(
            f"Listing code exceeds configured digit width {listing_code_digits}: {value}"
        )
    return stripped.zfill(listing_code_digits)


def canonicalize_kr_ticker(
    value: str,
    conventions: TickerConventions | None = None,
) -> CanonicalKRTicker:
    stripped = str(value).strip().upper()
    if not stripped:
        raise ValueError("Ticker cannot be empty.")

    listing_code_digits = conventions.listing_code_digits if conventions else 6
    vendor_suffix = None
    if "." in stripped:
        code_part, suffix = stripped.split(".", 1)
        vendor_suffix = suffix or None
    else:
        code_part = stripped

    listing_code = normalize_listing_code(code_part, listing_code_digits=listing_code_digits)
    return CanonicalKRTicker(listing_code=listing_code, vendor_suffix=vendor_suffix)

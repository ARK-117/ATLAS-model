"""Venue parameters for isolated, single-currency paper runs."""

from dataclasses import dataclass
from datetime import timedelta, timezone


@dataclass(frozen=True, slots=True)
class MarketProfile:
    name: str
    currency: str
    timezone: timezone
    settlement_days: int
    require_trade_date_open_for_settlement: bool

    def amount_field(self, field: str) -> str:
        # The original NEPSE schema names the quote currency in its amount fields.
        if field.endswith("_npr"):
            return f"{field[:-4]}_{self.currency.lower()}"
        return field


NEPSE = MarketProfile("NEPSE", "NPR", timezone(timedelta(hours=5, minutes=45)), 2, True)
NSE_INDIA = MarketProfile("NSE India", "INR", timezone(timedelta(hours=5, minutes=30)), 1, False)

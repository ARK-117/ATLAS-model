"""Read a redacted portfolio snapshot for offline research."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path


from .json_inputs import reject_constant, unique_object
from .markets import NEPSE, MarketProfile


NEPAL_TIME = NEPSE.timezone


@dataclass(frozen=True, slots=True)
class Holding:
    symbol: str
    settled_quantity: int
    sellable_quantity: int | None
    pending_buy_quantity: int


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    as_of: datetime
    available_cash_npr: Decimal | None
    pending_sale_proceeds_npr: Decimal | None
    positions: tuple[Holding, ...]

    @property
    def session_date(self) -> date:
        return self.session_date_for(NEPSE)

    def session_date_for(self, market: MarketProfile) -> date:
        return self.as_of.astimezone(market.timezone).date()


def parse_npr_amount(value: object, name: str) -> Decimal | None:
    return parse_amount(value, name, currency=NEPSE.currency)


def parse_amount(value: object, name: str, *, currency: str) -> Decimal | None:
    if value is None:
        return None
    if (
        not isinstance(value, (int, Decimal)) or isinstance(value, bool)
        or not Decimal(value).is_finite() or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative {currency} amount or null")
    return Decimal(value)


def _quantity(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative whole number")
    return value


def load_portfolio_snapshot(path: Path, *, market: MarketProfile = NEPSE) -> PortfolioSnapshot:
    with path.open(encoding="utf-8-sig") as file:
        data = json.load(file, parse_float=Decimal, object_pairs_hook=unique_object, parse_constant=reject_constant)

    if not isinstance(data, dict) or data.get("currency") != market.currency:
        raise ValueError(f"portfolio must be a JSON object with currency {market.currency}")
    if not isinstance(data.get("as_of"), str):
        raise ValueError("portfolio as_of must be an ISO timestamp with an offset")
    as_of = datetime.fromisoformat(data["as_of"])
    if as_of.tzinfo is None:
        raise ValueError("portfolio as_of must include a UTC offset")
    if not isinstance(data.get("positions"), list):
        raise ValueError("portfolio positions must be a list")

    holdings: dict[str, Holding] = {}
    for position in data["positions"]:
        if not isinstance(position, dict) or not isinstance(position.get("symbol"), str):
            raise ValueError("each position must have a symbol")
        symbol = position["symbol"].strip().upper()
        if not symbol or symbol in holdings:
            raise ValueError(f"portfolio position has an empty or duplicate symbol {symbol!r}")
        settled = _quantity(position.get("settled_quantity"), f"{symbol} settled_quantity")
        pending = _quantity(position.get("pending_buy_quantity"), f"{symbol} pending_buy_quantity")
        sellable = position.get("sellable_quantity")
        if sellable is not None:
            sellable = _quantity(sellable, f"{symbol} sellable_quantity")
            if sellable > settled:
                raise ValueError(f"{symbol} sellable_quantity exceeds settled_quantity")
        holdings[symbol] = Holding(symbol, settled, sellable, pending)

    cash_field = market.amount_field("available_cash_npr")
    proceeds_field = market.amount_field("pending_sale_proceeds_npr")
    return PortfolioSnapshot(
        as_of=as_of,
        available_cash_npr=parse_amount(data.get(cash_field), cash_field, currency=market.currency),
        pending_sale_proceeds_npr=parse_amount(data.get(proceeds_field), proceeds_field, currency=market.currency),
        positions=tuple(holdings.values()),
    )

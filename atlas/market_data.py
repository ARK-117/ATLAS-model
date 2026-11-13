"""Read and validate local daily OHLCV files."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import StringIO
from pathlib import Path

from .markets import NEPSE, MarketProfile


BAR_COLUMNS = (
    "symbol", "session_date", "interval", "open", "high", "low", "close",
    "volume", "currency", "price_basis", "source",
)
CALENDAR_COLUMNS = ("session_date", "status")
MAX_PRICE_DIGITS = 128  # Bound fixed-point expansion before storing canonical observations.


@dataclass(frozen=True, slots=True)
class DailyBar:
    """A completed session bar with an unknown publication time."""

    symbol: str
    session_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    price_basis: str
    source: str
    interval: str = "1d"
    currency: str = "NPR"


def _csv_rows(
    path: Path, columns: tuple[str, ...], expected_sha256: str | None = None,
) -> Iterator[tuple[str, dict[str, str]]]:
    if expected_sha256 is None:
        stream = path.open(encoding="utf-8-sig", newline="")
    else:
        data = path.read_bytes()
        if sha256(data).hexdigest() != expected_sha256:
            raise ValueError(f"{path}: CSV checksum does not match the export manifest")
        stream = StringIO(data.decode("utf-8-sig"), newline="")
    with stream as file:
        reader = csv.reader(file, strict=True)
        try:
            header = next(reader, [])
            if len(header) != len(columns) or set(header) != set(columns):
                raise ValueError(f"{path}: expected columns {', '.join(columns)}")
            for values in reader:
                if not values:
                    continue
                location = f"{path}:{reader.line_num}"
                if len(values) != len(columns) or not all(value.strip() for value in values):
                    raise ValueError(f"{location}: expected a value for every column")
                yield location, dict(zip(header, (value.strip() for value in values)))
        except csv.Error as error:
            raise ValueError(f"{path}:{reader.line_num}: invalid CSV ({error})") from error


def _session_date(value: str, location: str) -> date:
    try:
        session = date.fromisoformat(value)
        if session.isoformat() == value:
            return session
    except ValueError:
        pass
    raise ValueError(f"{location}: session_date must be a valid YYYY-MM-DD date")


def _price(value: str, field: str, location: str) -> Decimal:
    try:
        price = Decimal(value)
        if price.is_finite() and price > 0:
            adjusted = price.adjusted()
            integer_digits = max(1, adjusted + 1)
            fractional_digits = max(0, len(price.as_tuple().digits) - adjusted - 1)
            if integer_digits + fractional_digits > MAX_PRICE_DIGITS:
                raise ValueError(f"{location}: {field} expands beyond {MAX_PRICE_DIGITS} supported price digits")
            return price
    except InvalidOperation:
        pass
    raise ValueError(f"{location}: {field} must be a positive finite price")


def _daily_bar(row: dict[str, str], location: str, currency: str) -> DailyBar:
    if row["interval"] != "1d" or row["currency"] != currency:
        raise ValueError(f"{location}: expected interval 1d and currency {currency}")
    if row["price_basis"] not in {"raw", "adjusted"}:
        raise ValueError(f"{location}: price_basis must be raw or adjusted")
    session = _session_date(row["session_date"], location)
    opening = _price(row["open"], "open", location)
    high = _price(row["high"], "high", location)
    low = _price(row["low"], "low", location)
    close = _price(row["close"], "close", location)
    if low > min(opening, close) or high < max(opening, close):
        raise ValueError(f"{location}: open and close must lie between low and high")
    if not row["volume"].isdecimal():
        raise ValueError(f"{location}: volume must be a nonnegative whole number")
    try:
        volume = int(row["volume"])
    except ValueError as error:
        raise ValueError(f"{location}: volume is too large") from error
    return DailyBar(
        symbol=row["symbol"].upper(),
        session_date=session,
        open=opening,
        high=high,
        low=low,
        close=close,
        volume=volume,
        price_basis=row["price_basis"],
        source=row["source"],
        currency=currency,
    )


def load_session_calendar(
    path: Path, *, expected_sha256: str | None = None,
) -> dict[date, str]:
    sessions: dict[date, str] = {}
    for location, row in _csv_rows(path, CALENDAR_COLUMNS, expected_sha256):
        session = _session_date(row["session_date"], location)
        status = row["status"]
        if status not in {"open", "closed"}:
            raise ValueError(f"{location}: status must be open or closed")
        if session in sessions:
            raise ValueError(f"{location}: duplicate calendar date {session}")
        sessions[session] = status
    return sessions


def next_open_session(after: date, calendar: Mapping[date, str]) -> date:
    session = after
    while True:
        session += timedelta(days=1)
        status = calendar.get(session)
        if status == "open":
            return session
        if status != "closed":
            raise ValueError(f"calendar has no status for {session}")


def _check_calendar(
    path: Path, bars: tuple[DailyBar, ...], by_symbol: dict[str, dict[date, DailyBar]]
) -> None:
    sessions = load_session_calendar(path)
    first, last = bars[0].session_date, bars[-1].session_date
    open_sessions = []
    for offset in range((last - first).days + 1):
        session = first + timedelta(days=offset)
        if session not in sessions:
            raise ValueError(f"{path}: calendar has no status for {session}")
        if sessions[session] == "open":
            open_sessions.append(session)

    for bar in bars:
        if sessions[bar.session_date] == "closed":
            raise ValueError(f"{path}: {bar.symbol} has a bar on closed session {bar.session_date}")

    for symbol, series in by_symbol.items():
        first, last = next(iter(series)), next(reversed(series))
        for session in open_sessions:
            if first < session < last and session not in series:
                raise ValueError(f"{path}: {symbol} is missing a bar on open session {session}")


def load_nepse_daily_bars(
    path: Path, *, calendar: Path | None = None, expected_sha256: str | None = None,
) -> tuple[DailyBar, ...]:
    return load_daily_bars(path, calendar=calendar, expected_sha256=expected_sha256)


def load_daily_bars(
    path: Path, *, calendar: Path | None = None, expected_sha256: str | None = None,
    market: MarketProfile = NEPSE,
) -> tuple[DailyBar, ...]:
    """Return single-currency bars in session and symbol order."""
    by_symbol: dict[str, dict[date, DailyBar]] = {}
    for location, row in _csv_rows(path, BAR_COLUMNS, expected_sha256):
        bar = _daily_bar(row, location, market.currency)
        series = by_symbol.setdefault(bar.symbol, {})
        session = bar.session_date
        if session in series:
            raise ValueError(f"{location}: duplicate bar for {bar.symbol} on {session}")
        if series and session < next(reversed(series)):
            raise ValueError(f"{location}: dates for {bar.symbol} must be in ascending order")
        series[session] = bar

    if not by_symbol:
        raise ValueError(f"{path}: expected at least one daily bar")
    bars = tuple(sorted(
        (bar for series in by_symbol.values() for bar in series.values()),
        key=lambda bar: (bar.session_date, bar.symbol),
    ))
    if calendar is not None:
        _check_calendar(calendar, bars, by_symbol)
    return bars

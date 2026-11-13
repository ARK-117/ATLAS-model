"""Replay completed market observations in exchange-session order."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Context, Decimal, localcontext
from itertools import groupby

from .market_data import DailyBar


LOOKBACK = 5


@dataclass(frozen=True, slots=True)
class DailyFeatures:
    bar: DailyBar
    return_from_previous_close: Decimal | None
    prior_high: Decimal | None
    prior_low: Decimal | None
    relative_volume: Decimal | None


@dataclass(frozen=True, slots=True)
class SessionFeatures:
    session_date: date
    features: tuple[DailyFeatures, ...]
    lookbacks: dict[str, tuple[DailyBar, ...]] | None = None


def _features_for(bar: DailyBar, history: deque[DailyBar]) -> DailyFeatures:
    if history and history[-1].currency != bar.currency:
        raise ValueError(f"{bar.symbol} changes currency on {bar.session_date}")
    if history and history[-1].price_basis != bar.price_basis:
        raise ValueError(f"{bar.symbol} changes price basis on {bar.session_date}")

    with localcontext(Context(prec=28)):
        previous_return = bar.close / history[-1].close - 1 if history else None
        if len(history) < LOOKBACK:
            return DailyFeatures(bar, previous_return, None, None, None)

        total_volume = sum(previous.volume for previous in history)
        relative_volume = Decimal(bar.volume * LOOKBACK) / total_volume if total_volume else None
        return DailyFeatures(
            bar=bar,
            return_from_previous_close=previous_return,
            prior_high=max(previous.high for previous in history),
            prior_low=min(previous.low for previous in history),
            relative_volume=relative_volume,
        )


def replay_daily_bars(
    bars: Iterable[DailyBar], *, lookback_on: date | None = None, retain_lookbacks: bool = False,
) -> Iterator[SessionFeatures]:
    """Advance the logical clock after each completed session, with no assumed wall time."""
    histories: dict[str, deque[DailyBar]] = {}
    previous_key: tuple[date, str] | None = None
    for session, session_bars in groupby(bars, key=lambda bar: bar.session_date):
        features = []
        lookbacks = {} if retain_lookbacks or session == lookback_on else None
        for bar in session_bars:
            key = (session, bar.symbol)
            if previous_key is not None and key <= previous_key:
                raise ValueError("replay bars must be unique and ordered by session date and symbol")
            history = histories.setdefault(bar.symbol, deque(maxlen=LOOKBACK))
            features.append(_features_for(bar, history))
            if lookbacks is not None:
                lookbacks[bar.symbol] = tuple(history)
            history.append(bar)
            previous_key = key
        yield SessionFeatures(session, tuple(features), lookbacks)

"""Generate reviewable long-only proposals from completed market sessions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Context, Decimal, localcontext
from itertools import takewhile
from typing import Literal

from .market_data import DailyBar
from .markets import NEPSE, MarketProfile
from .portfolio import Holding, PortfolioSnapshot
from .replay import LOOKBACK, DailyFeatures, SessionFeatures, replay_daily_bars


VOLUME_THRESHOLD = Decimal("1.5")
COST_BUFFER = Decimal("0.01")
FORECAST_HURDLE = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class Proposal:
    symbol: str
    action: Literal["ADD", "TRIM", "EXIT", "HOLD"]
    quantity: int
    reason: str
    reference_close_npr: Decimal | None = None
    illustrative_cost_npr: Decimal | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class PortfolioReview:
    as_of: datetime
    session_date: date
    available_cash_npr: Decimal | None
    strategy: str
    proposals: tuple[Proposal, ...]


def _decide(
    symbol: str,
    features: DailyFeatures | None,
    holding: Holding | None,
    remaining_cash: Decimal | None,
    hold_only: bool,
    entry_quantity: int,
    forecasts: Mapping[str, Decimal] | None,
    trace: dict | None,
) -> Proposal:
    if features is None:
        return Proposal(symbol, "HOLD", 0, "no bar for the portfolio session")

    bar = features.bar
    price = bar.close

    def hold(reason: str) -> Proposal:
        return Proposal(symbol, "HOLD", 0, reason, price, source=bar.source)

    if hold_only:
        return hold("no-trade control")
    if bar.price_basis != "raw" or bar.volume == 0:
        return hold("a raw price and positive session volume are required")
    if features.prior_high is None or features.prior_low is None:
        return hold("need five preceding observations")

    if forecasts is None:
        exit_signal = price < features.prior_low
        above_high = price > features.prior_high
        volume_passes = features.relative_volume is not None and features.relative_volume >= VOLUME_THRESHOLD
        enter_signal = above_high and volume_passes
        if trace is not None:
            trace.update(close_above_prior_high=above_high, relative_volume_passes=volume_passes)
        exit_reason = "close below the prior five-bar low"
        entry_reason = "close above prior high with elevated volume"
    else:
        forecast = forecasts.get(symbol)
        if trace is not None:
            trace["forecast_return"] = forecast
        if forecast is None:
            return hold("no point-in-time forecast")
        exit_signal = forecast < -FORECAST_HURDLE
        enter_signal = forecast > FORECAST_HURDLE
        exit_reason = "forecast below the negative research hurdle"
        entry_reason = "forecast above the positive research hurdle"

    if trace is not None:
        trace.update(entry_signal=enter_signal, exit_signal=exit_signal)

    if holding and holding.settled_quantity and exit_signal:
        quantity = holding.sellable_quantity
        if not quantity:
            return hold("sellable inventory is zero or unknown")
        can_exit = quantity == holding.settled_quantity and not holding.pending_buy_quantity
        return Proposal(
            symbol, "EXIT" if can_exit else "TRIM", quantity,
            exit_reason, price, price * quantity * COST_BUFFER, bar.source,
        )

    if enter_signal:
        if holding and (holding.settled_quantity or holding.pending_buy_quantity):
            return hold("baseline adds only to new positions")
        if entry_quantity <= 0:
            return hold("entry lot size is unavailable")
        notional = price * entry_quantity
        cost = notional * COST_BUFFER
        if trace is not None:
            trace.update(entry_notional_npr=notional, entry_buffer_npr=cost, entry_cash_needed_npr=notional + cost)
        if remaining_cash is None or notional + cost > remaining_cash:
            return hold("available cash cannot cover the entry lot and the illustrative cost buffer")
        return Proposal(symbol, "ADD", entry_quantity, entry_reason, price, cost, bar.source)

    return hold("no baseline signal")


def review_portfolio(
    snapshot: PortfolioSnapshot, bars: Iterable[DailyBar], *, hold_only: bool = False,
    trace: dict | None = None, market: MarketProfile = NEPSE, entry_lots: Mapping[str, int] | None = None,
) -> PortfolioReview:
    """Review the completed session named by the snapshot in the venue's local time."""
    signal = snapshot.session_date_for(market)
    historical_bars = takewhile(lambda bar: bar.session_date <= signal, bars)
    lookback_on = signal if trace is not None else None
    session = next(
        (batch for batch in replay_daily_bars(historical_bars, lookback_on=lookback_on)
         if batch.session_date == signal),
        None,
    )
    if session is None:
        raise ValueError(f"no completed bars match portfolio session {signal}")

    return review_session(snapshot, session, hold_only=hold_only, trace=trace, market=market, entry_lots=entry_lots)


def review_session(
    snapshot: PortfolioSnapshot,
    session: SessionFeatures,
    *,
    hold_only: bool = False,
    entry_lots: Mapping[str, int] | None = None,
    forecasts: Mapping[str, Decimal] | None = None,
    trace: dict | None = None,
    market: MarketProfile = NEPSE,
) -> PortfolioReview:
    if session.session_date != snapshot.session_date_for(market):
        raise ValueError("portfolio snapshot and features must match the session date")
    if any(item.bar.currency != market.currency for item in session.features):
        raise ValueError(f"signal bars must use currency {market.currency}")

    holdings = {holding.symbol: holding for holding in snapshot.positions}
    features = {item.bar.symbol: item for item in session.features}
    remaining_cash = snapshot.available_cash_npr
    proposals = []
    if trace is not None:
        trace["strategy"] = {
            "lookback_bars_required": LOOKBACK, "minimum_relative_volume": VOLUME_THRESHOLD,
            "entry_cost_buffer_fraction": COST_BUFFER, "forecast_return_hurdle": FORECAST_HURDLE,
            "symbols_in_review_order": {},
        }
    with localcontext(Context(prec=28)):
        for symbol in sorted(holdings.keys() | features.keys()):
            entry_quantity = 1 if entry_lots is None else entry_lots.get(symbol, 0)
            item, holding = features.get(symbol), holdings.get(symbol)
            detail = None
            if trace is not None:
                prior = (session.lookbacks or {}).get(symbol, ())
                detail = {
                    "features": asdict(item) if item is not None else None,
                    "preceding_bars": [asdict(bar) for bar in prior],
                    "mean_preceding_volume": Decimal(sum(bar.volume for bar in prior)) / len(prior)
                    if len(prior) == LOOKBACK else None,
                    "holding": asdict(holding) if holding is not None else None,
                    "cash_before_npr": remaining_cash,
                    "configured_entry_quantity": entry_quantity,
                }
                trace["strategy"]["symbols_in_review_order"][symbol] = detail
            proposal = _decide(symbol, item, holding, remaining_cash, hold_only, entry_quantity, forecasts, detail)
            proposals.append(proposal)
            if proposal.action == "ADD":
                assert remaining_cash is not None and proposal.reference_close_npr is not None
                assert proposal.illustrative_cost_npr is not None
                remaining_cash -= proposal.reference_close_npr * proposal.quantity + proposal.illustrative_cost_npr
            if detail is not None:
                detail.update(proposal=asdict(proposal), cash_after_npr=remaining_cash)

    return PortfolioReview(
        as_of=snapshot.as_of,
        session_date=session.session_date,
        available_cash_npr=snapshot.available_cash_npr,
        strategy="hold-only" if hold_only else "five-bar-breakout-v1" if forecasts is None else "regression-experiment",
        proposals=tuple(proposals),
    )

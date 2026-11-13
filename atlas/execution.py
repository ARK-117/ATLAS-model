"""Simulate a reviewed batch of next-session paper orders."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal, localcontext
from pathlib import Path

from .json_inputs import reject_constant, unique_object
from .market_data import DailyBar, next_open_session
from .markets import NEPSE, MarketProfile
from .paper_risk import PaperRiskInputs, assess_paper_review, settlement_session
from .portfolio import Holding, PortfolioSnapshot, parse_amount
from .strategy import PortfolioReview, Proposal


CENT = Decimal("0.01")
BPS = Decimal(10000)
ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class ExecutionTerms:
    effective_from: date
    commission_bps: Decimal
    minimum_commission_npr: Decimal
    regulatory_bps: Decimal
    sell_transfer_fee_npr: Decimal
    spread_bps: Decimal
    slippage_bps: Decimal
    max_volume_fraction: Decimal

    def fee(self, notional: Decimal, *, buy: bool, trace: dict | None = None) -> Decimal:
        commission = max(self.minimum_commission_npr, notional * self.commission_bps / BPS)
        regulatory = notional * self.regulatory_bps / BPS
        transfer = ZERO if buy else self.sell_transfer_fee_npr
        total = (commission + regulatory + transfer).quantize(CENT, rounding=ROUND_CEILING)
        if trace is not None:
            trace["fees"] = {
                "commission_npr": commission, "regulatory_npr": regulatory,
                "transfer_npr": transfer, "total_rounded_up_npr": total,
            }
        return total


@dataclass(frozen=True, slots=True)
class PaperOrder:
    proposal: Proposal
    cash_reserved_npr: Decimal = ZERO
    shares_reserved: int = 0


@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    side: str
    quantity: int
    price_npr: Decimal
    fee_npr: Decimal
    trade_date: date
    settlement_date: date

    @property
    def notional_npr(self) -> Decimal:
        return self.price_npr * self.quantity


@dataclass(frozen=True, slots=True)
class OrderOutcome:
    symbol: str
    status: str
    reason: str
    quantity: int
    fill: Fill | None = None


@dataclass(frozen=True, slots=True)
class LedgerSession:
    session_date: date
    available_cash_npr: Decimal
    simulated_pending_sale_proceeds_npr: Decimal
    settled_fills: tuple[Fill, ...]
    positions: tuple[Holding, ...]


@dataclass(frozen=True, slots=True)
class SimulationReport:
    signal_date: date
    eligible_session: date
    terms_effective_from: date
    reserved_cash_npr: Decimal
    reserved_shares: dict[str, int]
    legacy_pending_sale_proceeds_npr: Decimal | None
    outcomes: tuple[OrderOutcome, ...]
    sessions: tuple[LedgerSession, ...]


def mark_paper_portfolio(
    snapshot: PortfolioSnapshot, marks: Mapping[str, DailyBar], unsettled: Sequence[Fill],
) -> tuple[Decimal, Decimal]:
    sold: defaultdict[str, int] = defaultdict(int)
    proceeds = ZERO
    for fill in unsettled:
        if fill.side == "sell":
            sold[fill.symbol] += fill.quantity
            proceeds += fill.notional_npr - fill.fee_npr

    if snapshot.available_cash_npr is None:
        raise ValueError("portfolio valuation requires available cash")
    values = []
    for holding in snapshot.positions:
        quantity = holding.settled_quantity + holding.pending_buy_quantity - sold[holding.symbol]
        if quantity == 0:
            continue
        bar = marks.get(holding.symbol)
        if quantity < 0 or bar is None or bar.price_basis != "raw":
            raise ValueError(f"missing or invalid held mark for {holding.symbol} on {snapshot.session_date}")
        values.append(bar.close * quantity)
    equity = snapshot.available_cash_npr + proceeds + sum(values, ZERO)
    if equity <= 0:
        raise ValueError("portfolio equity must stay positive")
    return equity, max(values, default=ZERO) / equity


def load_execution_terms(path: Path, *, market: MarketProfile = NEPSE) -> tuple[ExecutionTerms, ...]:
    with path.open(encoding="utf-8-sig") as file:
        rows = json.load(file, parse_float=Decimal, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(rows, list) or not rows:
        raise ValueError("execution assumptions must be a nonempty list")

    schedule: dict[date, ExecutionTerms] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("effective_from"), str):
            raise ValueError("each execution schedule entry requires an effective_from date")
        if row.get("currency", NEPSE.currency) != market.currency:
            raise ValueError(f"execution schedule must use currency {market.currency}")
        effective = date.fromisoformat(row["effective_from"])
        if effective in schedule:
            raise ValueError(f"duplicate execution schedule date {effective}")

        amounts = {}
        for field in ExecutionTerms.__dataclass_fields__:
            if field == "effective_from":
                continue
            input_field = market.amount_field(field)
            value = parse_amount(row.get(input_field), input_field, currency=market.currency)
            if value is None:
                raise ValueError(f"execution schedule requires {input_field}")
            amounts[field] = value
        terms = ExecutionTerms(effective, **amounts)
        if terms.max_volume_fraction > 1 or terms.spread_bps / 2 + terms.slippage_bps >= BPS:
            raise ValueError("execution volume fraction or price friction is out of range")
        schedule[effective] = terms
    return tuple(schedule[effective] for effective in sorted(schedule))


def _try_fill(
    order: PaperOrder, signal_bar: DailyBar, fill_bar: DailyBar | None, lot_size: int,
    terms: ExecutionTerms, max_risk_participation: Decimal, settlement_calendar: Mapping[date, str],
    *, market: MarketProfile, trace: dict | None = None,
) -> OrderOutcome:
    proposal = order.proposal
    symbol = proposal.symbol
    buy = proposal.action == "ADD"
    limit = proposal.reference_close_npr
    assert limit is not None

    def expires(reason: str) -> OrderOutcome:
        return OrderOutcome(symbol, "expired", reason, proposal.quantity)

    if trace is not None:
        trace.update(
            reserved_cash_npr=order.cash_reserved_npr, reserved_shares=order.shares_reserved,
            requested_quantity=proposal.quantity, limit_price_npr=limit, lot_size=lot_size,
            signal_session_volume=signal_bar.volume,
            eligible_session_bar=asdict(fill_bar) if fill_bar is not None else None,
        )
    if fill_bar is None or fill_bar.price_basis != "raw":
        return expires("no raw bar on the eligible session")
    volume = min(signal_bar.volume, fill_bar.volume)
    participation = min(terms.max_volume_fraction, max_risk_participation)
    capacity = int(volume * participation)
    lot_capacity = capacity - capacity % lot_size
    quantity = min(proposal.quantity, lot_capacity)
    if trace is not None:
        trace.update(
            cap_base_volume=volume, effective_participation_fraction=participation,
            volume_capacity_shares=capacity, lot_rounded_capacity_shares=lot_capacity, candidate_quantity=quantity,
        )
    if quantity == 0:
        return expires("volume cap leaves no fillable lot")

    friction = (terms.spread_bps / 2 + terms.slippage_bps) / BPS
    if buy:
        unrounded_price = fill_bar.open * (1 + friction)
        price = unrounded_price.quantize(CENT, rounding=ROUND_CEILING)
        within_limit = price <= limit
    else:
        unrounded_price = fill_bar.open * (1 - friction)
        price = unrounded_price.quantize(CENT, rounding=ROUND_FLOOR)
        within_limit = price >= limit
    within_range = fill_bar.low <= price <= fill_bar.high
    if trace is not None:
        trace.update(
            price_friction_fraction=friction, unrounded_price_npr=unrounded_price, adjusted_price_npr=price,
            within_limit=within_limit, within_observed_range=within_range,
        )
    if not within_limit or not within_range:
        return expires("adjusted opening price fails the limit or observed range")

    notional = price * quantity
    fee = terms.fee(notional, buy=buy, trace=trace)
    if trace is not None:
        trace.update(
            notional_npr=notional,
            cash_debit_if_filled_npr=notional + fee if buy else None,
            settlement_proceeds_if_filled_npr=notional - fee if not buy else None,
        )
    if buy and notional + fee > order.cash_reserved_npr:
        return OrderOutcome(symbol, "rejected", "fill exceeds reserved cash after fees", proposal.quantity)
    if not buy and fee >= notional:
        return OrderOutcome(symbol, "rejected", "fees consume the sale proceeds", proposal.quantity)

    settlement = settlement_session(fill_bar.session_date, settlement_calendar, market=market)
    fill = Fill(symbol, "buy" if buy else "sell", quantity, price, fee, fill_bar.session_date, settlement)
    if quantity < proposal.quantity:
        return OrderOutcome(symbol, "partial_expired", "volume cap limited fill and remainder expired", proposal.quantity, fill)
    return OrderOutcome(symbol, "filled", "next-session opening model", proposal.quantity, fill)


def simulate_paper(
    review: PortfolioReview,
    snapshot: PortfolioSnapshot,
    bars: Sequence[DailyBar],
    risk: PaperRiskInputs,
    schedule: Sequence[ExecutionTerms],
    trading_calendar: Mapping[date, str],
    settlement_calendar: Mapping[date, str],
    *,
    sessions: int = 3,
    cancelled_symbols: frozenset[str] = frozenset(),
    pending_fills: Sequence[Fill] = (),
    trace: dict | None = None,
    market: MarketProfile = NEPSE,
) -> SimulationReport:
    """Run one batch with day-only orders and end-of-day simulated settlement."""
    if sessions < 1 or trading_calendar.get(review.session_date) != "open":
        raise ValueError("simulation needs a positive session count and an open signal date")
    if snapshot.available_cash_npr is None:
        raise ValueError("simulation needs a known available cash balance")

    dates = []
    cursor = review.session_date
    for _ in range(sessions):
        cursor = next_open_session(cursor, trading_calendar)
        dates.append(cursor)
    eligible = dates[0]
    terms = max((item for item in schedule if item.effective_from <= eligible),
                key=lambda item: item.effective_from, default=None)
    if terms is None:
        raise ValueError(f"no execution schedule applies on {eligible}")

    signal_bars = {bar.symbol: bar for bar in bars if bar.session_date == review.session_date}
    fill_bars = {bar.symbol: bar for bar in bars if bar.session_date == eligible}
    if any(bar.currency != market.currency for bar in fill_bars.values()):
        raise ValueError(f"eligible session bars must use currency {market.currency}")
    assessments = assess_paper_review(review, snapshot, bars, risk, trace=trace, market=market)
    proposed_symbols = {proposal.symbol for proposal in review.proposals if proposal.action != "HOLD"}
    if cancelled_symbols - proposed_symbols:
        raise ValueError("cancellation must name an active proposal symbol")

    holdings = {holding.symbol: holding for holding in snapshot.positions}
    outcomes = []
    orders = []
    reserved_shares = {}
    pending = list(pending_fills)
    if any(not fill.trade_date <= review.session_date < fill.settlement_date for fill in pending):
        raise ValueError("carried fills must be unsettled at the signal date")
    ledger = []
    attempts: dict[str, dict] = {}
    if trace is not None:
        trace["execution"] = {
            "terms": asdict(terms), "eligible_session": eligible,
            "cancelled_symbols": sorted(cancelled_symbols),
            "starting_portfolio": asdict(snapshot),
            "carried_unsettled_fills": [asdict(fill) for fill in pending],
            "fill_attempts": attempts,
        }
    with localcontext(Context(prec=28)):
        cash = snapshot.available_cash_npr
        for proposal, assessment in zip(review.proposals, assessments, strict=True):
            symbol = proposal.symbol
            if proposal.action == "HOLD":
                continue
            if assessment.status != "paper_review":
                outcomes.append(OrderOutcome(symbol, "rejected", assessment.reason, proposal.quantity))
                continue
            if symbol in cancelled_symbols:
                outcomes.append(OrderOutcome(symbol, "cancelled", "cancelled before activation", proposal.quantity))
                continue

            if proposal.action == "ADD":
                assert proposal.reference_close_npr is not None
                assert proposal.illustrative_cost_npr is not None
                reserve = proposal.reference_close_npr * proposal.quantity + proposal.illustrative_cost_npr
                cash -= reserve
                orders.append(PaperOrder(proposal, cash_reserved_npr=reserve))
            else:
                holding = holdings[symbol]
                assert holding.sellable_quantity is not None
                holdings[symbol] = replace(holding, sellable_quantity=holding.sellable_quantity - proposal.quantity)
                reserved_shares[symbol] = proposal.quantity
                orders.append(PaperOrder(proposal, shares_reserved=proposal.quantity))

        for order in orders:
            symbol = order.proposal.symbol
            detail = attempts.setdefault(symbol, {}) if trace is not None else None
            outcome = _try_fill(
                order, signal_bars[symbol], fill_bars.get(symbol), risk.instruments[symbol].lot_size,
                terms, risk.limits.max_volume_fraction, settlement_calendar, market=market, trace=detail,
            )
            outcomes.append(outcome)
            fill = outcome.fill
            if order.shares_reserved:
                holding = holdings[symbol]
                assert holding.sellable_quantity is not None
                released = order.shares_reserved - (fill.quantity if fill else 0)
                holdings[symbol] = replace(holding, sellable_quantity=holding.sellable_quantity + released)
            else:
                cash += order.cash_reserved_npr
                if fill:
                    cash -= fill.notional_npr + fill.fee_npr
                    holding = holdings.get(symbol, Holding(symbol, 0, 0, 0))
                    holdings[symbol] = replace(holding, pending_buy_quantity=holding.pending_buy_quantity + fill.quantity)
            if fill:
                pending.append(fill)

        for session in dates:
            settled = tuple(fill for fill in pending if fill.settlement_date <= session)
            pending = [fill for fill in pending if fill.settlement_date > session]
            for fill in settled:
                holding = holdings[fill.symbol]
                if fill.side == "buy":
                    sellable = holding.sellable_quantity
                    holdings[fill.symbol] = replace(
                        holding, settled_quantity=holding.settled_quantity + fill.quantity,
                        sellable_quantity=None if sellable is None else sellable + fill.quantity,
                        pending_buy_quantity=holding.pending_buy_quantity - fill.quantity,
                    )
                else:
                    cash += fill.notional_npr - fill.fee_npr
                    holdings[fill.symbol] = replace(holding, settled_quantity=holding.settled_quantity - fill.quantity)

            pending_sales = sum((fill.notional_npr - fill.fee_npr for fill in pending if fill.side == "sell"), ZERO)
            ledger.append(LedgerSession(
                session, cash, pending_sales, settled, tuple(holdings[symbol] for symbol in sorted(holdings)),
            ))

        return SimulationReport(
            review.session_date, eligible, terms.effective_from, sum((order.cash_reserved_npr for order in orders), ZERO),
            reserved_shares, snapshot.pending_sale_proceeds_npr, tuple(sorted(outcomes, key=lambda item: item.symbol)),
            tuple(ledger),
        )

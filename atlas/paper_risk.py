"""Apply paper risk limits without an order or broker interface."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Context, Decimal, localcontext
from pathlib import Path

from .json_inputs import reject_constant, unique_object
from .market_data import DailyBar, next_open_session
from .markets import NEPSE, MarketProfile
from .portfolio import PortfolioSnapshot, parse_amount
from .strategy import PortfolioReview, Proposal


@dataclass(frozen=True, slots=True)
class PaperLimits:
    max_order_npr: Decimal
    max_symbol_fraction: Decimal
    max_sector_fraction: Decimal
    max_gross_fraction: Decimal
    max_daily_loss_npr: Decimal
    max_drawdown_fraction: Decimal
    max_volume_fraction: Decimal


@dataclass(frozen=True, slots=True)
class InstrumentRules:
    sector: str
    lot_size: int
    tradable: bool


@dataclass(frozen=True, slots=True)
class PaperRiskInputs:
    session_date: date
    kill_switch_active: bool
    equity_npr: Decimal
    daily_loss_npr: Decimal
    drawdown_fraction: Decimal
    limits: PaperLimits
    instruments: Mapping[str, InstrumentRules]


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    symbol: str
    status: str
    reason: str


def _required_amount(data: dict, field: str, market: MarketProfile) -> Decimal:
    field = market.amount_field(field)
    value = parse_amount(data.get(field), field, currency=market.currency)
    if value is None:
        raise ValueError(f"risk inputs require {field}")
    if field.endswith("_fraction") and value > 1:
        raise ValueError(f"{field} must be between zero and one")
    return value


def load_paper_risk_inputs(path: Path, *, market: MarketProfile = NEPSE) -> PaperRiskInputs:
    with path.open(encoding="utf-8-sig") as file:
        data = json.load(file, parse_float=Decimal, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(data, dict) or not isinstance(data.get("session_date"), str):
        raise ValueError("risk inputs must specify a session_date")
    if data.get("currency", NEPSE.currency) != market.currency:
        raise ValueError(f"risk inputs must use currency {market.currency}")
    session_date = date.fromisoformat(data["session_date"])
    if not isinstance(data.get("kill_switch_active"), bool):
        raise ValueError("risk inputs must specify a boolean kill_switch_active")
    if not isinstance(data.get("limits"), dict) or not isinstance(data.get("instruments"), dict):
        raise ValueError("risk inputs require limits and instrument rules")

    limits = PaperLimits(**{field: _required_amount(data["limits"], field, market)
                            for field in PaperLimits.__dataclass_fields__})
    instruments = {}
    for symbol, rule in data["instruments"].items():
        if (
            not isinstance(rule, dict) or not isinstance(rule.get("sector"), str)
            or not rule["sector"].strip() or type(rule.get("lot_size")) is not int
            or rule["lot_size"] <= 0 or not isinstance(rule.get("tradable"), bool)
        ):
            raise ValueError(f"{symbol} requires a sector, positive lot size and tradable flag")
        instruments[symbol] = InstrumentRules(rule["sector"].strip().casefold(), rule["lot_size"], rule["tradable"])
    equity = _required_amount(data, "equity_npr", market)
    if equity == 0:
        raise ValueError(f'{market.amount_field("equity_npr")} must be positive')
    return PaperRiskInputs(
        session_date, data["kill_switch_active"], equity, _required_amount(data, "daily_loss_npr", market),
        _required_amount(data, "drawdown_fraction", market), limits, instruments,
    )


def _restriction(proposal: Proposal, bar: DailyBar | None, inputs: PaperRiskInputs) -> str | None:
    if inputs.kill_switch_active:
        return "operator stop is active"
    if inputs.daily_loss_npr >= inputs.limits.max_daily_loss_npr:
        return "daily loss limit reached"
    if inputs.drawdown_fraction >= inputs.limits.max_drawdown_fraction:
        return "drawdown limit reached"
    rule = inputs.instruments.get(proposal.symbol)
    if rule is None or not rule.tradable:
        return "tradability is restricted or unverified"
    if proposal.quantity <= 0 or proposal.quantity % rule.lot_size:
        return "quantity does not meet the configured lot size"
    if bar is None or bar.price_basis != "raw" or proposal.reference_close_npr != bar.close:
        return "a matching raw close is required"
    if proposal.quantity > bar.volume * inputs.limits.max_volume_fraction:
        return "quantity exceeds the session volume cap"
    if proposal.quantity * bar.close > inputs.limits.max_order_npr:
        return "order notional exceeds the paper limit"
    return None


def assess_paper_review(
    review: PortfolioReview, snapshot: PortfolioSnapshot, bars: Sequence[DailyBar], inputs: PaperRiskInputs,
    *, trace: dict | None = None, market: MarketProfile = NEPSE,
) -> tuple[RiskAssessment, ...]:
    if inputs.session_date != review.session_date or snapshot.session_date_for(market) != review.session_date:
        raise ValueError("risk inputs, portfolio and review must use the same session date")

    marks = {bar.symbol: bar for bar in bars if bar.session_date == review.session_date}
    if any(bar.currency != market.currency for bar in marks.values()):
        raise ValueError(f"risk marks must use currency {market.currency}")
    holdings = {holding.symbol: holding for holding in snapshot.positions}
    cash = snapshot.available_cash_npr
    gross = Decimal(0)
    by_symbol: dict[str, Decimal] = {}
    by_sector: defaultdict[str, Decimal] = defaultdict(Decimal)
    exposure_complete = True
    assessments = []
    with localcontext(Context(prec=28)):
        for symbol, holding in holdings.items():
            quantity = holding.settled_quantity + holding.pending_buy_quantity
            if not quantity:
                continue
            bar = marks.get(symbol)
            rule = inputs.instruments.get(symbol)
            if bar is None or bar.price_basis != "raw" or rule is None:
                exposure_complete = False
                continue
            value = bar.close * quantity
            by_symbol[symbol] = value
            by_sector[rule.sector] += value
            gross += value

        if trace is not None:
            trace["risk"] = {"inputs": asdict(inputs), "symbols_in_review_order": {}}
        for proposal in review.proposals:
            if trace is not None:
                symbol = proposal.symbol
                rule = inputs.instruments.get(symbol)
                trace["risk"]["symbols_in_review_order"][symbol] = {
                    "proposal": asdict(proposal),
                    "cash_before_npr": cash,
                    "gross_exposure_before_npr": gross,
                    "symbol_exposure_before_npr": by_symbol.get(symbol, Decimal(0)),
                    "sector_exposure_before_npr": by_sector.get(rule.sector, Decimal(0)) if rule else None,
                    "exposure_marks_complete": exposure_complete,
                }
            if proposal.action == "HOLD":
                assessments.append(RiskAssessment(proposal.symbol, "hold", proposal.reason))
                continue
            reason = _restriction(proposal, marks.get(proposal.symbol), inputs)
            if reason is None and proposal.action == "ADD":
                assert proposal.reference_close_npr is not None
                sector = inputs.instruments[proposal.symbol].sector
                notional = proposal.quantity * proposal.reference_close_npr
                cost = proposal.illustrative_cost_npr
                if cash is None or cost is None or notional + cost > cash:
                    reason = "available cash cannot fund the proposal"
                elif not exposure_complete:
                    reason = "current portfolio exposure could not be valued"
                elif by_symbol.get(proposal.symbol, Decimal(0)) + notional > inputs.equity_npr * inputs.limits.max_symbol_fraction:
                    reason = "single-symbol exposure limit exceeded"
                elif by_sector[sector] + notional > inputs.equity_npr * inputs.limits.max_sector_fraction:
                    reason = "sector exposure limit exceeded"
                elif gross + notional > inputs.equity_npr * inputs.limits.max_gross_fraction:
                    reason = "gross exposure limit exceeded"
                else:
                    cash -= notional + cost
                    gross += notional
                    by_sector[sector] += notional
                    by_symbol[proposal.symbol] = by_symbol.get(proposal.symbol, Decimal(0)) + notional
            elif reason is None:
                holding = holdings.get(proposal.symbol)
                if holding is None or holding.sellable_quantity is None or proposal.quantity > holding.sellable_quantity:
                    reason = "proposal exceeds verified sellable inventory"
            assessments.append(
                RiskAssessment(proposal.symbol, "blocked" if reason else "paper_review", reason or "passed paper checks")
            )
    if trace is not None:
        for assessment in assessments:
            trace["risk"]["symbols_in_review_order"][assessment.symbol]["assessment"] = asdict(assessment)
    return tuple(assessments)


def t_plus_two(trade_date: date, settlement_calendar: Mapping[date, str]) -> date:
    """Find ordinary T+2 using explicit settlement-day coverage."""
    return settlement_session(trade_date, settlement_calendar, market=NEPSE)


def settlement_session(
    trade_date: date, settlement_calendar: Mapping[date, str], *, market: MarketProfile,
) -> date:
    status = settlement_calendar.get(trade_date)
    if market.require_trade_date_open_for_settlement and status != "open":
        raise ValueError(f"settlement calendar must mark trade date {trade_date} open")
    if status not in {"open", "closed"}:
        raise ValueError(f"settlement calendar has no status for trade date {trade_date}")
    settlement = trade_date
    for _ in range(market.settlement_days):
        settlement = next_open_session(settlement, settlement_calendar)
    return settlement

"""Chronological paper evaluation of the fixed ATLAS breakout baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time, timedelta
from decimal import Context, Decimal, localcontext

from .execution import ExecutionTerms, Fill, mark_paper_portfolio, simulate_paper
from .market_data import DailyBar
from .paper_risk import PaperRiskInputs
from .portfolio import NEPAL_TIME, PortfolioSnapshot
from .replay import SessionFeatures, replay_daily_bars
from .strategy import review_session


ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class Window:
    start: date
    end: date


@dataclass(frozen=True, slots=True)
class EvaluationInputs:
    bars: Sequence[DailyBar]
    benchmark: Sequence[DailyBar]
    trading_calendar: Mapping[date, str]
    settlement_calendar: Mapping[date, str]
    terms: Sequence[ExecutionTerms]
    risk: PaperRiskInputs
    initial_cash_npr: Decimal


@dataclass(frozen=True, slots=True)
class SessionValue:
    session_date: date
    equity_npr: Decimal
    cash_npr: Decimal
    benchmark_value_npr: Decimal


@dataclass(frozen=True, slots=True)
class Evaluation:
    ending_equity_npr: Decimal
    net_return: Decimal
    max_drawdown: Decimal
    turnover: Decimal
    max_single_name_weight: Decimal
    order_count: int
    trade_count: int
    fees_npr: Decimal
    benchmark_return: Decimal
    benchmark_max_drawdown: Decimal
    no_trade_return: Decimal
    sessions: tuple[SessionValue, ...]


@dataclass(slots=True)
class EvaluationTrace:
    signal_date: date
    details: dict = field(default_factory=dict)

    def section(self, name: str) -> EvaluationTrace:
        return EvaluationTrace(self.signal_date, self.details.setdefault(name, {}))


def _window_dates(window: Window, calendar: Mapping[date, str]) -> list[date]:
    if window.start >= window.end:
        raise ValueError("an evaluation window requires at least two dates")
    dates = []
    for offset in range((window.end - window.start).days + 1):
        day = window.start + timedelta(days=offset)
        status = calendar.get(day)
        if status == "open":
            dates.append(day)
        elif status != "closed":
            raise ValueError(f"trading calendar has no status for {day}")
    if len(dates) < 2 or dates[0] != window.start or dates[-1] != window.end:
        raise ValueError("evaluation boundaries must be open sessions")
    return dates


def _as_of(day: date) -> datetime:
    return datetime.combine(day, time.max, NEPAL_TIME)


def evaluate_window(
    inputs: EvaluationInputs,
    window: Window,
    features: Mapping[date, SessionFeatures],
    *,
    forecasts: Mapping[date, Mapping[str, Decimal]] | None = None,
    trace: EvaluationTrace | None = None,
    session_traces: dict[date, dict] | None = None,
    hold_only: bool = False,
) -> Evaluation:
    if inputs.initial_cash_npr <= 0:
        raise ValueError("starting cash must be positive")

    dates = _window_dates(window, inputs.trading_calendar)
    benchmark = {bar.session_date: bar.close for bar in inputs.benchmark}
    by_day = {
        day: {item.bar.symbol: item.bar for item in features[day].features}
        for day in dates if day in features
    }
    for day in dates:
        if day not in by_day or day not in benchmark:
            raise ValueError(f"missing market or benchmark data on {day}")

    snapshot = PortfolioSnapshot(_as_of(dates[0]), inputs.initial_cash_npr, ZERO, ())
    equity = peak = benchmark_peak = inputs.initial_cash_npr
    daily_loss = drawdown = benchmark_drawdown = max_weight = turnover = fees = ZERO
    orders = trades = 0
    unsettled: tuple[Fill, ...] = ()
    path = [SessionValue(dates[0], equity, equity, equity)]
    entry_lots = {symbol: rule.lot_size for symbol, rule in inputs.risk.instruments.items()}
    with localcontext(Context(prec=28)):
        for signal, trade_date in zip(dates, dates[1:]):
            detail = trace.details if trace is not None and trace.signal_date == signal else None
            if detail is None and session_traces is not None:
                detail = {}
            risk = replace(
                inputs.risk, session_date=signal, equity_npr=equity, daily_loss_npr=daily_loss,
                drawdown_fraction=(peak - equity) / peak,
            )
            review = review_session(
                snapshot, features[signal], entry_lots=entry_lots, hold_only=hold_only,
                forecasts=None if forecasts is None else forecasts.get(signal, {}), trace=detail,
            )
            report = simulate_paper(
                review, snapshot, (*by_day[signal].values(), *by_day[trade_date].values()), risk,
                inputs.terms, inputs.trading_calendar, inputs.settlement_calendar,
                sessions=1, pending_fills=unsettled, trace=detail,
            )
            if detail is not None:
                detail["batch_result"] = asdict(report)
            ledger = report.sessions[0]
            orders += len(report.outcomes)
            new_fills = tuple(outcome.fill for outcome in report.outcomes if outcome.fill is not None)
            trades += len(new_fills)
            turnover += sum((fill.notional_npr for fill in new_fills), ZERO)
            fees += sum((fill.fee_npr for fill in new_fills), ZERO)
            unsettled = tuple(
                fill for fill in (*unsettled, *new_fills) if fill.settlement_date > trade_date
            )
            snapshot = PortfolioSnapshot(_as_of(trade_date), ledger.available_cash_npr, ZERO, ledger.positions)
            next_equity, weight = mark_paper_portfolio(snapshot, by_day[trade_date], unsettled)
            daily_loss = max(ZERO, equity - next_equity)
            equity = next_equity
            peak = max(peak, equity)
            drawdown = max(drawdown, (peak - equity) / peak)
            max_weight = max(max_weight, weight)
            benchmark_value = inputs.initial_cash_npr * benchmark[trade_date] / benchmark[dates[0]]
            benchmark_peak = max(benchmark_peak, benchmark_value)
            benchmark_drawdown = max(benchmark_drawdown, (benchmark_peak - benchmark_value) / benchmark_peak)
            path.append(SessionValue(trade_date, equity, ledger.available_cash_npr, benchmark_value))
            if detail is not None:
                detail["mark_after_batch"] = asdict(path[-1])
                if session_traces is not None:
                    session_traces[signal] = {**detail, "pending_fills": tuple(asdict(fill) for fill in unsettled)}

        average_equity = sum((point.equity_npr for point in path), ZERO) / len(path)
        return Evaluation(
            equity, equity / inputs.initial_cash_npr - 1, drawdown, turnover / average_equity,
            max_weight, orders, trades, fees, path[-1].benchmark_value_npr / inputs.initial_cash_npr - 1,
            benchmark_drawdown, ZERO, tuple(path),
        )


def prepare_features(
    inputs: EvaluationInputs, *, lookback_on: date | None = None, retain_lookbacks: bool = False,
) -> dict[date, SessionFeatures]:
    if not inputs.benchmark or len({(bar.symbol, bar.price_basis) for bar in inputs.benchmark}) != 1:
        raise ValueError("benchmark must use one symbol and one consistent price basis")
    if any(bar.price_basis != "raw" for bar in inputs.bars):
        raise ValueError("execution backtests require raw prices")

    # Replay in time order once. An event can only use its session and earlier bars.
    sessions = replay_daily_bars(inputs.bars, lookback_on=lookback_on, retain_lookbacks=retain_lookbacks)
    return {session.session_date: session for session in sessions}


def execution_scenarios(inputs: EvaluationInputs) -> dict[str, EvaluationInputs]:
    with localcontext(Context(prec=28)):
        high_costs = tuple(
            replace(
                term, commission_bps=term.commission_bps * 2, regulatory_bps=term.regulatory_bps * 2,
                minimum_commission_npr=term.minimum_commission_npr * 2,
                sell_transfer_fee_npr=term.sell_transfer_fee_npr * 2, spread_bps=term.spread_bps * 2,
                slippage_bps=term.slippage_bps * 2,
            )
            for term in inputs.terms
        )
        if any(term.spread_bps / 2 + term.slippage_bps >= 10000 for term in high_costs):
            raise ValueError("doubled spread and slippage exceed the supported sensitivity range")
        thin_terms = tuple(replace(term, max_volume_fraction=term.max_volume_fraction / 2) for term in inputs.terms)
        thin_risk = replace(
            inputs.risk, limits=replace(inputs.risk.limits, max_volume_fraction=inputs.risk.limits.max_volume_fraction / 2),
        )
        return {
            "base": inputs,
            "higher_costs": replace(inputs, terms=high_costs),
            "lower_liquidity": replace(inputs, terms=thin_terms, risk=thin_risk),
        }


def compare_windows(
    inputs: EvaluationInputs, development: Window, holdout: Window, *, trace: EvaluationTrace | None = None,
) -> dict[str, dict[str, Evaluation]]:
    if development.end >= holdout.start:
        raise ValueError("development must finish before the holdout begins")
    features = prepare_features(inputs, lookback_on=trace.signal_date if trace is not None else None)
    if trace is not None and (
        inputs.trading_calendar.get(trace.signal_date) != "open"
        or not any(window.start <= trace.signal_date < window.end for window in (development, holdout))
    ):
        raise ValueError("trace date must be an open signal session before the end of an evaluation window")
    return {
        label: {
            name: evaluate_window(
                case, window, features,
                trace=trace.section(label).section(name)
                if trace is not None and window.start <= trace.signal_date < window.end else None,
            )
            for name, window in (("development", development), ("holdout", holdout))
        }
        for label, case in execution_scenarios(inputs).items()
    }

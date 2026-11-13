"""Run the ATLAS chronological backtest and print an offline JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from .json_inputs import reject_constant, unique_object
from .backtest import EvaluationTrace, EvaluationInputs, Window, compare_windows
from .execution import load_execution_terms
from .market_data import DailyBar, load_nepse_daily_bars, load_session_calendar
from .paper_risk import InstrumentRules, load_paper_risk_inputs
from .portfolio import parse_npr_amount
from .paper_lock import require_paper_lock


SAMPLES = Path(__file__).resolve().parent


def demo_inputs(*, calendar_days: int = 63) -> tuple[EvaluationInputs, Window, Window]:
    first_day = date(2026, 1, 1)
    calendar = {
        first_day + timedelta(days=offset): "open" if (first_day + timedelta(days=offset)).weekday() < 5 else "closed"
        for offset in range(calendar_days)
    }
    days = [day for day, status in calendar.items() if status == "open"]
    bars = []
    benchmark = []
    closes = (100, 100, 100, 100, 100, 110, 113, 90)
    for index, day in enumerate(days):
        pattern = index % len(closes)
        close = closes[pattern]
        opening = 109 if pattern == 6 else close
        volume = {5: 100000, 6: 800}.get(pattern, 10000)
        for symbol, offset in (("DEMO_A", 0), ("DEMO_B", 15)):
            bars.append(DailyBar(
                symbol, day, Decimal(opening + offset), Decimal(max(opening, close) + 1 + offset),
                Decimal(min(opening, close) - 1 + offset), Decimal(close + offset), volume, "raw", "synthetic",
            ))
        level = Decimal(1000 + index)
        benchmark.append(DailyBar("DEMO_INDEX", day, level, level, level, level, 0, "raw", "synthetic"))

    risk = load_paper_risk_inputs(SAMPLES / "nepse_risk_sample.json")
    risk = replace(risk, instruments={
        "DEMO_A": InstrumentRules("demo a", 10, True),
        "DEMO_B": InstrumentRules("demo b", 10, True),
    })
    inputs = EvaluationInputs(
        bars, benchmark, calendar, calendar, load_execution_terms(SAMPLES / "nepse_execution_sample.json"),
        risk, Decimal("15000"),
    )
    terms = tuple(replace(term, effective_from=first_day) for term in inputs.terms)
    return replace(inputs, terms=terms), Window(days[5], days[19]), Window(days[20], days[35])


def _load_window(spec: dict, key: str) -> Window:
    values = spec.get(key)
    if not isinstance(values, list) or len(values) != 2 or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{key} must contain a start and an end date")
    return Window(date.fromisoformat(values[0]), date.fromisoformat(values[1]))


def load_backtest_spec(path: Path) -> tuple[EvaluationInputs, Window, Window, int]:
    with path.open(encoding="utf-8-sig") as source:
        spec = json.load(source, parse_float=Decimal, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(spec, dict):
        raise ValueError("backtest specification must be a JSON object")
    attestations = spec.get("history")
    required = ("point_in_time_universe", "corporate_action_free", "constant_instrument_rules")
    if not isinstance(attestations, dict) or any(attestations.get(key) is not True for key in required):
        raise ValueError("history must confirm point-in-time universe, corporate-action-free data and constant instrument rules")
    starting_cash = parse_npr_amount(spec.get("initial_cash_npr"), "initial_cash_npr")
    min_fills = spec.get("minimum_holdout_fills")
    if not starting_cash or type(min_fills) is not int or min_fills <= 0:
        raise ValueError("specification requires positive initial_cash_npr and minimum_holdout_fills")

    paths = {}
    for key in ("bars", "benchmark", "calendar", "settlement_calendar", "risk", "terms"):
        value = spec.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"specification requires a file path for {key}")
        paths[key] = path.parent / value

    development, holdout = _load_window(spec, "development"), _load_window(spec, "holdout")
    inputs = EvaluationInputs(
        load_nepse_daily_bars(paths["bars"], calendar=paths["calendar"]),
        load_nepse_daily_bars(paths["benchmark"], calendar=paths["calendar"]),
        load_session_calendar(paths["calendar"]), load_session_calendar(paths["settlement_calendar"]),
        load_execution_terms(paths["terms"]), load_paper_risk_inputs(paths["risk"]), starting_cash,
    )
    return inputs, development, holdout, min_fills


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the fixed ATLAS baseline on separate historical windows.")
    parser.add_argument("--spec", type=Path, help="local evaluation specification (default: synthetic demo)")
    parser.add_argument("--trace-date", type=date.fromisoformat, help="explain calculations on one signal date (YYYY-MM-DD)")
    args = parser.parse_args()
    trace = EvaluationTrace(args.trace_date) if args.trace_date is not None else None

    try:
        require_paper_lock()
        if args.spec is None:
            inputs, development, holdout = demo_inputs()
            min_fills = 20
        else:
            inputs, development, holdout, min_fills = load_backtest_spec(args.spec)
        cases = compare_windows(inputs, development, holdout, trace=trace)
    except (OSError, ValueError, ArithmeticError) as error:
        print(f"ATLAS backtest failed: {error}", file=sys.stderr)
        return 1

    max_drawdown = inputs.risk.limits.max_drawdown_fraction
    criteria_met = all(
        result.trade_count >= min_fills and result.max_drawdown <= max_drawdown
        and result.net_return > max(result.no_trade_return, result.benchmark_return)
        for result in (case["holdout"] for case in cases.values())
    )
    print(json.dumps({
        "mode": "historical_paper_backtest", "data": "synthetic_demo" if args.spec is None else "local_inputs",
        **({"explanation": asdict(trace)} if trace is not None else {}),
        "live_trading": "locked", "strategy": "five-bar-breakout-v1, one configured lot on entry",
        "initial_cash_npr": inputs.initial_cash_npr,
        "benchmark": {"symbol": inputs.benchmark[0].symbol, "price_basis": inputs.benchmark[0].price_basis},
        "windows": {"development": asdict(development), "holdout": asdict(holdout)},
        "criteria": {"minimum_holdout_fills": min_fills, "max_drawdown": max_drawdown,
                     "outperform_both_controls_in_all_holdout_scenarios": True},
        "criteria_met": criteria_met if args.spec else None,
        "scenarios": {label: {name: asdict(result) for name, result in case.items()} for label, case in cases.items()},
    }, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

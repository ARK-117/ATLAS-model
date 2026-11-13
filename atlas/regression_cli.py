"""Run the time-ordered ATLAS regression experiment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path

from .backtest import EvaluationTrace, Window
from .backtest_cli import demo_inputs, load_backtest_spec
from .regression import run_experiment
from .paper_lock import require_paper_lock


def demo_experiment_inputs():
    inputs, _, _ = demo_inputs(calendar_days=210)
    dates = sorted(day for day, status in inputs.trading_calendar.items() if status == "open")
    return inputs, Window(dates[5], dates[95]), Window(dates[100], dates[139]), 20


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare linear and ridge forecasts against the paper baseline.")
    parser.add_argument("--spec", type=Path, help="local backtest specification (default: synthetic demo)")
    parser.add_argument("--trace-date", type=date.fromisoformat, help="explain calculations on one signal date (YYYY-MM-DD)")
    args = parser.parse_args()
    trace = EvaluationTrace(args.trace_date) if args.trace_date is not None else None
    try:
        require_paper_lock()
        inputs = demo_experiment_inputs() if args.spec is None else load_backtest_spec(args.spec)
        report = run_experiment(*inputs, trace=trace)
        if args.spec is None:
            report["research_criteria_met"] = None
            for scenario in report["holdout_scenarios"].values():
                scenario["criteria_met"] = None
        encoded = json.dumps({
            "mode": "paper_regression_experiment",
            "data": "synthetic_demo" if args.spec is None else "local_inputs",
            **({"explanation": asdict(trace)} if trace is not None else {}),
            "live_trading": "locked",
            **report,
        }, default=lambda value: asdict(value) if is_dataclass(value) and not isinstance(value, type) else str(value), indent=2, allow_nan=False)
    except (OSError, ValueError, TypeError, ArithmeticError) as error:
        print(f"ATLAS regression failed: {error}", file=sys.stderr)
        return 1
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fit and evaluate small, time-ordered return forecasts for paper research."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Context, Decimal, localcontext
from math import ceil, fsum, hypot, isfinite, log1p, sqrt

from .backtest import (
    EvaluationInputs, EvaluationTrace, Window, _window_dates, evaluate_window, execution_scenarios, prepare_features,
)
from .replay import DailyFeatures, SessionFeatures
from .strategy import FORECAST_HURDLE


HORIZONS = (5, 20)
RIDGE_ALPHA = 0.10
MIN_TRAIN_DATES = 12
FEATURES = ("return_1", "close_vs_prior_high", "log_relative_volume")


@dataclass(frozen=True, slots=True)
class Observation:
    session_date: date
    symbol: str
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Label:
    observation: Observation
    known_on: date
    forward_return: float


@dataclass(frozen=True, slots=True)
class Regression:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def __post_init__(self) -> None:
        if not self.means or len(self.means) != len(self.scales) or len(self.means) != len(self.coefficients):
            raise ValueError("regression parameters must have matching nonempty feature dimensions")
        if not all(isfinite(value) for value in (*self.means, *self.scales, *self.coefficients, self.intercept)):
            raise ValueError("regression parameters must be finite")
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("regression feature scales must be positive")

    def predict(self, values: Sequence[float]) -> float:
        if len(values) != len(self.means) or not all(isfinite(value) for value in values):
            raise ValueError("forecast features must match the model and be finite")
        try:
            result = self.intercept + fsum(
                coefficient * ((value - mean) / scale)
                for value, mean, scale, coefficient in zip(
                    values, self.means, self.scales, self.coefficients, strict=True,
                ) if coefficient
            )
        except OverflowError as error:
            raise ValueError("forecast exceeds the finite numerical range") from error
        if not isfinite(result):
            raise ValueError("forecast exceeds the finite numerical range")
        return result


def _solve(matrix: list[list[float]], target: list[float]) -> tuple[float, ...]:
    """Solve the small standardized system with partial pivoting."""
    rows = [row + [value] for row, value in zip(matrix, target, strict=True)]
    for col in range(len(rows)):
        pivot = max(range(col, len(rows)), key=lambda row: abs(rows[row][col]))
        if abs(rows[pivot][col]) < 1e-10:
            raise ValueError("training features are collinear")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [value / scale for value in rows[col]]
        for index, row in enumerate(rows):
            if index != col:
                rows[index] = [
                    value - row[col] * reference for value, reference in zip(row, rows[col], strict=True)
                ]
    return tuple(row[-1] for row in rows)


def fit(labels: Sequence[Label], alpha: float) -> Regression:
    if not isfinite(alpha) or alpha < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if len({label.observation.session_date for label in labels}) < MIN_TRAIN_DATES:
        raise ValueError(f"need at least {MIN_TRAIN_DATES} matured training dates")
    samples = [label.observation.values for label in labels]
    n = len(samples)
    if not samples[0] or any(len(sample) != len(samples[0]) for sample in samples):
        raise ValueError("training features must have consistent nonempty dimensions")
    if any(not isfinite(label.forward_return) or not all(isfinite(value) for value in label.observation.values)
           for label in labels):
        raise ValueError("training features and targets must be finite")
    try:
        means = tuple(fsum(value / n for value in column) for column in zip(*samples, strict=True))
        root_n = sqrt(n)
        raw_scales = tuple(
            hypot(*((value - mean) / root_n for value in column))
            for column, mean in zip(zip(*samples, strict=True), means, strict=True)
        )
        scales = tuple(scale or 1.0 for scale in raw_scales)
        active = [index for index, scale in enumerate(raw_scales) if scale > 0]
        intercept = fsum(label.forward_return / n for label in labels)
        y = [label.forward_return - intercept for label in labels]
        z = [[(sample[i] - means[i]) / scales[i] for sample in samples] for i in active]
        gram = [
            [fsum(a * b for a, b in zip(left, right, strict=True)) / n + (alpha if i == j else 0)
             for j, right in enumerate(z)]
            for i, left in enumerate(z)
        ]
        target = [fsum(a * b for a, b in zip(column, y, strict=True)) / n for column in z]
        solved = dict(zip(active, _solve(gram, target), strict=True))
        return Regression(means, scales, tuple(solved.get(index, 0.0) for index in range(len(means))), intercept)
    except OverflowError as error:
        raise ValueError("training exceeds the finite numerical range") from error



def _feature_values(item: DailyFeatures) -> tuple[float, ...] | None:
    if (
        item.bar.volume == 0 or item.return_from_previous_close is None
        or item.prior_high is None or item.relative_volume is None
    ):
        return None
    values = (
        float(item.return_from_previous_close), float(item.bar.close / item.prior_high - 1),
        log1p(float(item.relative_volume)),
    )
    return values if all(isfinite(value) for value in values) else None


def _examples(
    features: Mapping[date, SessionFeatures], sessions: Sequence[date], horizon: int,
) -> tuple[dict[date, list[Observation]], list[Label]]:
    with localcontext(Context(prec=28)):
        prices = {
            (day, item.bar.symbol): item.bar.close
            for day, batch in features.items() for item in batch.features if item.bar.volume > 0
        }
        observations: dict[date, list[Observation]] = {}
        labels = []
        for index, day in enumerate(sessions):
            batch = features.get(day)
            if batch is None:
                continue
            for item in batch.features:
                values = _feature_values(item)
                if values is None:
                    continue
                symbol = item.bar.symbol
                observation = Observation(day, symbol, values)
                observations.setdefault(day, []).append(observation)
                if index + horizon < len(sessions):
                    known_on = sessions[index + horizon]
                    future = prices.get((known_on, symbol))
                    if future is not None:
                        forward_return = float(future / item.bar.close - 1)
                        if isfinite(forward_return):
                            labels.append(Label(observation, known_on, forward_return))
        return observations, labels


def matured(labels: Sequence[Label], start: date, before: date) -> list[Label]:
    return [
        label for label in labels
        if start <= label.observation.session_date and label.known_on < before
    ]


def _forecast(
    model: Regression, observations: Mapping[date, list[Observation]], dates: Sequence[date],
) -> dict[date, dict[str, Decimal]]:
    return {
        day: {item.symbol: Decimal(str(model.predict(item.values))) for item in observations.get(day, [])}
        for day in dates
    }


def _trace_forecast(
    trace: EvaluationTrace, model: Regression, observations: Mapping[date, list[Observation]],
    predictions: Mapping[date, Mapping[str, Decimal]], training: Sequence[Label],
    training_cutoff: date, alpha: float,
) -> None:
    day = trace.signal_date
    trace.details.update(
        model=model, alpha=alpha, training_label_count=len(training),
        training_cutoff_exclusive=training_cutoff,
        latest_training_label_date=max(label.known_on for label in training),
        symbols={
            item.symbol: {
                "features_in_model_order": item.values,
                "forecast_return": predictions[day][item.symbol],
            }
            for item in observations.get(day, ())
        },
    )


def _diagnostics(
    predictions: Mapping[date, Mapping[str, Decimal]], labels: Sequence[Label], end: date,
) -> dict[str, float | int | None]:
    errors, zero_errors = [], []
    direction = 0
    for label in labels:
        item = label.observation
        predicted = predictions.get(item.session_date, {}).get(item.symbol)
        if predicted is None or label.known_on > end:
            continue
        estimate = float(predicted)
        errors.append(abs(estimate - label.forward_return))
        zero_errors.append(abs(label.forward_return))
        direction += (estimate > 0) - (estimate < 0) == (label.forward_return > 0) - (label.forward_return < 0)
    n = len(errors)
    mae = fsum(error/n for error in errors) if n else None
    zero_mae = fsum(error/n for error in zero_errors) if n else None
    return {
        "scored_predictions": n,
        "mae": mae,
        "rmse": hypot(*errors)/sqrt(n) if n else None,
        "zero_return_mae": zero_mae,
        "zero_return_rmse": hypot(*zero_errors)/sqrt(n) if n else None,
        "mae_skill_vs_zero": 1-mae/zero_mae if mae is not None and zero_mae else None,
        "p90_absolute_error": sorted(errors)[ceil(n * 0.9) - 1] if n else None,
        "direction_agreement": direction / n if n else None,
    }


def run_experiment(
    inputs: EvaluationInputs, development: Window, holdout: Window, minimum_fills: int,
    *, trace: EvaluationTrace | None = None, include_decay: bool = False,
) -> dict:
    from .decayed_regression import effective_dates, fit_decay
    if type(include_decay) is not bool:
        raise ValueError("the recency candidate switch must be boolean")
    if development.end >= holdout.start or minimum_fills < 1:
        raise ValueError("regression requires ordered windows and a positive minimum fill count")
    features = prepare_features(inputs, lookback_on=trace.signal_date if trace is not None else None)
    dev_dates = _window_dates(development, inputs.trading_calendar)
    test_dates = _window_dates(holdout, inputs.trading_calendar)
    dates = _window_dates(Window(development.start, holdout.end), inputs.trading_calendar)
    first_validation = len(dev_dates) // 2
    if first_validation < max(HORIZONS) + MIN_TRAIN_DATES:
        raise ValueError("development needs at least 64 open sessions for the purged 20-session model")
    validation = dev_dates[first_validation:]
    if trace is not None and trace.signal_date not in validation and trace.signal_date not in test_dates:
        raise ValueError("trace date must be an open development validation or holdout session")
    comparison = Window(validation[0], development.end)
    fold_starts = [len(validation) * index // 3 for index in range(4)]
    candidates, registry = {}, {}
    families: list[tuple[str, float, int | None]] = [("linear", 0.0, None), ("ridge", RIDGE_ALPHA, None)]
    if include_decay:
        families.extend((("decay20", RIDGE_ALPHA, 20), ("decay60", RIDGE_ALPHA, 60)))
    finalists = {}
    def fit_model(training, alpha, half_life):
        return fit(training, alpha) if half_life is None else fit_decay(training, alpha, half_life)
    final_training_traces: dict[str, tuple[list[Label], float]] = {}
    for horizon in HORIZONS:
        observations, labels = _examples(features, dates, horizon)
        for name, alpha, half_life in families:
            key = f"{name}_{horizon}"
            registry[key] = {"family": "date_decay_ridge" if half_life is not None else name,
                             "horizon_sessions": horizon, "ridge_alpha": alpha,
                             "half_life_training_dates": half_life}
            predictions = {}
            training_dates = []
            candidate_trace = (
                trace.section("development_candidates").section(key)
                if trace is not None and trace.signal_date in validation else None
            )
            try:
                for begin, end in zip(fold_starts, fold_starts[1:]):
                    fold = validation[begin:end]
                    training = matured(labels, development.start, fold[0])
                    model = fit_model(training, alpha, half_life)
                    training_dates.append(max(label.known_on for label in training))
                    predictions.update(_forecast(model, observations, fold))
                    if candidate_trace is not None and candidate_trace.signal_date in fold:
                        _trace_forecast(candidate_trace, model, observations, predictions, training, fold[0], alpha)
                final_training = matured(labels, development.start, development.end)
                final = fit_model(final_training, alpha, half_life)
            except ValueError as error:
                candidates[key] = {"status": "unavailable", "reason": str(error)}
                continue
            result = evaluate_window(
                inputs, comparison, features, forecasts=predictions,
                trace=candidate_trace.section("paper_evaluation") if candidate_trace is not None else None,
            )
            candidates[key] = {
                "status": "evaluated", "development": result,
                "diagnostics": _diagnostics(predictions, labels, development.end),
                "fold_last_label_dates": training_dates,
            }
            finalists[key] = (result.net_return, -result.max_drawdown, final, observations, labels)
            if trace is not None and trace.signal_date in test_dates:
                final_training_traces[key] = (final_training, alpha)

    baseline_development = evaluate_window(
        inputs, comparison, features, trace=trace.section("development_baseline")
        if trace is not None and trace.signal_date in validation[:-1] else None,
    )
    if not finalists:
        raise ValueError("no regression candidate has sufficient independent training data")
    # Only development statistics participate in selection. Ties follow the declared candidate order.
    selected = max(finalists, key=lambda key: finalists[key][:2])
    _, _, model, observations, labels = finalists[selected]
    predictions = _forecast(model, observations, test_dates)
    if trace is not None and selected in final_training_traces:
        training, alpha = final_training_traces[selected]
        _trace_forecast(trace.section("frozen_forecast"), model, observations, predictions, training, development.end, alpha)
    checks = {}
    passed = True
    for name, scenario in execution_scenarios(inputs).items():
        case_trace = (
            trace.section("holdout_scenarios").section(name)
            if trace is not None and trace.signal_date in test_dates[:-1] else None
        )
        baseline = evaluate_window(
            scenario, holdout, features, trace=case_trace.section("baseline") if case_trace is not None else None,
        )
        candidate = evaluate_window(
            scenario, holdout, features, forecasts=predictions,
            trace=case_trace.section("candidate") if case_trace is not None else None,
        )
        clears = (
            candidate.trade_count >= minimum_fills
            and candidate.net_return > max(baseline.net_return, baseline.benchmark_return, baseline.no_trade_return)
            and candidate.max_drawdown <= min(baseline.max_drawdown, inputs.risk.limits.max_drawdown_fraction)
        )
        checks[name] = {"baseline": baseline, "candidate": candidate, "criteria_met": clears}
        passed = passed and clears

    selected_development = candidates[selected]["development"]
    final_labels = matured(labels, development.start, development.end)
    selected_half_life = registry[selected]["half_life_training_dates"]
    weight_effective_dates = (effective_dates(final_labels, selected_half_life) if selected_half_life is not None
                              else float(len({label.observation.session_date for label in final_labels})))
    selection_evidence = {
        "evaluated_candidates": len(finalists),
        "selected_development_fills": selected_development.trade_count,
        "has_development_fills": selected_development.trade_count > 0,
        "all_candidates_inactive": all(row["development"].trade_count == 0 for row in candidates.values()
                                       if row["status"] == "evaluated"),
        "development_tie_count": sum(value[:2] == finalists[selected][:2] for value in finalists.values()),
        "readiness": "insufficient_development_activity" if selected_development.trade_count == 0 else "research_only",
    }
    return {
        "candidate_registry": registry, "recency_candidates_enabled": include_decay,
        "weight_effective_training_dates": weight_effective_dates,
        "selection_evidence": selection_evidence,
        "validation_folds": [{"start": validation[begin], "end": validation[end-1],
                              "last_training_label": candidates[selected]["fold_last_label_dates"][index]}
                             for index, (begin, end) in enumerate(zip(fold_starts, fold_starts[1:]))],
        "features": FEATURES,
        "horizons_sessions": HORIZONS,
        "ridge_alpha": RIDGE_ALPHA,
        "forecast_hurdle": FORECAST_HURDLE,
        "development_comparison": comparison,
        "holdout": holdout,
        "minimum_holdout_fills": minimum_fills,
        "baseline_development": baseline_development,
        "development_candidates": candidates,
        "selected_on_development": selected,
        "frozen_model": model,
        "holdout_diagnostics": _diagnostics(predictions, labels, holdout.end),
        "holdout_scenarios": checks,
        "research_criteria_met": passed and selection_evidence["has_development_fills"],
    }

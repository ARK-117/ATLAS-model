from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import date, timedelta
from decimal import Context, Decimal, ROUND_FLOOR, localcontext
from unittest.mock import patch

from atlas.backtest import evaluate_window, prepare_features
from atlas.regression import Label, Observation, Regression, _diagnostics, fit, matured, run_experiment
from atlas.regression_cli import demo_experiment_inputs, main


class RegressionTests(unittest.TestCase):
    def labels(self):
        start = date(2025, 1, 1)
        return [Label(Observation(start + timedelta(days=i), "A", (float(i), 2.0)),
                      start + timedelta(days=i + 1), 3.0 + 2.0 * i) for i in range(20)]

    def test_ols_recovers_known_line_and_ridge_shrinks_it(self) -> None:
        labels = self.labels()
        linear = fit(labels, 0)
        ridge = fit(labels, 0.1)
        self.assertAlmostEqual(linear.predict((25.0, 2.0)), 53.0)
        self.assertAlmostEqual(ridge.predict((25.0, 2.0)), 22.0 + (53.0 - 22.0) / 1.1)
        self.assertEqual(linear.coefficients[1], 0.0)

    def test_training_excludes_targets_known_at_or_after_fold_boundary(self) -> None:
        start = date(2025, 1, 1)
        observation = Observation(start, "A", (1.0,))
        labels = [Label(observation, start + timedelta(days=days), 0.1) for days in (4, 5, 6)]
        self.assertEqual(matured(labels, start, start + timedelta(days=5)), labels[:1])

    def test_holdout_change_does_not_change_selection_or_frozen_fit(self) -> None:
        inputs, development, holdout, minimum = demo_experiment_inputs()
        report = run_experiment(inputs, development, holdout, minimum)
        altered = replace(inputs, bars=tuple(
            replace(bar, close=bar.high) if bar.session_date == holdout.end else bar for bar in inputs.bars
        ))
        changed = run_experiment(altered, development, holdout, minimum)
        self.assertEqual(report["development_candidates"], changed["development_candidates"])
        self.assertEqual(report["selected_on_development"], changed["selected_on_development"])
        self.assertEqual(report["frozen_model"], changed["frozen_model"])
        for scenario, result in report["holdout_scenarios"].items():
            self.assertEqual(result["candidate"].sessions[:-1],
                             changed["holdout_scenarios"][scenario]["candidate"].sessions[:-1])
        self.assertGreater(report["holdout_diagnostics"]["scored_predictions"], 0)

    def test_forecast_signal_uses_the_shared_execution_path(self) -> None:
        inputs, development, _, _ = demo_experiment_inputs()
        result = evaluate_window(
            inputs, development, prepare_features(inputs),
            forecasts={development.start: {"DEMO_A": Decimal("0.02")}},
        )
        self.assertEqual(result.trade_count, 1)
        self.assertGreater(result.fees_npr, 0)

    def test_requires_sufficient_development_for_long_horizon(self) -> None:
        inputs, development, holdout, minimum = demo_experiment_inputs()
        short = replace(development, end=development.start + timedelta(days=13))
        with self.assertRaisesRegex(ValueError, "64 open sessions"):
            run_experiment(inputs, short, holdout, minimum)


    def test_nonfinite_or_negative_regularization_is_rejected(self):
        for alpha in (-0.1, float("nan"), float("inf")):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "regularization"):
                    fit(self.labels(), alpha)

    def test_nonfinite_training_features_and_targets_are_rejected(self):
        for field in ("features", "target"):
            with self.subTest(field=field):
                labels = self.labels()
                if field == "features":
                    labels[0] = replace(labels[0], observation=replace(labels[0].observation, values=(float("nan"), 2.0)))
                else:
                    labels[0] = replace(labels[0], forward_return=float("inf"))
                with self.assertRaisesRegex(ValueError, "finite"):
                    fit(labels, 0.1)

    def test_standardization_keeps_a_known_fit_under_large_unit_changes(self):
        for units in (1e-200, 1.0, 1e200):
            with self.subTest(units=units):
                labels = [replace(label, observation=replace(label.observation, values=(label.observation.values[0] * units, 2.0)))
                          for label in self.labels()]
                self.assertAlmostEqual(fit(labels, 0).predict((25.0 * units, 2.0)), 53.0)
                self.assertAlmostEqual(fit(labels, 0.1).predict((25.0 * units, 2.0)), 22.0 + 31.0 / 1.1)

    def test_invalid_model_parameters_cannot_construct_a_forecaster(self):
        for means, scales, coefficients, intercept in (
            ((0.0,), (0.0,), (1.0,), 0.0),
            ((0.0,), (1.0,), (1.0, 2.0), 0.0),
            ((0.0,), (1.0,), (float("nan"),), 0.0),
            ((0.0,), (1.0,), (1.0,), float("inf")),
        ):
            with self.subTest(parameters=(means, scales, coefficients, intercept)):
                with self.assertRaises(ValueError):
                    Regression(means, scales, coefficients, intercept)

    def test_prediction_rejects_nonfinite_inputs_and_unrepresentable_results(self):
        model = fit(self.labels(), 0)
        for value in (float("nan"), float("inf"), 1e308):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    model.predict((value, 2.0))

    def test_cli_refuses_a_nonfinite_report_without_printing_invalid_json(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["atlas.regression_cli"]), \
             patch("atlas.regression_cli.run_experiment", return_value={"holdout_scenarios": {}, "invalid": float("nan")}), \
             redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(main(), 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("failed", errors.getvalue())


    def test_experiment_is_independent_of_the_callers_decimal_context(self):
        inputs = demo_experiment_inputs()
        baseline = run_experiment(*inputs)
        for context in (Context(prec=6), Context(prec=50, rounding=ROUND_FLOOR)):
            with self.subTest(precision=context.prec, rounding=context.rounding):
                with localcontext(context) as active:
                    actual = run_experiment(*inputs)
                    self.assertEqual(active.prec, context.prec)
                    self.assertEqual(active.rounding, context.rounding)
                self.assertEqual(actual, baseline)

    def test_no_change_baseline_uses_the_same_matured_targets_and_undefined_skill_is_explicit(self):
        start = date(2026,1,1)
        labels = [Label(Observation(start+timedelta(days=i),"A",(1.0,)),
                        start+timedelta(days=i+1),target) for i,target in enumerate((0.0,0.2,0.1))]
        predictions = {label.observation.session_date:{"A":Decimal("0.1")} for label in labels}
        boundary = start+timedelta(days=3)
        diagnostics = _diagnostics(predictions,labels,boundary)
        self.assertEqual(diagnostics["scored_predictions"],3)
        for key, expected in (("mae",0.2/3),("zero_return_mae",0.1),("mae_skill_vs_zero",1/3)):
            actual = diagnostics[key]
            assert actual is not None
            self.assertAlmostEqual(float(actual),expected)
        future = Label(Observation(start,"A",(1.0,)),boundary+timedelta(days=1),1e200)
        self.assertEqual(diagnostics,_diagnostics(predictions,labels+[future],boundary))
        zeros = [Label(Observation(start,"A",(1.0,)),start,0.0)]
        self.assertIsNone(_diagnostics({start:{"A":Decimal(0)}},zeros,boundary)["mae_skill_vs_zero"])
        self.assertIsNone(_diagnostics({},zeros,boundary)["zero_return_mae"])

    def test_forecast_rmse_handles_finite_large_errors_without_squaring_overflow(self):
        start = date(2026,1,1)
        labels = [Label(Observation(start,"A",(1.0,)),start,1e200)]
        diagnostics = _diagnostics({start:{"A":Decimal(0)}},labels,start)
        self.assertEqual(diagnostics["rmse"],1e200)
        self.assertEqual(diagnostics["zero_return_rmse"],1e200)
        self.assertEqual(diagnostics["mae_skill_vs_zero"],0)

    def test_zero_activity_and_fixed_order_ties_are_reported_without_a_profit_claim(self):
        report = run_experiment(*demo_experiment_inputs())
        evidence = report["selection_evidence"]
        self.assertEqual(evidence["selected_development_fills"],0)
        self.assertTrue(evidence["all_candidates_inactive"])
        self.assertEqual(evidence["development_tie_count"],4)
        self.assertEqual(evidence["readiness"],"insufficient_development_activity")
        for fold in report["validation_folds"]:
            self.assertLess(fold["last_training_label"],fold["start"])

    def test_a_lucky_holdout_cannot_qualify_a_model_with_no_validation_fills(self):
        inputs,development,holdout,minimum = demo_experiment_inputs()
        def lucky(case,window,features,**kwargs):
            result = evaluate_window(case,window,features,**kwargs)
            if window == holdout and kwargs.get("forecasts") is not None:
                return replace(result,net_return=Decimal("0.20"),max_drawdown=Decimal(0),
                               trade_count=minimum,order_count=max(minimum,result.order_count))
            return result
        with patch("atlas.regression.evaluate_window",side_effect=lucky):
            report = run_experiment(inputs,development,holdout,minimum)
        self.assertTrue(all(case["criteria_met"] for case in report["holdout_scenarios"].values()))
        self.assertFalse(report["selection_evidence"]["has_development_fills"])
        self.assertFalse(report["research_criteria_met"])

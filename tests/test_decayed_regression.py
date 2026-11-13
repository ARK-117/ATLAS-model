from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, timedelta
from decimal import Context, localcontext

from atlas.decayed_regression import effective_dates, fit_decay
from atlas.regression import Label, Observation, fit, run_experiment
from atlas.regression_cli import demo_experiment_inputs


class DecayedRegressionTests(unittest.TestCase):
    def labels(self):
        start = date(2025,1,1)
        return [Label(Observation(start+timedelta(days=i),"A",(float(i),2.0)),
                      start+timedelta(days=i+1),3.0+2*i) for i in range(40)]

    def test_known_line_and_ridge_shrinkage_hold_with_weighted_centering(self):
        labels = self.labels()
        linear, ridge = fit_decay(labels,0,20), fit_decay(labels,0.1,20)
        self.assertAlmostEqual(linear.predict((45.0,2.0)),93.0)
        self.assertAlmostEqual(ridge.predict((45.0,2.0)),linear.intercept+(93.0-linear.intercept)/1.1)
        self.assertEqual(linear.coefficients[1],0)

    def test_date_balancing_prevents_duplicate_assets_from_inflating_time_evidence(self):
        labels = self.labels()
        repeated = labels+[replace(labels[10],observation=replace(labels[10].observation,symbol="B"))]*100
        before, after = fit_decay(labels,0.1,20),fit_decay(repeated,0.1,20)
        self.assertAlmostEqual(before.predict((45.0,2.0)),after.predict((45.0,2.0)),places=11)
        self.assertAlmostEqual(effective_dates(labels,20),effective_dates(repeated,20))
        with self.assertRaisesRegex(ValueError,"weight-effective"):
            fit_decay(repeated,0.1,0.1)

    def test_recency_adapts_to_a_synthetic_regime_change_without_claiming_market_profit(self):
        start = date(2025,1,1)
        labels = [Label(Observation(start+timedelta(days=i),"A",(float(i%5+1),)),
                        start+timedelta(days=i+1),float(i%5+1)*(1 if i<30 else 3)) for i in range(60)]
        ordinary,recency = fit(labels,0.1),fit_decay(labels,0.1,20)
        self.assertLess(abs(recency.predict((3.0,))-9),abs(ordinary.predict((3.0,))-9))

    def test_invalid_weights_numbers_and_numerical_unit_changes(self):
        for half in (0,-1,float("nan"),float("inf"),True):
            with self.subTest(half=half),self.assertRaises(ValueError):
                fit_decay(self.labels(),0.1,half)
        for alpha in (-1,float("nan"),float("inf")):
            with self.subTest(alpha=alpha),self.assertRaises(ValueError):
                fit_decay(self.labels(),alpha,20)
        for units in (1e-200,1.0,1e200):
            labels=[replace(label,observation=replace(label.observation,values=(label.observation.values[0]*units,2.0)))
                    for label in self.labels()]
            self.assertAlmostEqual(fit_decay(labels,0,20).predict((45.0*units,2.0)),93.0)
        bad=self.labels()
        bad[0]=replace(bad[0],forward_return=float("nan"))
        with self.assertRaisesRegex(ValueError,"finite"):
            fit_decay(bad,0.1,20)

    def test_optional_candidates_preserve_default_family_and_cannot_use_holdout_to_select(self):
        inputs,development,holdout,minimum = demo_experiment_inputs()
        baseline=run_experiment(inputs,development,holdout,minimum)
        self.assertEqual(len(baseline["development_candidates"]),4)
        experimental=run_experiment(inputs,development,holdout,minimum,include_decay=True)
        self.assertEqual(len(experimental["development_candidates"]),8)
        altered=replace(inputs,bars=tuple(replace(bar,close=bar.high) if bar.session_date==holdout.end else bar for bar in inputs.bars))
        changed=run_experiment(altered,development,holdout,minimum,include_decay=True)
        self.assertEqual(experimental["development_candidates"],changed["development_candidates"])
        self.assertEqual(experimental["frozen_model"],changed["frozen_model"])
        self.assertEqual(experimental["selected_on_development"],changed["selected_on_development"])

    def test_weighted_fit_is_independent_of_decimal_context(self):
        expected=fit_decay(self.labels(),0.1,20)
        with localcontext(Context(prec=6)):
            self.assertEqual(expected,fit_decay(self.labels(),0.1,20))

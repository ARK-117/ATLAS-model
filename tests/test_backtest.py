from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, time
from decimal import Context, Decimal, localcontext

from atlas.backtest import Window, compare_windows, evaluate_window, execution_scenarios, prepare_features
from atlas.backtest_cli import demo_inputs
from atlas.execution import Fill, mark_paper_portfolio
from atlas.portfolio import Holding, NEPAL_TIME, PortfolioSnapshot


class BacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs, self.development, self.holdout = demo_inputs()

    def test_each_window_starts_with_equal_cash_and_reports_controls(self) -> None:
        cases = compare_windows(self.inputs, self.development, self.holdout)
        self.assertEqual(set(cases), {"base", "higher_costs", "lower_liquidity"})
        for case in cases.values():
            for result in case.values():
                first = result.sessions[0]
                self.assertEqual(first.cash_npr, self.inputs.initial_cash_npr)
                self.assertEqual(first.benchmark_value_npr, first.equity_npr)
                self.assertEqual(result.no_trade_return, 0)
                self.assertGreater(result.benchmark_return, 0)
                self.assertLessEqual(result.trade_count, result.order_count)
        self.assertGreater(cases["base"]["development"].trade_count, 0)
        self.assertGreater(cases["base"]["holdout"].trade_count, 0)
        self.assertLess(cases["lower_liquidity"]["holdout"].trade_count, cases["base"]["holdout"].trade_count)

    def test_earlier_results_do_not_use_future_bars(self) -> None:
        baseline = compare_windows(self.inputs, self.development, self.holdout)["base"]
        altered = tuple(
            replace(bar, close=bar.high, volume=0) if bar.session_date == self.holdout.end else bar
            for bar in self.inputs.bars
        )
        changed = compare_windows(replace(self.inputs, bars=altered), self.development, self.holdout)["base"]
        self.assertEqual(changed["development"], baseline["development"])
        self.assertEqual(changed["holdout"].sessions[:-1], baseline["holdout"].sessions[:-1])

    def test_session_traces_follow_the_same_book_without_lookahead(self) -> None:
        inputs, window = self.inputs, self.holdout
        traces: dict = {}
        result = evaluate_window(inputs, window, prepare_features(inputs, retain_lookbacks=True), session_traces=traces)
        self.assertEqual(result, evaluate_window(inputs, window, prepare_features(inputs)))
        self.assertEqual(list(traces), [mark.session_date for mark in result.sessions[:-1]])
        self.assertTrue(any(trace["pending_fills"] for trace in traces.values()))
        for signal, trace in traces.items():
            batch = trace["batch_result"]
            close = batch["sessions"][0]["session_date"]
            self.assertEqual(close, trace["mark_after_batch"]["session_date"])
            self.assertTrue(all(fill["settlement_date"] > close for fill in trace["pending_fills"]))
            for detail in trace["strategy"]["symbols_in_review_order"].values():
                self.assertEqual(detail["features"]["bar"]["session_date"], signal)
                self.assertTrue(all(bar["session_date"] < signal for bar in detail["preceding_bars"]))

    def test_marking_counts_unsettled_sale_proceeds_instead_of_sold_shares(self) -> None:
        bar = self.inputs.bars[0]
        snapshot = PortfolioSnapshot(
            datetime.combine(bar.session_date, time.max, NEPAL_TIME), Decimal("500"), Decimal(0),
            (Holding(bar.symbol, 10, 5, 0),),
        )
        fill = Fill(bar.symbol, "sell", 5, Decimal("100"), Decimal("2"), bar.session_date, self.development.start)
        equity, weight = mark_paper_portfolio(snapshot, {bar.symbol: bar}, (fill,))
        self.assertEqual(equity, Decimal("1498"))
        self.assertEqual(weight, Decimal(500) / Decimal(1498))

    def test_rejects_overlap_and_incomplete_benchmark(self) -> None:
        with self.assertRaisesRegex(ValueError, "development must finish"):
            compare_windows(self.inputs, self.development, Window(self.development.end, self.holdout.end))
        missing = tuple(bar for bar in self.inputs.benchmark if bar.session_date != self.holdout.start)
        with self.assertRaisesRegex(ValueError, "missing market or benchmark data"):
            compare_windows(replace(self.inputs, benchmark=missing), self.development, self.holdout)


    def test_sensitivity_assumptions_do_not_use_the_callers_decimal_precision(self):
        terms = tuple(replace(term, spread_bps=Decimal("0.123456789123")) for term in self.inputs.terms)
        inputs = replace(self.inputs, terms=terms)
        expected = execution_scenarios(inputs)
        with localcontext(Context(prec=6)) as active:
            actual = execution_scenarios(inputs)
            self.assertEqual(active.prec, 6)
        self.assertEqual(actual, expected)

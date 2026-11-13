from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from atlas.market_data import load_nepse_daily_bars
from atlas.paper_risk import assess_paper_review, load_paper_risk_inputs, t_plus_two
from atlas.portfolio import load_portfolio_snapshot
from atlas.strategy import review_portfolio


SAMPLES = Path(__file__).resolve().parents[1] / "atlas"


class PaperRiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = load_nepse_daily_bars(SAMPLES / "nepse_daily_sample.csv")
        self.snapshot = load_portfolio_snapshot(SAMPLES / "nepse_portfolio_sample.json")
        self.review = review_portfolio(self.snapshot, self.bars)
        self.inputs = load_paper_risk_inputs(SAMPLES / "nepse_risk_sample.json")

    def assess(self, inputs=None, snapshot=None):
        return assess_paper_review(self.review, snapshot or self.snapshot, self.bars, inputs or self.inputs)

    def test_demo_passes_and_operator_stop_blocks(self) -> None:
        self.assertEqual([item.status for item in self.assess()], ["paper_review", "paper_review"])
        stopped = self.assess(replace(self.inputs, kill_switch_active=True))
        self.assertTrue(all(item.reason == "operator stop is active" for item in stopped))

    def test_blocks_stale_risk_inputs(self) -> None:
        stale = replace(self.inputs, session_date=self.inputs.session_date - timedelta(days=1))
        with self.assertRaisesRegex(ValueError, "same session date"):
            self.assess(stale)

    def test_blocks_unknown_rules_and_exposure_breaches(self) -> None:
        no_rules = replace(self.inputs, instruments={})
        self.assertTrue(all(item.status == "blocked" for item in self.assess(no_rules)))
        for field, reason in (
            ("max_symbol_fraction", "single-symbol"),
            ("max_sector_fraction", "sector"),
            ("max_gross_fraction", "gross"),
        ):
            with self.subTest(field=field):
                limits = replace(self.inputs.limits, **{field: Decimal("0.001")})
                trim, add = self.assess(replace(self.inputs, limits=limits))
                self.assertEqual(trim.status, "paper_review")
                self.assertEqual(add.status, "blocked")
                self.assertIn(reason, add.reason)

    def test_blocks_cash_and_sellable_inventory_shortages(self) -> None:
        position = replace(self.snapshot.positions[0], sellable_quantity=0)
        snapshot = replace(self.snapshot, available_cash_npr=Decimal(0), positions=(position,))
        trim, add = self.assess(snapshot=snapshot)
        self.assertIn("sellable inventory", trim.reason)
        self.assertIn("available cash", add.reason)

    def test_t_plus_two_skips_explicit_closures_and_rejects_unknown_dates(self) -> None:
        trade = date(2026, 10, 20)
        calendar = {
            trade: "open",
            trade + timedelta(days=1): "closed",
            trade + timedelta(days=2): "open",
        }
        with self.assertRaisesRegex(ValueError, "no status for 2026-10-23"):
            t_plus_two(trade, calendar)
        calendar[trade + timedelta(days=3)] = "open"
        self.assertEqual(t_plus_two(trade, calendar), date(2026, 10, 23))

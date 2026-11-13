from __future__ import annotations

import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from atlas.market_data import load_nepse_daily_bars
from atlas.portfolio import Holding, load_portfolio_snapshot
from atlas.strategy import review_portfolio


SAMPLES = Path(__file__).resolve().parents[1] / "atlas"


class PortfolioReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = load_nepse_daily_bars(SAMPLES / "nepse_daily_sample.csv")
        self.snapshot = load_portfolio_snapshot(SAMPLES / "nepse_portfolio_sample.json")

    def test_proposes_trim_of_sellable_inventory_and_add(self) -> None:
        trim, add = review_portfolio(self.snapshot, self.bars).proposals
        self.assertEqual((trim.action, trim.quantity), ("TRIM", 70))
        self.assertEqual((add.action, add.quantity), ("ADD", 1))
        self.assertEqual(add.illustrative_cost_npr, Decimal("4.28"))

    def test_exits_if_the_whole_position_is_sellable(self) -> None:
        snapshot = replace(self.snapshot, positions=(Holding("DEMO_A", 80, 80, 0),))
        exit_proposal = review_portfolio(snapshot, self.bars).proposals[0]
        self.assertEqual((exit_proposal.action, exit_proposal.quantity), ("EXIT", 80))

    def test_holds_when_cash_including_costs_is_insufficient(self) -> None:
        snapshot = replace(self.snapshot, available_cash_npr=Decimal("430"))
        add_candidate = review_portfolio(snapshot, self.bars).proposals[-1]
        self.assertEqual(add_candidate.action, "HOLD")
        self.assertIn("cost buffer", add_candidate.reason)

    def test_holds_when_the_breakout_already_has_pending_exposure(self) -> None:
        snapshot = replace(
            self.snapshot, positions=(*self.snapshot.positions, Holding("DEMO_B", 0, 0, 20)),
        )
        add_candidate = review_portfolio(snapshot, self.bars).proposals[-1]
        self.assertEqual(add_candidate.action, "HOLD")
        self.assertIn("new positions", add_candidate.reason)

    def test_uses_cash_once_across_new_proposals(self) -> None:
        extra = [replace(bar, symbol="DEMO_C") for bar in self.bars if bar.symbol == "DEMO_B"]
        bars = sorted([*self.bars, *extra], key=lambda bar: (bar.session_date, bar.symbol))
        snapshot = replace(self.snapshot, available_cash_npr=Decimal("500"))
        _, first, second = review_portfolio(snapshot, bars).proposals
        self.assertEqual((first.action, second.action), ("ADD", "HOLD"))

    def test_hold_control_and_missing_session(self) -> None:
        review = review_portfolio(self.snapshot, self.bars, hold_only=True)
        self.assertEqual({proposal.action for proposal in review.proposals}, {"HOLD"})
        with self.assertRaisesRegex(ValueError, "no completed bars match portfolio session"):
            prior_bars = [bar for bar in self.bars if bar.session_date < self.snapshot.session_date]
            review_portfolio(self.snapshot, prior_bars)

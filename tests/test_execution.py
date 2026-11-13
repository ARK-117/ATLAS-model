from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

from atlas.execution import load_execution_terms, simulate_paper
from atlas.market_data import load_nepse_daily_bars, load_session_calendar
from atlas.paper_risk import load_paper_risk_inputs
from atlas.portfolio import load_portfolio_snapshot
from atlas.strategy import review_portfolio


SAMPLES = Path(__file__).resolve().parents[1] / "atlas"


class PaperExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calendar = load_session_calendar(SAMPLES / "nepse_sessions_sample.csv")
        self.bars = load_nepse_daily_bars(
            SAMPLES / "nepse_daily_sample.csv", calendar=SAMPLES / "nepse_sessions_sample.csv",
        )
        self.snapshot = load_portfolio_snapshot(SAMPLES / "nepse_portfolio_sample.json")
        self.review = review_portfolio(self.snapshot, self.bars)
        self.risk = load_paper_risk_inputs(SAMPLES / "nepse_risk_sample.json")
        self.schedule = load_execution_terms(SAMPLES / "nepse_execution_sample.json")

    def simulate(self, *, bars=None, review=None, risk=None, schedule=None, **kwargs):
        return simulate_paper(
            review or self.review, self.snapshot, self.bars if bars is None else bars,
            risk or self.risk, schedule or self.schedule, self.calendar, self.calendar, **kwargs,
        )

    def test_reserves_fills_and_reconciles_cash_and_shares_at_settlement(self) -> None:
        result = self.simulate()
        self.assertEqual(result.eligible_session, date(2026, 10, 21))
        self.assertEqual(result.reserved_cash_npr, Decimal("432.28"))
        self.assertEqual(result.reserved_shares, {"DEMO_A": 70})
        self.assertEqual(result.legacy_pending_sale_proceeds_npr, Decimal("3000"))
        sell, buy = result.outcomes
        self.assertEqual((sell.status, buy.status), ("partial_expired", "filled"))
        assert sell.fill is not None and buy.fill is not None
        self.assertEqual((sell.fill.quantity, sell.fill.price_npr, sell.fill.fee_npr),
                         (48, Decimal("246.62"), Decimal("16.84")))
        self.assertEqual((buy.fill.quantity, buy.fill.price_npr, buy.fill.fee_npr),
                         (1, Decimal("426.64"), Decimal("0.43")))

        first, second, settled = result.sessions
        self.assertEqual(first.available_cash_npr, Decimal("14572.93"))
        self.assertEqual(first.available_cash_npr, second.available_cash_npr)
        self.assertEqual(first.simulated_pending_sale_proceeds_npr, Decimal("11820.92"))
        self.assertEqual((first.positions[0].settled_quantity, first.positions[0].sellable_quantity), (80, 22))
        self.assertEqual(first.positions[1].pending_buy_quantity, 1)
        self.assertEqual(settled.session_date, date(2026, 10, 25))
        self.assertEqual(settled.available_cash_npr, Decimal("26393.85"))
        self.assertEqual(settled.simulated_pending_sale_proceeds_npr, Decimal(0))
        self.assertEqual((settled.positions[0].settled_quantity, settled.positions[0].pending_buy_quantity), (32, 20))
        self.assertEqual((settled.positions[1].settled_quantity, settled.positions[1].sellable_quantity), (1, 1))

    def test_cancellation_and_risk_rejection_leave_balances_alone(self) -> None:
        cancelled = self.simulate(cancelled_symbols=frozenset({"DEMO_A"}))
        self.assertEqual(cancelled.outcomes[0].status, "cancelled")
        self.assertEqual(cancelled.reserved_shares, {})
        stopped = self.simulate(risk=replace(self.risk, kill_switch_active=True))
        self.assertEqual({outcome.status for outcome in stopped.outcomes}, {"rejected"})
        self.assertEqual(stopped.sessions[-1].available_cash_npr, self.snapshot.available_cash_npr)
        self.assertEqual(stopped.sessions[-1].positions, self.snapshot.positions)

    def test_intraday_limit_touch_without_eligible_open_does_not_fill(self) -> None:
        bars = tuple(
            replace(bar, open=Decimal("432")) if bar.symbol == "DEMO_B" and bar.session_date == date(2026, 10, 21)
            else bar for bar in self.bars
        )
        result = self.simulate(bars=bars)
        self.assertEqual(result.outcomes[1].status, "expired")
        self.assertEqual(result.sessions[0].available_cash_npr, self.snapshot.available_cash_npr)

    def test_uses_effective_schedule_for_fill_session(self) -> None:
        initial, = self.schedule
        effective = replace(initial, effective_from=date(2026, 10, 21), commission_bps=Decimal("50"))
        future = replace(effective, effective_from=date(2026, 10, 22), commission_bps=Decimal("500"))
        result = self.simulate(schedule=(initial, effective, future))
        self.assertEqual(result.terms_effective_from, effective.effective_from)
        assert result.outcomes[1].fill is not None
        self.assertEqual(result.outcomes[1].fill.fee_npr, Decimal("2.22"))

    def test_rejects_a_fill_when_effective_fees_exceed_the_reservation(self) -> None:
        expensive = replace(self.schedule[0], minimum_commission_npr=Decimal("20"))
        result = self.simulate(schedule=(expensive,))
        self.assertEqual(result.outcomes[1].status, "rejected")
        self.assertEqual(result.sessions[0].available_cash_npr, self.snapshot.available_cash_npr)

    def test_no_trade_control_produces_no_orders_or_settlement(self) -> None:
        review = review_portfolio(self.snapshot, self.bars, hold_only=True)
        result = self.simulate(review=review)
        self.assertEqual(result.outcomes, ())
        self.assertEqual(result.sessions[-1].available_cash_npr, self.snapshot.available_cash_npr)
        self.assertEqual(result.sessions[-1].positions, self.snapshot.positions)

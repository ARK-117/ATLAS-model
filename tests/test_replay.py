from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal, localcontext

from atlas.market_data import DailyBar
from atlas.replay import LOOKBACK, replay_daily_bars


def sample_bars(symbol: str = "DEMO_A") -> list[DailyBar]:
    return [
        DailyBar(
            symbol=symbol,
            session_date=date(2026, 10, 12) + timedelta(days=day),
            open=Decimal(100 + day),
            high=Decimal(101 + day),
            low=Decimal(99 + day),
            close=Decimal(100 + day),
            volume=100,
            price_basis="raw",
            source="synthetic",
        )
        for day in range(LOOKBACK + 2)
    ]


class ReplayTests(unittest.TestCase):
    def test_features_exclude_current_bar_from_their_reference_window(self) -> None:
        bars = sample_bars()
        bars[LOOKBACK] = replace(bars[LOOKBACK], high=Decimal("160"), volume=250)
        sessions = list(replay_daily_bars(bars))

        self.assertIsNone(sessions[0].features[0].return_from_previous_close)
        self.assertIsNone(sessions[LOOKBACK - 1].features[0].prior_high)
        features = sessions[LOOKBACK].features[0]
        self.assertEqual(features.prior_high, Decimal("105"))
        self.assertEqual(features.prior_low, Decimal("99"))
        self.assertEqual(features.relative_volume, Decimal("2.5"))

    def test_future_observation_changes_do_not_affect_earlier_sessions(self) -> None:
        bars = sample_bars()
        baseline = list(replay_daily_bars(bars))
        bars[-1] = replace(bars[-1], close=bars[-1].high)
        with localcontext() as context:
            context.prec = 8
            revised = list(replay_daily_bars(bars))
        self.assertEqual(baseline[:-1], revised[:-1])
        self.assertNotEqual(baseline[-1], revised[-1])

    def test_session_batches_keep_symbol_histories_separate(self) -> None:
        other = sample_bars("DEMO_B")[LOOKBACK]
        bars = sorted([*sample_bars(), other], key=lambda bar: (bar.session_date, bar.symbol))
        batch = list(replay_daily_bars(bars))[LOOKBACK]
        self.assertEqual(batch.session_date, other.session_date)
        self.assertEqual([item.bar.symbol for item in batch.features], ["DEMO_A", "DEMO_B"])
        self.assertIsNotNone(batch.features[0].prior_high)
        self.assertIsNone(batch.features[1].prior_high)

    def test_mixed_price_basis_and_unordered_bars_are_rejected(self) -> None:
        first, second = sample_bars()[:2]
        with self.assertRaisesRegex(ValueError, "changes price basis"):
            list(replay_daily_bars([first, replace(second, price_basis="adjusted")]))
        with self.assertRaisesRegex(ValueError, "ordered by session date and symbol"):
            list(replay_daily_bars([second, first]))

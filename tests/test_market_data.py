from __future__ import annotations

import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from atlas.market_data import BAR_COLUMNS, load_nepse_daily_bars


FIRST = "demo_a,2026-10-12,1d,248.10,254,247,252,18500,NPR,raw,synthetic"
LAST = "demo_a,2026-10-14,1d,252,256,250,255,17400,NPR,raw,synthetic"


class NepseDailyBarTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write_bars(self, *rows: str) -> Path:
        path = self.directory / "bars.csv"
        path.write_text("\n".join((",".join(BAR_COLUMNS), *rows)), encoding="utf-8-sig")
        return path

    def write_calendar(self, *rows: str) -> Path:
        path = self.directory / "sessions.csv"
        path.write_text("\n".join(("session_date,status", *rows)), encoding="utf-8")
        return path

    def test_normalizes_symbols_and_sorts_by_session_without_changing_prices(self) -> None:
        other = FIRST.replace("demo_a", "demo_b").replace(",18500,", ",0,").replace(",raw,", ",adjusted,")
        path = self.write_bars(FIRST, LAST, other)
        bars = load_nepse_daily_bars(path)
        self.assertEqual([bar.symbol for bar in bars], ["DEMO_A", "DEMO_B", "DEMO_A"])
        self.assertEqual(bars[0].open, Decimal("248.10"))
        self.assertEqual(bars[0].session_date, date(2026, 10, 12))
        self.assertEqual(bars[1].volume, 0)
        self.assertEqual(bars[1].price_basis, "adjusted")
        self.assertEqual(bars[1].source, "synthetic")

    def test_rejects_duplicates_even_when_a_symbol_reappears_later(self) -> None:
        path = self.write_bars(FIRST, LAST, FIRST.upper().replace("1D", "1d").replace("RAW", "raw"))
        with self.assertRaisesRegex(ValueError, r":4: duplicate bar for DEMO_A on 2026-10-12"):
            load_nepse_daily_bars(path)

    def test_rejects_out_of_order_dates_for_one_symbol(self) -> None:
        path = self.write_bars(LAST, FIRST)
        with self.assertRaisesRegex(ValueError, "must be in ascending order"):
            load_nepse_daily_bars(path)

    def test_rejects_malformed_rows_with_line_and_reason(self) -> None:
        for old, new, reason in (
            ("2026-10-12", "2026-02-30", "session_date must be a valid YYYY-MM-DD date"),
            ("248.10", "NaN", "open must be a positive finite price"),
            (",254,", ",240,", "open and close must lie between low and high"),
            (",18500,", ",-1,", "volume must be a nonnegative whole number"),
            (",1d,", ",5m,", "expected interval 1d and currency NPR"),
            (",raw,", ",,", "expected a value for every column"),
            (",raw,", ",unspecified,", "price_basis must be raw or adjusted"),
        ):
            with self.subTest(reason=reason):
                path = self.write_bars(FIRST.replace(old, new))
                with self.assertRaisesRegex(ValueError, f":2: {reason}"):
                    load_nepse_daily_bars(path)

    def test_requires_the_daily_header_and_at_least_one_bar(self) -> None:
        path = self.write_bars()
        with self.assertRaisesRegex(ValueError, "expected at least one daily bar"):
            load_nepse_daily_bars(path)
        path.write_text("symbol,close\nDEMO_A,252\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected columns"):
            load_nepse_daily_bars(path)

    def test_calendar_distinguishes_gaps_closed_sessions_and_unknown_dates(self) -> None:
        path = self.write_bars(FIRST, LAST)
        calendar = self.write_calendar("2026-10-12,open", "2026-10-13,closed", "2026-10-14,open")
        self.assertEqual(len(load_nepse_daily_bars(path, calendar=calendar)), 2)

        path = self.write_bars(FIRST, LAST.replace("2026-10-14", "2026-10-13"))
        with self.assertRaisesRegex(ValueError, "has a bar on closed session 2026-10-13"):
            load_nepse_daily_bars(path, calendar=calendar)

        path = self.write_bars(FIRST, LAST)
        calendar = self.write_calendar("2026-10-12,open", "2026-10-14,open")
        with self.assertRaisesRegex(ValueError, "calendar has no status for 2026-10-13"):
            load_nepse_daily_bars(path, calendar=calendar)

        calendar = self.write_calendar("2026-10-12,open", "2026-10-13,open", "2026-10-14,open")
        with self.assertRaisesRegex(ValueError, "DEMO_A is missing a bar on open session 2026-10-13"):
            load_nepse_daily_bars(path, calendar=calendar)

        zero_volume = LAST.replace("2026-10-14", "2026-10-13").replace(",17400,", ",0,")
        path = self.write_bars(FIRST, zero_volume, LAST)
        self.assertEqual(load_nepse_daily_bars(path, calendar=calendar)[1].volume, 0)


    def test_short_exponents_cannot_expand_past_the_price_digit_bound(self):
        for value in ("1e100000000", "1e-100000000", "9" * 129, "1." + "0" * 128):
            with self.subTest(value=value[:32]):
                row = list[str](FIRST.split(","))
                row[3:7] = [value] * 4
                path = self.write_bars(",".join(row))
                with self.assertRaisesRegex(ValueError, "128.*digits"):
                    load_nepse_daily_bars(path)

    def test_supported_price_boundaries_keep_their_exact_decimal_values(self):
        for value in ("1e127", "1e-127", "9" * 128, "1.0500E2"):
            with self.subTest(value=value[:32]):
                row = list[str](FIRST.split(","))
                row[3:7] = [value] * 4
                loaded, = load_nepse_daily_bars(self.write_bars(",".join(row)))
                self.assertEqual(loaded.close, Decimal(value))

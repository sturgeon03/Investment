from __future__ import annotations

import argparse
import unittest

from us_invest_ai.transformer_sweep import format_objective_name, parse_int_grid, parse_optional_float_grid


class TransformerSweepTests(unittest.TestCase):
    def test_parse_int_grid_parses_comma_list(self) -> None:
        self.assertEqual(parse_int_grid("4, 8,16"), [4, 8, 16])

    def test_parse_int_grid_rejects_empty_values(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_int_grid(" , ")

    def test_parse_optional_float_grid_supports_none_and_values(self) -> None:
        self.assertEqual(parse_optional_float_grid("none, 0.9, raw, 0.95"), [None, 0.9, None, 0.95])

    def test_parse_optional_float_grid_rejects_empty_values(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_optional_float_grid(" , ")

    def test_format_objective_name_formats_raw_and_quantiles(self) -> None:
        self.assertEqual(format_objective_name(None), "raw")
        self.assertEqual(format_objective_name(0.95), "clip_q95")
        self.assertEqual(format_objective_name(0.925), "clip_q92p5")


if __name__ == "__main__":
    unittest.main()

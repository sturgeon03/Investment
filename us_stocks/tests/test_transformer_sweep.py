from __future__ import annotations

import argparse
import unittest

from us_invest_ai.transformer_sweep import parse_int_grid


class TransformerSweepTests(unittest.TestCase):
    def test_parse_int_grid_parses_comma_list(self) -> None:
        self.assertEqual(parse_int_grid("4, 8,16"), [4, 8, 16])

    def test_parse_int_grid_rejects_empty_values(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_int_grid(" , ")


if __name__ == "__main__":
    unittest.main()

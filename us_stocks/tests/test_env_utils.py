from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from us_invest_ai.env_utils import load_env_file


class EnvUtilsTests(unittest.TestCase):
    def test_load_env_file_sets_key_value_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                """
# comment
SEC_USER_AGENT=InvestmentResearch user@example.com
export OPENAI_API_KEY="dummy"
""".strip(),
                encoding="utf-8",
            )

            loaded = load_env_file(env_path)

        self.assertEqual(loaded["SEC_USER_AGENT"], "InvestmentResearch user@example.com")
        self.assertEqual(loaded["OPENAI_API_KEY"], "dummy")
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "dummy")


if __name__ == "__main__":
    unittest.main()

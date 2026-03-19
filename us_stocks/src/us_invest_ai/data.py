from __future__ import annotations

import sys

from invest_ai_core import market_data as _shared_market_data


sys.modules[__name__] = _shared_market_data

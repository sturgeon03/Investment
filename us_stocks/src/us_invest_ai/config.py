from __future__ import annotations

import sys

from invest_ai_core import config as _shared_config


sys.modules[__name__] = _shared_config

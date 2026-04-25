"""
VELVETFRUIT_EXTRACT + VEV_* (Round 3 options on extract).

Duplicate of `trading_algos/ROUND3/tryout.py` — edit that file to change behavior.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_canon = _root / "trading_algos" / "ROUND3" / "tryout.py"
_spec = importlib.util.spec_from_file_location("tryout_canonical", _canon)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)
Trader = _mod.Trader

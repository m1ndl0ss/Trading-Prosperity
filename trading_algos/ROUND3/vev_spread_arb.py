"""
VEV options — model-free call-spread mean reversion.

Motivation:
  Empirical analysis shows the VEV options are priced too efficiently for
  single-option BS scalping (flat-IV residuals < 3 units = inside bid-ask).
  BUT: the SPREAD between adjacent strikes is a more stable quantity that
  can be mean-reverted without any model or implied vol.

Theory:
  For European calls on the same underlying:
    C(K1) − C(K2)  ∈  [0, K2 − K1]   (no-arb bounds, never violated in data)

  The spread C(K1) − C(K2) ≈ (K2−K1) × P(S_T > midpoint)
  — it's a scaled probability, so it should be stable and mean-reverting
  around a level set by the current underlying price.

  When C(K1) − C(K2) is unexpectedly WIDE (market overestimates the bull
  spread), we SHORT the spread (sell K1, buy K2).
  When it is unexpectedly NARROW, we LONG the spread (buy K1, sell K2).

Pairs traded (all strike gaps = 100):
  (5100, 5200):  mean≈71.3  spread gap=100
  (5200, 5300):  mean≈48.8  spread gap=100
  (5300, 5400):  mean≈30.8  spread gap=100
  (5400, 5500):  mean≈9.3   spread gap=100

  Skipped: (5000,5100) because VEV_5000 has wide bid-ask (6 units);
  execution cost exceeds typical spread deviation.

Why this is model-free:
  - No Black-Scholes, no IV, no TTE, no vol surface.
  - We only need the RELATIVE price between two strikes.
  - The mean and std are calibrated directly from 30,000 training ticks.
  - This approach remains valid even if the underlying's vol or TTE model
    is wrong — we're trading a RATIO, not an absolute level.

Entry/exit logic:
  At each tick, compute spread_obs = C(K1)_mid − C(K2)_mid.
  Compare to MEAN[pair] ± ENTRY_THRESH × STD[pair].
  If spread_obs > MEAN + ENTRY_THRESH × STD:  SHORT spread (sell K1, buy K2).
  If spread_obs < MEAN - ENTRY_THRESH × STD:  LONG  spread (buy K1, sell K2).
  No explicit exit: positions are closed by the opposite signal firing, or
  held to end-of-day (marked at mid).

Delta exposure:
  A bull-call spread has net delta = delta(K1) − delta(K2) ≈ 0.2–0.5.
  For clip=10, max VF-equivalent exposure = 10 × 0.5 = 5 units. Trivial.

No VELVETFRUIT_EXTRACT trading. No BS. No vol.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ── Products ──────────────────────────────────────────────────────────────────
_ALL_PRODUCTS = (
    "VELVETFRUIT_EXTRACT",
    "VEV_4000", "VEV_4500",
    "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500",
    "VEV_6000", "VEV_6500",
)

# ── Spread pairs: (K_lower, K_upper, historical_mean, historical_std) ─────────
# Calibrated from 30,000-tick training data (mid prices).
# Mean = E[C(K_lower) − C(K_upper)], Std = σ of that difference.
# NOTE: these are from training; re-calibrate if live data differs significantly.
_PAIRS: Tuple[Tuple[int, int, float, float], ...] = (
    (5100, 5200, 71.26, 6.0),   # std estimated from range / 6
    (5200, 5300, 48.79, 5.5),
    (5300, 5400, 30.81, 4.0),
    (5400, 5500,  9.31, 2.5),
)

# ── Parameters ────────────────────────────────────────────────────────────────
PARAMS: Dict[str, Any] = {
    # Entry at this many standard deviations from the historical mean.
    # 1.5 σ ≈ 7% of ticks; 2.0 σ ≈ 2% of ticks.
    "entry_sigma": 1.5,
    # Units to trade per leg when a signal fires.
    "clip": 10,
    # Position limit per option (official Round 3 rules: 300 per VEV voucher).
    "opt_limit": 300,
}


def _mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0


class Trader:
    def __init__(self) -> None:
        self.params = dict(PARAMS)

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        p          = self.params
        entry_sig  = float(p["entry_sigma"])
        clip       = int(p["clip"])
        lim        = int(p["opt_limit"])

        result: Dict[str, List[Order]] = {sym: [] for sym in _ALL_PRODUCTS}

        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                pass

        for K_lo, K_hi, mean, std in _PAIRS:
            sym_lo = f"VEV_{K_lo}"
            sym_hi = f"VEV_{K_hi}"

            d_lo = state.order_depths.get(sym_lo)
            d_hi = state.order_depths.get(sym_hi)
            if not d_lo or not d_hi:
                continue

            mid_lo = _mid(d_lo)
            mid_hi = _mid(d_hi)
            if mid_lo is None or mid_hi is None:
                continue

            spread_obs = mid_lo - mid_hi
            z = (spread_obs - mean) / std

            pos_lo = int(state.position.get(sym_lo, 0))
            pos_hi = int(state.position.get(sym_hi, 0))

            orders_lo: List[Order] = []
            orders_hi: List[Order] = []

            if z > entry_sig:
                # Spread too WIDE → short K_lo (overpriced), long K_hi (underpriced).
                # Sell K_lo at best_bid, Buy K_hi at best_ask.
                sq = min(clip, lim + pos_lo)
                bq = min(clip, lim - pos_hi)
                if sq > 0 and d_lo.buy_orders:
                    best_bid_lo = max(d_lo.buy_orders)
                    orders_lo.append(Order(sym_lo, best_bid_lo, -sq))
                if bq > 0 and d_hi.sell_orders:
                    best_ask_hi = min(d_hi.sell_orders)
                    orders_hi.append(Order(sym_hi, best_ask_hi, bq))

            elif z < -entry_sig:
                # Spread too NARROW → long K_lo (underpriced), short K_hi (overpriced).
                # Buy K_lo at best_ask, Sell K_hi at best_bid.
                bq = min(clip, lim - pos_lo)
                sq = min(clip, lim + pos_hi)
                if bq > 0 and d_lo.sell_orders:
                    best_ask_lo = min(d_lo.sell_orders)
                    orders_lo.append(Order(sym_lo, best_ask_lo, bq))
                if sq > 0 and d_hi.buy_orders:
                    best_bid_hi = max(d_hi.buy_orders)
                    orders_hi.append(Order(sym_hi, best_bid_hi, -sq))

            if orders_lo:
                result[sym_lo] = orders_lo
            if orders_hi:
                result[sym_hi] = orders_hi

        return result, 0, json.dumps(td, separators=(",", ":"))

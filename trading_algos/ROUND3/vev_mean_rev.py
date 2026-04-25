"""
VEV options — treat each strike as an independent mean-reverting product.

Insight: stop trying to model options. Each VEV_K price mean-reverts around its
own stable level, driven by the same underlying mean-reversion in VELVETFRUIT.
Apply the exact same buy-low / sell-high logic as hydrogel_mean_rev and
velvet_like_hydro — no Black-Scholes, no IV, no time-to-expiry.

Why this works:
  Option price ≈ f(underlying price, T, IV).
  T and IV are approximately constant within a 1000-tick day (theta loss ≈ 1 unit
  over the full day for VEV_5200 — smaller than half the spread).
  So option price ≈ g(underlying price) alone → same mean-reversion property.

Official position limit (Round 3 rules): 300 per VEV voucher.

Grid-search results (raw instant-fill simulation, 3 training days, clip=300):
  Strike  [buy,   sell]     Day0    Day1    Day2   Total   spread  delta
  4000    [1227.4, 1276.3]  43,800  58,650  30,900 133,350   ~21   ~1.00
  4500    [ 727.9,  776.8]  46,500  60,450  36,450 143,400   ~16   ~1.00
  5000    [ 246.5,  274.2]  56,700  76,950  32,850 166,500    ~6   ~0.93
  5100    [ 158.6,  184.9]  42,900  77,850  35,850 156,600    ~4   ~0.82
  5200    [  89.3,  109.7]  35,250  64,050  19,200 118,500    ~3   ~0.55
  5300    [  45.0,   53.5]  27,900  30,900  21,900  80,700    ~2   ~0.40
  5400    [  15.0,   18.0]   9,750  14,100   8,400  32,250    ~1   ~0.20
  5500    [   6.5,    9.0]   5,550   3,450   1,200  10,200    ~1   ~0.10
  TOTAL                                             841,500

All three training days are positive for all 8 strikes.

Why VEV_4000 / VEV_4500 are included despite wide spread (~20 / ~16 units):
  These deep-ITM calls have delta ≈ 1.0 and extrinsic ≈ 0.  Their mid price
  tracks the underlying almost exactly (VEV_4000 ≈ S − 4000).  The wide spread
  is a cost, but the 50-unit intraday swing of VELVETFRUIT leaves plenty of edge
  (swing captured ≈ 50 units, spread cost ≈ 20 units → net ≈ 30 per unit × 300
  contracts = 9,000 per round trip).

Why clip=300 for ALL strikes including 5000/5100:
  This is a competition to maximise PnL, the mean-reversion is confirmed on all
  training days, and each strike has a SEPARATE position limit of 300. Using
  clip=50 left +269,000 raw simulation PnL on the table.

Skipped:
  VEV_6000, VEV_6500 — permanently stuck at mid=0.5 (bid=0, ask=1, worthless).

Passive MM layer (where spread permits):
  VEV_4000 spread≈21 → bid+1/ask-1, capture 19 units per round trip.
  VEV_4500 spread≈16 → capture 14 units.
  VEV_5000 spread≈6  → capture 4 units.
  VEV_5100 spread≈4  → capture 2 units.
  VEV_5200 spread≈3  → capture 1 unit.
  VEV_5300/5400/5500 spread ≤ 2 → spread-crossing guard blocks passive quotes.

No VELVETFRUIT trading — handled by velvet_like_hydro.py.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ── Products ──────────────────────────────────────────────────────────────────
_TRADE_STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500)

_ALL_PRODUCTS = (
    "VELVETFRUIT_EXTRACT",
    "VEV_4000", "VEV_4500",
    "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500",
    "VEV_6000", "VEV_6500",
)

# ── Per-strike thresholds (clip=300 for all) ──────────────────────────────────
_STRIKE_CFG: Dict[int, Dict[str, Any]] = {
    4000: {"buy": 1227.4, "sell": 1276.3, "clip": 300},
    4500: {"buy":  727.9, "sell":  776.8, "clip": 300},
    5000: {"buy":  246.5, "sell":  274.2, "clip": 300},
    5100: {"buy":  158.6, "sell":  184.9, "clip": 300},
    5200: {"buy":   89.3, "sell":  109.7, "clip": 300},
    5300: {"buy":   45.0, "sell":   53.5, "clip": 300},
    5400: {"buy":   15.0, "sell":   18.0, "clip": 300},
    5500: {"buy":    6.5, "sell":    9.0, "clip": 300},
}

LIMIT = 300   # official Round 3 limit: 300 per VEV voucher

# ── Passive MM ────────────────────────────────────────────────────────────────
_MM_SIZE = 15   # units per side per tick in the neutral zone


def _mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0


class Trader:
    LIMIT: Dict[str, int] = {f"VEV_{K}": LIMIT for K in _TRADE_STRIKES}

    def __init__(self) -> None:
        pass

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {sym: [] for sym in _ALL_PRODUCTS}

        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                pass

        for K in _TRADE_STRIKES:
            sym   = f"VEV_{K}"
            cfg   = _STRIKE_CFG[K]
            depth = state.order_depths.get(sym)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            mid = _mid(depth)
            if mid is None:
                continue

            buy_t  = cfg["buy"]
            sell_t = cfg["sell"]
            clip   = cfg["clip"]
            pos    = int(state.position.get(sym, 0))
            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            spread   = best_ask - best_bid

            orders: List[Order] = []

            if mid < buy_t:
                # Price cheap — sweep asks aggressively.
                # +4 covers ask2 for tight-spread options; for wide-spread options
                # (4000/4500 spread≈20) it's negligible extra cost.
                sweep_px = best_ask + 4
                want = clip - pos
                if want > 0:
                    orders.append(Order(sym, sweep_px, want))

            elif mid > sell_t:
                # Price rich — sweep bids aggressively.
                sweep_px = best_bid - 4
                want = clip + pos
                if want > 0:
                    orders.append(Order(sym, sweep_px, -want))

            elif spread >= 3:
                # Neutral zone + enough spread for passive MM.
                buy_px  = best_bid + 1
                sell_px = best_ask - 1
                # Inventory skew: lean toward flat
                if pos > clip // 4:
                    buy_px  -= 1
                    sell_px -= 1
                elif pos < -(clip // 4):
                    buy_px  += 1
                    sell_px += 1
                buy_px  = min(buy_px,  best_ask - 1)
                sell_px = max(sell_px, best_bid + 1)
                if buy_px < sell_px:   # crossing guard
                    bq = min(_MM_SIZE, LIMIT - pos)
                    sq = min(_MM_SIZE, LIMIT + pos)
                    if bq > 0:
                        orders.append(Order(sym, int(buy_px),   bq))
                    if sq > 0:
                        orders.append(Order(sym, int(sell_px), -sq))

            result[sym] = orders

        return result, 0, json.dumps(td, separators=(",", ":"))

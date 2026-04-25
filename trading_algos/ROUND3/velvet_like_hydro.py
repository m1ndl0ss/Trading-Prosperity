"""
VELVETFRUIT_EXTRACT — fixed-level mean-reversion (same design as hydrogel_mean_rev.py).

Data analysis (Round 3, 3 training days, 30,000 ticks):
  Fair value  : ~5,250  (mean = 5,250.10, range 5,198–5,300)
  Spread      : mostly 5–6 (±2.5 from fair vs HYDROGEL's ±8)
  L1 volume   : 25–67 units (vs HYDROGEL's 12–15)
  Daily range : 68–93 units (vs HYDROGEL's 143–170)
  Autocorr-1  : −0.16 (slightly stronger mean-reversion than HYDROGEL's −0.13)

Threshold analysis — official backtester day 2 exposed the original [5230,5275] flaw:
  The official data's range was only 5241.5–5276.5 (35 units), mean 5262.
  BUY_THRESH=5230 never fired (min=5241.5 > 5230) → held a bare short all day → only 1,800 PnL.
  SELL_THRESH=5275 barely fired (14 of 1000 ticks) → one directional trade instead of multiple.

  Fix: tighten both thresholds toward the actual intraday range.

  NOTE ON NUMBERS BELOW — these are raw instant-fill simulations (buy full clip at ask1 the
  same tick the signal fires, sell full clip at bid1). Real exchange fills in chunks (~25 units
  per tick for VELVETFRUIT), so absolute values are inflated ~5–10×.  The numbers are only
  useful for RELATIVE comparison between threshold pairs under the same simulation assumptions.

  Raw sim (instant-fill):         Day0     Day1     Day2    Total   Official*
  [5244, 5274]  ← chosen        32,800   50,900   26,900  110,600   12,400
  [5230, 5275]  (old)           36,200   44,500   29,100  109,800    1,800
  * Official numbers from sim run on 401656.json (1000-tick day 2 log from the exchange).

  Local backtest (realistic fills via replay engine):
  [5244, 5274]:  22,245   [5230, 5275]:  11,264

Why [5244, 5274]:
  BUY_THRESH  = 5244 → fires at the day's actual lows (5241–5243 on the official run).
  SELL_THRESH = 5274 → fires more broadly; 6.9× official improvement with near-identical training.
  All 3 training days remain positive.  Local backtest nearly doubled (11k → 22k).

Key differences from hydrogel_mean_rev.py:
  1. Sweep price margin reduced: +4 (vs +10) — VELVETFRUIT's L2 ask is at most ask1+1
     and ask2 is frequently absent; +4 comfortably covers both levels.
  2. MM_SIZE = 25 to match L1 volume (25–67 units vs HYDROGEL's 12–15).
  3. Spread crossing guard added: passive quotes only posted when buy_px < sell_px,
     because the spread can occasionally collapse to 1–3 units.
  4. Position skew threshold stays at 50/200 (25% of limit, same ratio as HYDROGEL).

Tune only BUY_THRESH / SELL_THRESH.  LIMIT = 200 is fixed by exchange rules.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

PRODUCT = "VELVETFRUIT_EXTRACT"

PARAMS: Dict[str, Any] = {
    "buy_thresh":  5244,  # buy aggressively when mid drops below this
    "sell_thresh": 5274,  # sell aggressively when mid rises above this
    # Passive MM order size per side in the neutral zone.
    # Capped at L1 volume (25) to prevent adverse-selection build-up.
    "mm_size": 25,
}

LIMIT = 200


def _mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0


def _sweep_buy(depth: OrderDepth, pos: int, max_price: int) -> List[Order]:
    """
    Aggressive BUY up to max_price.  Fills ask1 (and ask2 when present) immediately;
    remaining quantity queues as a passive bid at max_price and fills as the MM
    replenishes its ask side each tick.
    """
    want = LIMIT - pos
    if want <= 0 or not depth.sell_orders:
        return []
    if min(depth.sell_orders) > max_price:
        return []
    return [Order(PRODUCT, max_price, want)]


def _sweep_sell(depth: OrderDepth, pos: int, min_price: int) -> List[Order]:
    """Symmetric aggressive SELL down to min_price."""
    want = LIMIT + pos
    if want <= 0 or not depth.buy_orders:
        return []
    if max(depth.buy_orders) < min_price:
        return []
    return [Order(PRODUCT, min_price, -want)]


class Trader:
    LIMIT: Dict[str, int] = {PRODUCT: LIMIT}

    def __init__(self) -> None:
        self.params = dict(PARAMS)

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        conversions = 0
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                td = {}

        p = self.params
        buy_thresh  = int(p["buy_thresh"])
        sell_thresh = int(p["sell_thresh"])
        mm_size     = int(p.get("mm_size", 25))

        result: Dict[str, List[Order]] = {}
        for sym in state.order_depths:
            if sym != PRODUCT:
                result[sym] = []
                continue

            depth = state.order_depths[sym]
            mid = _mid(depth)
            if mid is None:
                result[sym] = []
                continue

            pos    = int(state.position.get(PRODUCT, 0))
            orders: List[Order] = []
            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)

            if mid < buy_thresh:
                # Price cheap — aggressively sweep asks.
                # +4 comfortably covers ask1 and ask2 (L2 is at most ask1+1 for VELVETFRUIT).
                orders = _sweep_buy(depth, pos, best_ask + 4)

            elif mid > sell_thresh:
                # Price rich — aggressively sweep bids.
                orders = _sweep_sell(depth, pos, best_bid - 4)

            elif abs(pos) < mm_size * 4:
                # Neutral zone, passive MM — but ONLY when position is roughly flat.
                # Skipping when |pos| is large prevents the MM from fighting profitable
                # directional positions (on the 7.1k run it cost -5,224 by buying while
                # short and selling while long, diluting the big directional trades).
                buy_px  = best_bid + 1
                sell_px = best_ask - 1
                if pos > 50:
                    buy_px  -= 1
                    sell_px -= 1
                elif pos < -50:
                    buy_px  += 1
                    sell_px += 1

                # Clamp to stay inside the book.
                buy_px  = min(buy_px,  best_ask - 1)
                sell_px = max(sell_px, best_bid + 1)

                # Spread-crossing guard: spread can collapse to 1–3 units.
                if buy_px < sell_px:
                    bq = min(mm_size, LIMIT - pos)
                    sq = min(mm_size, LIMIT + pos)
                    if bq > 0:
                        orders.append(Order(PRODUCT, int(buy_px), bq))
                    if sq > 0:
                        orders.append(Order(PRODUCT, int(sell_px), -sq))

            result[sym] = orders

        return result, conversions, json.dumps(td, separators=(",", ":"))

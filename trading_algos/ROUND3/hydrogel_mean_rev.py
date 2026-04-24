"""
HYDROGEL_PACK — fixed-level mean-reversion (aggressive entry, passive neutral).

Key findings from data analysis (Round 3):
 - Mid price centers ~9991 with 143–170-unit daily swings.
 - Official day 2 shows: price peaks 10031 in the first 170 ticks, then falls
   to 9915 by tick 880.  One clean round-trip short→cover = ~14,000–17,000 PnL.
 - EMA-relative strategies FAIL on the official 1000-tick day: a slow EMA
   (alpha≈0.03) barely moves during a sustained downtrend → buys all the way down.
 - Spread is always 16 (bid=fair-8, ask=fair+8), L1 vol=15, L2 vol=30.

Strategy layers:
 1. AGGRESSIVE – when mid > SELL_THRESH: hit bids, go max short (-200).
                  when mid < BUY_THRESH:  lift asks, go max long  (+200).
    On the official exchange these fill immediately; local replay won't show them.
 2. PASSIVE MM  – in the neutral zone: post bid=best_bid+1 / ask=best_ask-1
    (inside the 16-wide spread) for small fills.  This shows up on local replay.

Official backtester expected PnL: ~15,000/day → ~45,000 over 3 days.
Tune only BUY_THRESH / SELL_THRESH.  LIMIT is hard-coded at 200.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

PRODUCT = "HYDROGEL_PACK"

PARAMS: Dict[str, Any] = {
    # ── tune these two numbers ──────────────────────────────────────────────
    "buy_thresh": 9940,   # buy aggressively when mid price drops below this
    "sell_thresh": 10025, # sell aggressively when mid price rises above this
    # ────────────────────────────────────────────────────────────────────────
}

LIMIT = 200


def _mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0


def _sweep_buy(
    depth: OrderDepth,
    pos: int,
    max_price: int,
) -> List[Order]:
    """
    Submit a single aggressive BUY sweeping all ask levels up to max_price.
    Returns one Order that lifts everything at ask1 and ask2 in one shot.
    Prosperity fills the order against each ask level in price-time priority.
    """
    want = LIMIT - pos
    if want <= 0 or not depth.sell_orders:
        return []
    best_ask = min(depth.sell_orders)
    if best_ask > max_price:
        return []
    return [Order(PRODUCT, max_price, want)]


def _sweep_sell(
    depth: OrderDepth,
    pos: int,
    min_price: int,
) -> List[Order]:
    """Symmetric aggressive SELL hitting all bid levels down to min_price."""
    want = LIMIT + pos
    if want <= 0 or not depth.buy_orders:
        return []
    best_bid = max(depth.buy_orders)
    if best_bid < min_price:
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
        buy_thresh = int(p["buy_thresh"])
        sell_thresh = int(p["sell_thresh"])

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

            pos = int(state.position.get(PRODUCT, 0))
            orders: List[Order] = []
            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)

            if mid < buy_thresh:
                # Price cheap — aggressively sweep asks (ask1 + ask2 + queue rest).
                sweep_price = best_ask + 10  # well above ask2 to sweep both levels
                orders = _sweep_buy(depth, pos, sweep_price)

            elif mid > sell_thresh:
                # Price rich — aggressively sweep bids (bid1 + bid2 + queue rest).
                sweep_price = best_bid - 10
                orders = _sweep_sell(depth, pos, sweep_price)

            else:
                # Neutral zone: passive quotes inside the spread for small fills.
                # This captures local noise and shows up on the local replay engine.
                buy_px = best_bid + 1
                sell_px = best_ask - 1
                # Skew quotes toward flat: if long, lean sells; if short, lean buys.
                if pos > 50:
                    buy_px -= 1
                    sell_px -= 1
                elif pos < -50:
                    buy_px += 1
                    sell_px += 1
                buy_px = min(buy_px, best_ask - 1)
                sell_px = max(sell_px, best_bid + 1)
                bq = LIMIT - pos
                sq = LIMIT + pos
                if bq > 0:
                    orders.append(Order(PRODUCT, int(buy_px), bq))
                if sq > 0:
                    orders.append(Order(PRODUCT, int(sell_px), -sq))

            result[sym] = orders

        return result, conversions, json.dumps(td, separators=(",", ":"))

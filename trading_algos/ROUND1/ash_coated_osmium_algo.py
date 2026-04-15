"""
ASH_COATED_OSMIUM: one tick inside best bid / best ask, with refinements:

 (1) Position limit 25 — matches typical tape (fills ≤10; avoids oversized quotes).
  (2) One-sided book: fade the dislocation with larger size at best_ask−1 (no bid)
      or best_bid+1 (no ask); ~62% of ±8 jumps revert next tick in the data.
  (5) Mean-reversion skew: store last mid in traderData; after a large mid move,
      scale down the buy size and up the sell size (and vice versa).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from datamodel import Order, TradingState

SYMBOL = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 25

# Extra size on the fade side when only one book side exists (see analysis).
FADE_SIZE_MULT = 1.35

# skew = clamp(mid_change / MID_CHANGE_SCALE, -1, 1); scales quote sizes.
MID_CHANGE_SCALE = 4.0
SKEW_HALF = 0.5  # bid_scale = 1 - SKEW_HALF*skew, ask_scale = 1 + SKEW_HALF*skew

TD_KEY = "ash_osmium"


def _load_td(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return dict(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return {}


def _implied_mid(
    has_bid: bool,
    has_ask: bool,
    best_bid: int | None,
    best_ask: int | None,
) -> float | None:
    if has_bid and has_ask and best_bid is not None and best_ask is not None:
        return (float(best_bid) + float(best_ask)) / 2.0
    if has_ask and best_ask is not None:
        return float(best_ask)
    if has_bid and best_bid is not None:
        return float(best_bid)
    return None


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td_in = getattr(state, "traderData", "") or ""
        td_dict = _load_td(td_in)
        blob: Dict[str, Any] = dict(td_dict.get(TD_KEY, {}))
        last_mid = blob.get("last_mid")
        if last_mid is not None:
            try:
                last_mid_f = float(last_mid)
            except (TypeError, ValueError):
                last_mid_f = None
        else:
            last_mid_f = None

        depth = state.order_depths.get(SYMBOL)
        if depth is None:
            new_td = json.dumps(td_dict, separators=(",", ":"))
            return {}, 0, new_td

        has_bid = bool(depth.buy_orders)
        has_ask = bool(depth.sell_orders)
        best_bid = max(depth.buy_orders) if has_bid else None
        best_ask = min(depth.sell_orders) if has_ask else None

        mid = _implied_mid(has_bid, has_ask, best_bid, best_ask)
        mid_change = 0.0
        if mid is not None and last_mid_f is not None:
            mid_change = mid - last_mid_f
        if mid is not None:
            blob["last_mid"] = mid
        td_dict[TD_KEY] = blob
        new_td = json.dumps(td_dict, separators=(",", ":"))

        if not has_bid and not has_ask:
            return {}, 0, new_td

        pos = state.position.get(SYMBOL, 0)
        skew = max(-1.0, min(1.0, mid_change / MID_CHANGE_SCALE))
        bid_scale = 1.0 - SKEW_HALF * skew
        ask_scale = 1.0 + SKEW_HALF * skew

        orders: List[Order] = []

        # ── (2) One-sided book: fade with extra size on the passive quote ──
        if has_ask and not has_bid and best_ask is not None:
            ask_px = best_ask - 1
            if ask_px >= best_ask:
                return {SYMBOL: orders}, 0, new_td
            base_ask = -POSITION_LIMIT - pos
            if base_ask < 0:
                faded = int(round(base_ask * FADE_SIZE_MULT))
                max_mag = pos + POSITION_LIMIT
                ask_vol = max(faded, -max_mag)
                ask_vol = int(round(ask_vol * ask_scale))
                ask_vol = max(ask_vol, -max_mag)
                if ask_vol < 0:
                    orders.append(Order(SYMBOL, ask_px, ask_vol))
            return {SYMBOL: orders}, 0, new_td

        if has_bid and not has_ask and best_bid is not None:
            bid_px = best_bid + 1
            if bid_px <= best_bid:
                return {SYMBOL: orders}, 0, new_td
            base_bid = POSITION_LIMIT - pos
            if base_bid > 0:
                faded = int(round(base_bid * FADE_SIZE_MULT))
                bid_vol = min(faded, POSITION_LIMIT - pos)
                bid_vol = int(round(bid_vol * bid_scale))
                bid_vol = min(max(bid_vol, 0), POSITION_LIMIT - pos)
                if bid_vol > 0:
                    orders.append(Order(SYMBOL, bid_px, bid_vol))
            return {SYMBOL: orders}, 0, new_td

        # ── Two-sided: one inside bid/ask + (5) skew ──
        if best_bid is None or best_ask is None:
            return {SYMBOL: orders}, 0, new_td
        if best_bid >= best_ask:
            return {SYMBOL: orders}, 0, new_td

        bid_px = best_bid + 1
        ask_px = best_ask - 1
        if bid_px >= ask_px:
            return {SYMBOL: orders}, 0, new_td

        base_bid_vol = POSITION_LIMIT - pos
        base_ask_vol = -POSITION_LIMIT - pos

        bid_vol = int(round(base_bid_vol * bid_scale)) if base_bid_vol > 0 else 0
        bid_vol = min(max(bid_vol, 0), POSITION_LIMIT - pos)

        ask_vol = base_ask_vol
        if base_ask_vol < 0:
            max_mag = pos + POSITION_LIMIT
            ask_vol = int(round(base_ask_vol * ask_scale))
            ask_vol = max(ask_vol, -max_mag)

        if bid_vol > 0:
            orders.append(Order(SYMBOL, bid_px, bid_vol))
        if ask_vol < 0:
            orders.append(Order(SYMBOL, ask_px, ask_vol))

        return {SYMBOL: orders}, 0, new_td

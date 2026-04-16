"""
Copy of test3_tasos_changes.py with an enhanced ASH_COATED_OSMIUM leg.

Pepper: unchanged (first-10 trend → buy/hold or sell/hold).

Osmium: still maker-first, but adds:
  - Fair reference ~10_000 (pearls analogue; see Stanford Cardinal R1 notes:
    https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal )
  - Slow EWMA of mid pulled toward that reference (coach: fair vs market)
  - L1 volume imbalance (coach: volume tips the scale)
  - Tick mean-reversion skew from last mid change (-0.5-ish lag-1 in CSV data)
  - One-sided book fade (common dislocation, often snaps back)
  - Wider spread → quote2 ticks inside instead of 1 when safe
  - At most one aggressive lift/hit per side per tick (replay allows one buy + one sell)

Replay (local dashboard) matches only tape prints; real exchange uses full book.
"""

from __future__ import annotations

import json

from datamodel import Order, OrderDepth, TradingState
from typing import Any, Dict, List, Optional

ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

_BOOT_LEN = 10
_TD_OSMIUM = "osmium_enh"

FAIR_REF = 10_000.0
FADE_MULT = 1.35
EWMA_ALPHA = 0.08
ANCHOR_BLEND = 0.5077  # weight on EWMA vs FAIR_REF for "anchor"
TAKE_DISLOC = 5.0  # |mid - anchor| beyond this → consider lift/hit
TAKE_CAP = 18
BASE_SIZE = 40 # per-side cap before skew (below full80 snap-back)


def _loads_td(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _tanh1(x: float) -> float:
    """Cheap bounded nonlinearity in [-1, 1]."""
    return x / (1.0 + abs(x))


class Trader:
    def __init__(self):
        self.limits = {
            ASH_COATED_OSMIUM: 80,
            INTARIAN_PEPPER_ROOT: 80,
        }

        self._pepper_boot: list[float] = []
        self._pepper_hold: Optional[str] = None

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    @staticmethod
    def _pepper_trend_from_first_10(mids: list[float]) -> int:
        if len(mids) < _BOOT_LEN:
            return 0
        d = mids[-1] - mids[0]
        if d > 0:
            return 1
        if d < 0:
            return -1
        step_sum = sum(mids[i + 1] - mids[i] for i in range(_BOOT_LEN - 1))
        if step_sum > 0:
            return 1
        if step_sum < 0:
            return -1
        return 0

    def _orders_osmium(self, depth: OrderDepth, pos: int, limit: int, blob: Dict[str, Any]) -> List[Order]:
        orders: List[Order] = []
        has_bid = bool(depth.buy_orders)
        has_ask = bool(depth.sell_orders)
        if not has_bid and not has_ask:
            return orders

        best_bid = max(depth.buy_orders) if has_bid else None
        best_ask = min(depth.sell_orders) if has_ask else None

        mid: Optional[float] = None
        if has_bid and has_ask and best_bid is not None and best_ask is not None:
            mid = (float(best_bid) + float(best_ask)) / 2.0
        elif has_ask and best_ask is not None:
            mid = float(best_ask)
        elif has_bid and best_bid is not None:
            mid = float(best_bid)
        else:
            return orders

        prev = blob.get("last_mid")
        try:
            prev_f = float(prev) if prev is not None else None
        except (TypeError, ValueError):
            prev_f = None
        mid_change = 0.0 if prev_f is None else (mid - prev_f)

        ema = blob.get("ema")
        try:
            ema_f = float(ema) if ema is not None else FAIR_REF
        except (TypeError, ValueError):
            ema_f = FAIR_REF
        ema_f = (1.0 - EWMA_ALPHA) * ema_f + EWMA_ALPHA * mid
        blob["ema"] = ema_f
        blob["last_mid"] = mid

        anchor = (1.0 - ANCHOR_BLEND) * FAIR_REF + ANCHOR_BLEND * ema_f

        bv = depth.buy_orders.get(best_bid, 0) if best_bid is not None else 0
        av_raw = depth.sell_orders.get(best_ask, 0) if best_ask is not None else 0
        av = abs(int(av_raw))

        skew_m = max(-1.0, min(1.0, mid_change / 4.0))
        imb = (float(bv) - float(av)) / float(bv + av) if (bv + av) > 0 else 0.0
        mean_rev = anchor - mid
        score = (
            0.48 * _tanh1(mean_rev / 8.0)
            - 0.38 * _tanh1(mid_change / 3.0)
            + 0.22 * imb
            - 0.42 * _tanh1(pos / 42.0)
        )

        # ── one-sided fade ─────────────────────────────────────────────
        if has_ask and not has_bid and best_ask is not None:
            ask_px = int(best_ask) - 1
            if ask_px > 0 and int(best_ask) > ask_px:
                base_ask = -limit - pos
                if base_ask < 0:
                    faded = int(round(base_ask * FADE_MULT))
                    max_mag = pos + limit
                    ask_vol = max(faded, -max_mag)
                    ask_scale = 1.0 + 0.35 * skew_m
                    ask_vol = int(round(ask_vol * ask_scale))
                    ask_vol = max(ask_vol, -max_mag)
                    if ask_vol < 0:
                        orders.append(Order(ASH_COATED_OSMIUM, ask_px, ask_vol))
            return orders

        if has_bid and not has_ask and best_bid is not None:
            bid_px = int(best_bid) + 1
            if bid_px > int(best_bid):
                base_bid = limit - pos
                if base_bid > 0:
                    faded = int(round(base_bid * FADE_MULT))
                    bid_vol = min(faded, limit - pos)
                    bid_scale = 1.0 - 0.35 * skew_m
                    bid_vol = int(round(bid_vol * bid_scale))
                    bid_vol = min(max(bid_vol, 0), limit - pos)
                    if bid_vol > 0:
                        orders.append(Order(ASH_COATED_OSMIUM, bid_px, bid_vol))
            return orders

        if best_bid is None or best_ask is None or best_bid >= best_ask:
            return orders

        bb, ba = int(best_bid), int(best_ask)
        spread = ba - bb
        k = 2 if spread >= 18 else 1
        bid_px = bb + k
        ask_px = ba - k
        if bid_px >= ask_px:
            k = 1
            bid_px = bb + 1
            ask_px = ba - 1
            if bid_px >= ask_px:
                return orders

        room_buy = limit - pos
        room_sell = pos + limit
        bid_mult = max(0.22, min(1.48, 1.0 - score))
        ask_mult = max(0.22, min(1.48, 1.0 + score))
        bid_qty = int(min(room_buy, round(BASE_SIZE * bid_mult)))
        ask_qty = int(min(room_sell, round(BASE_SIZE * ask_mult)))

        aggressive_buy = mid < anchor - TAKE_DISLOC and room_buy > 0 and has_ask
        aggressive_sell = mid > anchor + TAKE_DISLOC and room_sell > 0 and has_bid

        if aggressive_buy:
            take_vol = min(TAKE_CAP, room_buy, max(1, av))
            if take_vol > 0:
                orders.append(Order(ASH_COATED_OSMIUM, ba, take_vol))
        elif bid_qty > 0:
            orders.append(Order(ASH_COATED_OSMIUM, bid_px, bid_qty))

        if aggressive_sell:
            take_vol = min(TAKE_CAP, room_sell, max(1, int(bv)))
            if take_vol > 0:
                orders.append(Order(ASH_COATED_OSMIUM, bb, -take_vol))
        elif ask_qty > 0:
            orders.append(Order(ASH_COATED_OSMIUM, ask_px, -ask_qty))

        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        td = _loads_td(state.traderData if state.traderData else "")
        os_blob = td.get(_TD_OSMIUM, {})
        if not isinstance(os_blob, dict):
            os_blob = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            current_position = state.position.get(product, 0)
            max_limit = self.limits.get(product, 20)

            if product == ASH_COATED_OSMIUM:
                orders = self._orders_osmium(order_depth, current_position, max_limit, os_blob)

            elif product == INTARIAN_PEPPER_ROOT:
                mid_price = self.get_mid_price(order_depth)
                if mid_price is not None:
                    if self._pepper_hold is None:
                        self._pepper_boot.append(mid_price)
                        if len(self._pepper_boot) >= _BOOT_LEN:
                            t = self._pepper_trend_from_first_10(self._pepper_boot)
                            if t > 0:
                                self._pepper_hold = "long"
                            elif t < 0:
                                self._pepper_hold = "short"
                            else:
                                self._pepper_hold = "flat"

                    if self._pepper_hold == "long":
                        if current_position < max_limit and order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            want = max_limit - current_position
                            if want > 0:
                                orders.append(Order(product, best_ask, want))

                    elif self._pepper_hold == "short":
                        if current_position > -max_limit and order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            want = current_position + max_limit
                            if want > 0:
                                orders.append(Order(product, best_bid, -want))

            result[product] = orders

        td[_TD_OSMIUM] = os_blob
        new_td = json.dumps(td, separators=(",", ":"))
        return result, conversions, new_td

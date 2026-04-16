from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState

ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

_BOOT_LEN = 10
_OSMIUM_TD_KEY = "osmium_v2"


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _loads_td(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        out = json.loads(raw)
        if isinstance(out, dict):
            return out
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


class Trader:
    def __init__(self):
        self.limits = {
            ASH_COATED_OSMIUM: 80,
            INTARIAN_PEPPER_ROOT: 80,
        }

        # Pepper: first BOOT_LEN mids decide trend, then buy-and-hold or sell-and-hold.
        self._pepper_boot: list[float] = []
        self._pepper_hold: Optional[str] = None  # "long" | "short" | "flat"

    @staticmethod
    def _get_mid_price(order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        return (best_bid + best_ask) / 2.0

    @staticmethod
    def _pepper_trend_from_first_10(mids: List[float]) -> int:
        """+1 uptrend, -1 downtrend, 0 ambiguous (flat)."""
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

    def _trade_osmium(
        self,
        depth: OrderDepth,
        pos: int,
        limit: int,
        state_blob: Dict[str, Any],
    ) -> List[Order]:
        orders: List[Order] = []

        has_bid = bool(depth.buy_orders)
        has_ask = bool(depth.sell_orders)
        if not has_bid and not has_ask:
            return orders

        best_bid = max(depth.buy_orders) if has_bid else None
        best_ask = min(depth.sell_orders) if has_ask else None

        # Ask side can be stored as negative sizes in some environments.
        bid_l1 = depth.buy_orders[best_bid] if has_bid and best_bid is not None else 0
        ask_raw = depth.sell_orders[best_ask] if has_ask and best_ask is not None else 0
        ask_l1 = abs(ask_raw)

        if has_bid and has_ask and best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
        elif has_ask and best_ask is not None:
            mid = float(best_ask)
        elif has_bid and best_bid is not None:
            mid = float(best_bid)
        else:
            return orders

        prev_mid = state_blob.get("last_mid")
        ema = state_blob.get("ema")
        try:
            prev_mid = float(prev_mid) if prev_mid is not None else None
        except (TypeError, ValueError):
            prev_mid = None
        try:
            ema = float(ema) if ema is not None else None
        except (TypeError, ValueError):
            ema = None

        if ema is None:
            ema = mid
        else:
            ema = 0.90 * ema + 0.10 * mid

        last_move = 0.0 if prev_mid is None else (mid - prev_mid)
        mean_reversion = ema - mid
        imbalance = 0.0
        if bid_l1 + ask_l1 > 0:
            imbalance = (bid_l1 - ask_l1) / float(bid_l1 + ask_l1)

        # Positive score => bullish; negative => bearish.
        score = 0.95 * mean_reversion - 0.65 * last_move + 1.3 * imbalance
        inv_penalty = pos / 45.0
        score -= inv_penalty

        state_blob["last_mid"] = mid
        state_blob["ema"] = ema

        # One-sided books are frequent in this product and often snap back.
        if has_ask and not has_bid and best_ask is not None:
            ask_px = best_ask - 1
            if pos > -limit and ask_px > 0:
                size = int(round((pos + limit) * 0.45))
                size = max(1, min(size, pos + limit))
                orders.append(Order(ASH_COATED_OSMIUM, ask_px, -size))
            return orders
        if has_bid and not has_ask and best_bid is not None:
            bid_px = best_bid + 1
            if pos < limit:
                size = int(round((limit - pos) * 0.45))
                size = max(1, min(size, limit - pos))
                orders.append(Order(ASH_COATED_OSMIUM, bid_px, size))
            return orders

        if best_bid is None or best_ask is None or best_bid >= best_ask:
            return orders

        spread = best_ask - best_bid

        # (A) Selective market taking when quote is clearly away from dynamic fair.
        take_thresh = 4.0
        if best_ask <= ema - take_thresh and pos < limit:
            take_qty = min(abs(depth.sell_orders[best_ask]), limit - pos, 24)
            if take_qty > 0:
                orders.append(Order(ASH_COATED_OSMIUM, best_ask, take_qty))
                pos += take_qty
        if best_bid >= ema + take_thresh and pos > -limit:
            take_qty = min(depth.buy_orders[best_bid], pos + limit, 24)
            if take_qty > 0:
                orders.append(Order(ASH_COATED_OSMIUM, best_bid, -take_qty))
                pos -= take_qty

        room_buy = limit - pos
        room_sell = pos + limit
        if room_buy <= 0 and room_sell <= 0:
            return orders

        # (B) Dynamic maker placement: still inside spread, but side-skewed by score.
        # More aggressive side gets +1 extra tick improvement when confidence is strong.
        bid_improve = 1 + (1 if score > 1.6 and spread >= 12 else 0)
        ask_improve = 1 + (1 if score < -1.6 and spread >= 12 else 0)

        bid_px = best_bid + bid_improve
        ask_px = best_ask - ask_improve
        bid_px = min(bid_px, best_ask - 1)
        ask_px = max(ask_px, best_bid + 1)
        if bid_px >= ask_px:
            bid_px = best_bid + 1
            ask_px = best_ask - 1
            if bid_px >= ask_px:
                return orders

        base_size = 15 + max(0, (spread - 14) // 2)
        bid_mult = _clip(1.0 + 0.32 * score, 0.25, 1.75)
        ask_mult = _clip(1.0 - 0.32 * score, 0.25, 1.75)

        bid_qty = min(room_buy, int(round(base_size * bid_mult)))
        ask_qty = min(room_sell, int(round(base_size * ask_mult)))

        # Inventory guardrails.
        if pos > 56:
            bid_qty = min(bid_qty, 2)
            ask_qty = max(ask_qty, 20)
        elif pos < -56:
            ask_qty = min(ask_qty, 2)
            bid_qty = max(bid_qty, 20)

        if bid_qty > 0:
            orders.append(Order(ASH_COATED_OSMIUM, bid_px, bid_qty))
        if ask_qty > 0:
            orders.append(Order(ASH_COATED_OSMIUM, ask_px, -ask_qty))

        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        td = _loads_td(state.traderData if state.traderData else "")
        osmium_blob = td.get(_OSMIUM_TD_KEY, {})
        if not isinstance(osmium_blob, dict):
            osmium_blob = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            current_position = state.position.get(product, 0)
            max_limit = self.limits.get(product, 20)

            if product == ASH_COATED_OSMIUM:
                orders = self._trade_osmium(order_depth, current_position, max_limit, osmium_blob)

            elif product == INTARIAN_PEPPER_ROOT:
                mid_price = self._get_mid_price(order_depth)
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

        td[_OSMIUM_TD_KEY] = osmium_blob
        trader_data = json.dumps(td, separators=(",", ":"))
        return result, conversions, trader_data

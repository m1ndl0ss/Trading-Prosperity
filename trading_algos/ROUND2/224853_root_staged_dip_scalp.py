"""
INTARIAN_PEPPER_ROOT — staged dip (same as ``224853_root_staged_dip``) plus optional scalp.

**Base:** Phase 1 to ``_FIRST_TARGET``, phase 2 dip vs EMA to ``_LIMIT``, then
``_FORCE_COMPLETE_TS`` lift for any remainder.

**Scalp (only when position == ``_LIMIT``):** Hit the best bid for up to
``_SCALP_CLIP`` units **only if** ``best_bid > ema + _SCALP_SELL_EMA_PAD`` **and**
``best_bid >= last_buy_px + _SCALP_MIN_EDGE``. The gate must use **bid**, not mid:
``mid`` can sit above the EMA while ``best_bid`` is still below it, which would fill
sells under the EMA if you tested ``mid > ema``.

Each tick we either return that sell or return no sell — there is no separate “leave
working order open”; unfilled intent from a prior tick is not re-submitted unless gates still pass (live engine: confirm how stale orders are cleared).

Refill to ``_LIMIT`` uses the same phase-2 rules (dip or force-complete).

**Last buy price:** Updated from buy fills. Local replay does not populate
``own_trades``; we infer fills via ``pos_run_start`` vs current position and the
VWAP of aggressive buys placed the prior tick (``pending_buy_vwap``). Live runs
can extend this with ``own_trades`` later if buyer ids are present.

``ema`` and ``last_buy_px`` are stored in ``traderData`` (replay reads ``ema`` for charts).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

_LIMIT = 80
_FIRST_TARGET = 67
_EMA_SPAN = 20
_FORCE_COMPLETE_TS = 900_000

_SCALP_CLIP = 8
_SCALP_MIN_EDGE = 32
_SCALP_SELL_EMA_PAD = 0.0


def _ewma_alpha(span: int) -> float:
    return 2.0 / (float(span) + 1.0)
def _mid(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
def _next_ema(prev: Optional[float], mid: float, span: int) -> float:
    a = _ewma_alpha(span)
    if prev is None:
        return mid
    return a * mid + (1.0 - a) * prev
def _apply_fill_inference(td: Dict[str, Any], pos0: int) -> None:
    prev = td.get("pos_run_start")
    if prev is not None:
        try:
            delta = pos0 - int(prev)
        except (TypeError, ValueError):
            delta = 0
        if delta > 0:
            pb = td.get("pending_buy_vwap")
            if pb is not None:
                try:
                    td["last_buy_px"] = float(pb)
                except (TypeError, ValueError):
                    pass
    td.pop("pending_buy_vwap", None)
    td.pop("pending_buy_qty", None)


class Trader:
    def __init__(self) -> None:
        self.limit = _LIMIT
        self.first_target = _FIRST_TARGET
        self.ema_span = _EMA_SPAN
        self.force_complete_ts = _FORCE_COMPLETE_TS
        self.scalp_clip = _SCALP_CLIP
        self.scalp_min_edge = _SCALP_MIN_EDGE
        self.scalp_sell_ema_pad = _SCALP_SELL_EMA_PAD

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                td = {}

        result: Dict[str, List[Order]] = {}
        conversions = 0
        ts = int(state.timestamp)

        for product, od in state.order_depths.items():
            if product != INTARIAN_PEPPER_ROOT:
                result[product] = []
                continue

            pos0 = state.position.get(product, 0)
            _apply_fill_inference(td, pos0)

            mid = _mid(od)
            ema_val: Optional[float] = None
            if mid is not None:
                prev_ema = td.get("ema")
                prev_f = float(prev_ema) if prev_ema is not None else None
                ema_val = _next_ema(prev_f, float(mid), self.ema_span)
                td["ema"] = ema_val

            orders: List[Order] = []
            placed_buy_px = 0
            placed_buy_qty = 0

            if not od.sell_orders or not od.buy_orders:
                td["pos_run_start"] = pos0
                result[product] = orders
                continue

            best_ask = min(od.sell_orders.keys())
            best_bid = max(od.buy_orders.keys())

            if pos0 < self.first_target:
                want = self.first_target - pos0
                if want > 0:
                    orders.append(Order(product, best_ask, want))
                    placed_buy_px += best_ask * want
                    placed_buy_qty += want
            elif pos0 < self.limit:
                want = self.limit - pos0
                dip_ok = ema_val is not None and float(best_ask) < float(ema_val)
                force_ok = ts >= self.force_complete_ts
                if want > 0 and (dip_ok or force_ok):
                    orders.append(Order(product, best_ask, want))
                    placed_buy_px += best_ask * want
                    placed_buy_qty += want
            elif pos0 == self.limit:
                lb = td.get("last_buy_px")
                lb_f = float(lb) if lb is not None else None
                edge_ok = (
                    lb_f is not None
                    and float(best_bid) >= lb_f + float(self.scalp_min_edge)
                )
                bid_above_ema = (
                    ema_val is not None
                    and float(best_bid) > float(ema_val) + float(self.scalp_sell_ema_pad)
                )
                if edge_ok and bid_above_ema:
                    q = min(self.scalp_clip, pos0)
                    if q > 0:
                        orders.append(Order(product, best_bid, -q))

            if placed_buy_qty > 0:
                td["pending_buy_vwap"] = placed_buy_px / placed_buy_qty
                td["pending_buy_qty"] = placed_buy_qty

            td["pos_run_start"] = pos0
            result[product] = orders

        new_td = json.dumps(td, separators=(",", ":"))
        return result, conversions, new_td

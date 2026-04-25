"""
Round 3 combined mean-reversion — one Trader, three strategies:

  1. HYDROGEL_PACK     — logic from hydrogel_mean_rev.py
  2. VELVETFRUIT_EXTRACT — logic from velvet_like_hydro.py
  3. VEV_4000 … VEV_5500 — logic from vev_mean_rev.py (per-strike thresholds, no BS)

HYDROGEL and VELVET use LIMIT 200; each VEV strike uses LIMIT 300 (Round 3 rule).
See the source modules for tuning notes and backtest discussion.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ── HYDROGEL (hydrogel_mean_rev.py) ───────────────────────────────────────────
HYDROGEL = "HYDROGEL_PACK"
HYDRO_PARAMS: Dict[str, Any] = {
    "buy_thresh": 9940,
    "sell_thresh": 10025,
    "mm_size": 15,
}
HYDRO_LIMIT = 200

# ── VELVET (velvet_like_hydro.py) ─────────────────────────────────────────────
VELVET = "VELVETFRUIT_EXTRACT"
VELVET_PARAMS: Dict[str, Any] = {
    "buy_thresh": 5244,
    "sell_thresh": 5274,
    "mm_size": 25,
}
VELVET_LIMIT = 200

# ── VEV (vev_mean_rev.py) ─────────────────────────────────────────────────────
VEV_STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500)
VEV_STRIKE_CFG: Dict[int, Dict[str, Any]] = {
    4000: {"buy": 1227.4, "sell": 1276.3, "clip": 300},
    4500: {"buy": 727.9, "sell": 776.8, "clip": 300},
    5000: {"buy": 246.5, "sell": 274.2, "clip": 300},
    5100: {"buy": 158.6, "sell": 184.9, "clip": 300},
    5200: {"buy": 89.3, "sell": 109.7, "clip": 300},
    5300: {"buy": 45.0, "sell": 53.5, "clip": 300},
    5400: {"buy": 15.0, "sell": 18.0, "clip": 300},
    5500: {"buy": 6.5, "sell": 9.0, "clip": 300},
}
VEV_LIMIT = 300
VEV_MM_SIZE = 15

ALL_SYMBOLS: Tuple[str, ...] = (
    HYDROGEL,
    VELVET,
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
)


def _mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0


def _sweep_hydro_buy(depth: OrderDepth, pos: int, max_price: int) -> List[Order]:
    want = HYDRO_LIMIT - pos
    if want <= 0 or not depth.sell_orders:
        return []
    if min(depth.sell_orders) > max_price:
        return []
    return [Order(HYDROGEL, max_price, want)]


def _sweep_hydro_sell(depth: OrderDepth, pos: int, min_price: int) -> List[Order]:
    want = HYDRO_LIMIT + pos
    if want <= 0 or not depth.buy_orders:
        return []
    if max(depth.buy_orders) < min_price:
        return []
    return [Order(HYDROGEL, min_price, -want)]


def _sweep_velvet_buy(depth: OrderDepth, pos: int, max_price: int) -> List[Order]:
    want = VELVET_LIMIT - pos
    if want <= 0 or not depth.sell_orders:
        return []
    if min(depth.sell_orders) > max_price:
        return []
    return [Order(VELVET, max_price, want)]


def _sweep_velvet_sell(depth: OrderDepth, pos: int, min_price: int) -> List[Order]:
    want = VELVET_LIMIT + pos
    if want <= 0 or not depth.buy_orders:
        return []
    if max(depth.buy_orders) < min_price:
        return []
    return [Order(VELVET, min_price, -want)]


class Trader:
    LIMIT: Dict[str, int] = {
        HYDROGEL: HYDRO_LIMIT,
        VELVET: VELVET_LIMIT,
        **{f"VEV_{K}": VEV_LIMIT for K in VEV_STRIKES},
        "VEV_6000": 300,
        "VEV_6500": 300,
    }

    def __init__(self) -> None:
        self.hydro = dict(HYDRO_PARAMS)
        self.velvet = dict(VELVET_PARAMS)

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                td = {}

        result: Dict[str, List[Order]] = {p: [] for p in ALL_SYMBOLS}
        od = state.order_depths

        # --- HYDROGEL_PACK ---
        h = od.get(HYDROGEL)
        if h and h.buy_orders and h.sell_orders:
            result[HYDROGEL] = self._hydrogel_orders(h, int(state.position.get(HYDROGEL, 0)))

        # --- VELVETFRUIT_EXTRACT ---
        v = od.get(VELVET)
        if v and v.buy_orders and v.sell_orders:
            result[VELVET] = self._velvet_orders(
                v, int(state.position.get(VELVET, 0))
            )

        # --- VEV strikes (4000–5500) ---
        for k in VEV_STRIKES:
            sym = f"VEV_{k}"
            d = od.get(sym)
            if d and d.buy_orders and d.sell_orders:
                result[sym] = self._vev_strike_orders(
                    sym, d, VEV_STRIKE_CFG[k], int(state.position.get(sym, 0))
                )

        return result, 0, json.dumps(td, separators=(",", ":"))

    def _hydrogel_orders(self, depth: OrderDepth, pos: int) -> List[Order]:
        p = self.hydro
        buy_t = int(p["buy_thresh"])
        sell_t = int(p["sell_thresh"])
        mm = int(p.get("mm_size", 15))
        m = _mid(depth)
        if m is None:
            return []
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        orders: List[Order] = []
        if m < buy_t:
            orders = _sweep_hydro_buy(depth, pos, best_ask + 10)
        elif m > sell_t:
            orders = _sweep_hydro_sell(depth, pos, best_bid - 10)
        else:
            buy_px = best_bid + 1
            sell_px = best_ask - 1
            if pos > 50:
                buy_px -= 1
                sell_px -= 1
            elif pos < -50:
                buy_px += 1
                sell_px += 1
            buy_px = min(buy_px, best_ask - 1)
            sell_px = max(sell_px, best_bid + 1)
            bq = min(mm, HYDRO_LIMIT - pos)
            sq = min(mm, HYDRO_LIMIT + pos)
            if bq > 0:
                orders.append(Order(HYDROGEL, int(buy_px), bq))
            if sq > 0:
                orders.append(Order(HYDROGEL, int(sell_px), -sq))
        return orders

    def _velvet_orders(self, depth: OrderDepth, pos: int) -> List[Order]:
        p = self.velvet
        buy_t = int(p["buy_thresh"])
        sell_t = int(p["sell_thresh"])
        mm = int(p.get("mm_size", 25))
        m = _mid(depth)
        if m is None:
            return []
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        orders: List[Order] = []
        if m < buy_t:
            orders = _sweep_velvet_buy(depth, pos, best_ask + 4)
        elif m > sell_t:
            orders = _sweep_velvet_sell(depth, pos, best_bid - 4)
        elif abs(pos) < mm * 4:
            buy_px = best_bid + 1
            sell_px = best_ask - 1
            if pos > 50:
                buy_px -= 1
                sell_px -= 1
            elif pos < -50:
                buy_px += 1
                sell_px += 1
            buy_px = min(buy_px, best_ask - 1)
            sell_px = max(sell_px, best_bid + 1)
            if buy_px < sell_px:
                bq = min(mm, VELVET_LIMIT - pos)
                sq = min(mm, VELVET_LIMIT + pos)
                if bq > 0:
                    orders.append(Order(VELVET, int(buy_px), bq))
                if sq > 0:
                    orders.append(Order(VELVET, int(sell_px), -sq))
        return orders

    def _vev_strike_orders(
        self, sym: str, depth: OrderDepth, cfg: Dict[str, Any], pos: int
    ) -> List[Order]:
        m = _mid(depth)
        if m is None:
            return []
        buy_t = float(cfg["buy"])
        sell_t = float(cfg["sell"])
        clip = int(cfg["clip"])
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        spread = best_ask - best_bid
        orders: List[Order] = []
        if m < buy_t:
            want = clip - pos
            if want > 0:
                orders.append(Order(sym, int(best_ask + 4), want))
        elif m > sell_t:
            want = clip + pos
            if want > 0:
                orders.append(Order(sym, int(best_bid - 4), -want))
        elif spread >= 3:
            buy_px = best_bid + 1
            sell_px = best_ask - 1
            if pos > clip // 4:
                buy_px -= 1
                sell_px -= 1
            elif pos < -(clip // 4):
                buy_px += 1
                sell_px += 1
            buy_px = min(buy_px, best_ask - 1)
            sell_px = max(sell_px, best_bid + 1)
            if buy_px < sell_px:
                bq = min(VEV_MM_SIZE, VEV_LIMIT - pos)
                sq = min(VEV_MM_SIZE, VEV_LIMIT + pos)
                if bq > 0:
                    orders.append(Order(sym, int(buy_px), bq))
                if sq > 0:
                    orders.append(Order(sym, int(sell_px), -sq))
        return orders

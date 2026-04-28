"""
HYDROGEL-only strategy variants for research (single-product replay).
Production params match r4_m14_67_tilt_minseoks_option_strat.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from datamodel import Order, OrderDepth, TradingState

HYDROGEL_PACK = "HYDROGEL_PACK"
HYDRO_PARAMS: Dict[str, Any] = {
    "buy_thresh": 9935,
    "sell_thresh": 10030,
    "unwind_long": 9995,
    "mm_size": 15,
}
HYDRO_LIMIT = 200


def _mid(d: OrderDepth) -> Optional[float]:
    if not d.buy_orders or not d.sell_orders:
        return None
    return (max(d.buy_orders.keys()) + min(d.sell_orders.keys())) / 2.0


def _last_tick_mark_sides(
    state: TradingState, symbol: str, mark: str
) -> tuple[bool, bool]:
    buy = sell = False
    for t in state.market_trades.get(symbol, []):
        if t.buyer == mark:
            buy = True
        if t.seller == mark:
            sell = True
    return buy, sell


def _sweep_buy(d: OrderDepth, pos: int, max_price: int) -> List[Order]:
    want = HYDRO_LIMIT - pos
    if want <= 0 or not d.sell_orders:
        return []
    if min(d.sell_orders.keys()) > max_price:
        return []
    return [Order(HYDROGEL_PACK, max_price, want)]


def _sweep_sell(d: OrderDepth, pos: int, min_price: int) -> List[Order]:
    want = HYDRO_LIMIT + pos
    if want <= 0 or not d.buy_orders:
        return []
    if max(d.buy_orders.keys()) < min_price:
        return []
    return [Order(HYDROGEL_PACK, min_price, -want)]


def _unwind_long(d: OrderDepth, pos: int) -> List[Order]:
    if pos <= 0 or not d.buy_orders:
        return []
    return [Order(HYDROGEL_PACK, max(d.buy_orders.keys()), -pos)]


def _mm_quotes(
    bb: int,
    ba: int,
    pos: int,
    mm: int,
    m14b: bool,
    m14s: bool,
    m38b: bool,
    m38s: bool,
    m14_mode: str,  # "follow" | "none" | "follow_inv_m38" etc.
    m38_mode: str,  # "fade" | "follow" | "none"
) -> tuple[int, int]:
    bp, sp = bb + 1, ba - 1
    if pos > 50:
        bp -= 1
        sp -= 1
    elif pos < -50:
        bp += 1
        sp += 1
    if m14_mode == "follow":
        if m14b and not m14s:
            bp += 1
        elif m14s and not m14b:
            sp -= 1
    if m38_mode == "fade":
        if m38b and not m38s:
            sp -= 1
        elif m38s and not m38b:
            bp += 1
    elif m38_mode == "follow":
        if m38b and not m38s:
            bp += 1
        elif m38s and not m38b:
            sp -= 1
    bp = min(bp, ba - 1)
    sp = max(sp, bb + 1)
    return bp, sp


def _one_symbol_result(o: List[Order]) -> Tuple[Dict[str, List[Order]], int, str]:
    return {HYDROGEL_PACK: o}, 0, ""


class HydroNaiveMMTrader:
    """Quote bb+1/ba-1 (inner), clip mm_size, inventory skew; no sweeps, no marks."""

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        if HYDROGEL_PACK not in state.order_depths:
            return _one_symbol_result([])
        d = state.order_depths[HYDROGEL_PACK]
        if not d.buy_orders or not d.sell_orders:
            return _one_symbol_result([])
        if _mid(d) is None:
            return _one_symbol_result([])
        p = HYDRO_PARAMS
        mm = int(p.get("mm_size", 15))
        pos = int(state.position.get(HYDROGEL_PACK, 0))
        bb = max(d.buy_orders.keys())
        ba = min(d.sell_orders.keys())
        bp, sp = bb + 1, ba - 1
        if pos > 50:
            bp -= 1
            sp -= 1
        elif pos < -50:
            bp += 1
            sp += 1
        bp = min(bp, ba - 1)
        sp = max(sp, bb + 1)
        o: List[Order] = []
        if bp < sp:
            bq = min(mm, HYDRO_LIMIT - pos)
            sq = min(mm, HYDRO_LIMIT + pos)
            if bq > 0:
                o.append(Order(HYDROGEL_PACK, int(bp), bq))
            if sq > 0:
                o.append(Order(HYDROGEL_PACK, int(sp), -sq))
        return _one_symbol_result(o)


class HydroThresholdMRTrader:
    """Sweeps + unwind + inner MM; no M14/M38 (same as production neutral band, no mark tilt)."""

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        if HYDROGEL_PACK not in state.order_depths:
            return _one_symbol_result([])
        d = state.order_depths[HYDROGEL_PACK]
        if not d.buy_orders or not d.sell_orders:
            return _one_symbol_result([])
        mid = _mid(d)
        if mid is None:
            return _one_symbol_result([])
        p = HYDRO_PARAMS
        buy_t = int(p["buy_thresh"])
        sell_t = int(p["sell_thresh"])
        uw = int(p["unwind_long"])
        mm = int(p.get("mm_size", 15))
        pos = int(state.position.get(HYDROGEL_PACK, 0))
        bb = max(d.buy_orders.keys())
        ba = min(d.sell_orders.keys())
        o: List[Order] = []
        if mid > sell_t and pos > -HYDRO_LIMIT:
            o = _sweep_sell(d, pos, bb - 10)
        elif mid < buy_t and pos < HYDRO_LIMIT:
            o = _sweep_buy(d, pos, ba + 10)
        elif mid > uw and pos > 0:
            o = _unwind_long(d, pos)
        else:
            bp, sp = _mm_quotes(
                bb, ba, pos, mm, False, False, False, False, "none", "none"
            )
            bq = min(mm, HYDRO_LIMIT - pos)
            sq = min(mm, HYDRO_LIMIT + pos)
            if bq > 0:
                o.append(Order(HYDROGEL_PACK, int(bp), bq))
            if sq > 0:
                o.append(Order(HYDROGEL_PACK, int(sp), -sq))
        return _one_symbol_result(o)


class HydroMMNoMarksTrader:
    """Production sweeps + unwind + MM with M14 follow only (no M38 lines)."""

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        if HYDROGEL_PACK not in state.order_depths:
            return _one_symbol_result([])
        d = state.order_depths[HYDROGEL_PACK]
        if not d.buy_orders or not d.sell_orders:
            return _one_symbol_result([])
        mid = _mid(d)
        if mid is None:
            return _one_symbol_result([])
        p = HYDRO_PARAMS
        buy_t = int(p["buy_thresh"])
        sell_t = int(p["sell_thresh"])
        uw = int(p["unwind_long"])
        mm = int(p.get("mm_size", 15))
        pos = int(state.position.get(HYDROGEL_PACK, 0))
        bb = max(d.buy_orders.keys())
        ba = min(d.sell_orders.keys())
        m14b, m14s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 14")
        o: List[Order] = []
        if mid > sell_t and pos > -HYDRO_LIMIT:
            o = _sweep_sell(d, pos, bb - 10)
        elif mid < buy_t and pos < HYDRO_LIMIT:
            o = _sweep_buy(d, pos, ba + 10)
        elif mid > uw and pos > 0:
            o = _unwind_long(d, pos)
        else:
            bp, sp = bb + 1, ba - 1
            if pos > 50:
                bp -= 1
                sp -= 1
            elif pos < -50:
                bp += 1
                sp += 1
            if m14b and not m14s:
                bp += 1
            elif m14s and not m14b:
                sp -= 1
            bp = min(bp, ba - 1)
            sp = max(sp, bb + 1)
            bq = min(mm, HYDRO_LIMIT - pos)
            sq = min(mm, HYDRO_LIMIT + pos)
            if bq > 0:
                o.append(Order(HYDROGEL_PACK, int(bp), bq))
            if sq > 0:
                o.append(Order(HYDROGEL_PACK, int(sp), -sq))
        return _one_symbol_result(o)


class HydroProductionHydroCoreTrader:
    """
    Full M14 follow + M38 fade, sweeps, unwind, MM (copy of production HYDRO block).
    No VELVET/VEV to match single-instrument tape replay.
    """

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                pass
        if HYDROGEL_PACK not in state.order_depths:
            return {HYDROGEL_PACK: []}, 0, json.dumps(td, separators=(",", ":"))
        d = state.order_depths[HYDROGEL_PACK]
        if not d.buy_orders or not d.sell_orders:
            return {HYDROGEL_PACK: []}, 0, json.dumps(td, separators=(",", ":"))
        mid = _mid(d)
        if mid is None:
            return {HYDROGEL_PACK: []}, 0, json.dumps(td, separators=(",", ":"))
        p = HYDRO_PARAMS
        buy_t = int(p["buy_thresh"])
        sell_t = int(p["sell_thresh"])
        uw = int(p["unwind_long"])
        mm = int(p.get("mm_size", 15))
        pos = int(state.position.get(HYDROGEL_PACK, 0))
        bb = max(d.buy_orders.keys())
        ba = min(d.sell_orders.keys())
        m14b, m14s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 14")
        m38b, m38s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 38")
        o: List[Order] = []
        if mid > sell_t and pos > -HYDRO_LIMIT:
            o = _sweep_sell(d, pos, bb - 10)
        elif mid < buy_t and pos < HYDRO_LIMIT:
            o = _sweep_buy(d, pos, ba + 10)
        elif mid > uw and pos > 0:
            o = _unwind_long(d, pos)
        else:
            bp, sp = _mm_quotes(
                bb, ba, pos, mm, m14b, m14s, m38b, m38s, "follow", "fade"
            )
            bq = min(mm, HYDRO_LIMIT - pos)
            sq = min(mm, HYDRO_LIMIT + pos)
            if bq > 0:
                o.append(Order(HYDROGEL_PACK, int(bp), bq))
            if sq > 0:
                o.append(Order(HYDROGEL_PACK, int(sp), -sq))
        return {HYDROGEL_PACK: o}, 0, json.dumps(td, separators=(",", ":"))


class HydroFollowM38Trader:
    """Same as production but M38 follow (trend) instead of fade."""

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        if HYDROGEL_PACK not in state.order_depths:
            return _one_symbol_result([])
        d = state.order_depths[HYDROGEL_PACK]
        if not d.buy_orders or not d.sell_orders:
            return _one_symbol_result([])
        mid = _mid(d)
        if mid is None:
            return _one_symbol_result([])
        p = HYDRO_PARAMS
        buy_t = int(p["buy_thresh"])
        sell_t = int(p["sell_thresh"])
        uw = int(p["unwind_long"])
        mm = int(p.get("mm_size", 15))
        pos = int(state.position.get(HYDROGEL_PACK, 0))
        bb = max(d.buy_orders.keys())
        ba = min(d.sell_orders.keys())
        m14b, m14s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 14")
        m38b, m38s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 38")
        o: List[Order] = []
        if mid > sell_t and pos > -HYDRO_LIMIT:
            o = _sweep_sell(d, pos, bb - 10)
        elif mid < buy_t and pos < HYDRO_LIMIT:
            o = _sweep_buy(d, pos, ba + 10)
        elif mid > uw and pos > 0:
            o = _unwind_long(d, pos)
        else:
            bp, sp = _mm_quotes(
                bb, ba, pos, mm, m14b, m14s, m38b, m38s, "follow", "follow"
            )
            bq = min(mm, HYDRO_LIMIT - pos)
            sq = min(mm, HYDRO_LIMIT + pos)
            if bq > 0:
                o.append(Order(HYDROGEL_PACK, int(bp), bq))
            if sq > 0:
                o.append(Order(HYDROGEL_PACK, int(sp), -sq))
        return _one_symbol_result(o)


class HydroUnwindOnlyTrader:
    """No MM / no sweeps: only close longs when mid > unwind threshold."""

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        if HYDROGEL_PACK not in state.order_depths:
            return _one_symbol_result([])
        d = state.order_depths[HYDROGEL_PACK]
        if not d.buy_orders or not d.sell_orders:
            return _one_symbol_result([])
        mid = _mid(d)
        if mid is None:
            return _one_symbol_result([])
        p = HYDRO_PARAMS
        uw = int(p["unwind_long"])
        pos = int(state.position.get(HYDROGEL_PACK, 0))
        o: List[Order] = []
        if mid > uw and pos > 0:
            o = _unwind_long(d, pos)
        return _one_symbol_result(o)


class HydroWideMMTrader:
    """Sweeps + unwind + touch quotes (bb, ba) in neutral instead of bb+1/ba-1."""

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        if HYDROGEL_PACK not in state.order_depths:
            return _one_symbol_result([])
        d = state.order_depths[HYDROGEL_PACK]
        if not d.buy_orders or not d.sell_orders:
            return _one_symbol_result([])
        mid = _mid(d)
        if mid is None:
            return _one_symbol_result([])
        p = HYDRO_PARAMS
        buy_t = int(p["buy_thresh"])
        sell_t = int(p["sell_thresh"])
        uw = int(p["unwind_long"])
        mm = int(p.get("mm_size", 15))
        pos = int(state.position.get(HYDROGEL_PACK, 0))
        bb = max(d.buy_orders.keys())
        ba = min(d.sell_orders.keys())
        m14b, m14s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 14")
        m38b, m38s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 38")
        o: List[Order] = []
        if mid > sell_t and pos > -HYDRO_LIMIT:
            o = _sweep_sell(d, pos, bb - 10)
        elif mid < buy_t and pos < HYDRO_LIMIT:
            o = _sweep_buy(d, pos, ba + 10)
        elif mid > uw and pos > 0:
            o = _unwind_long(d, pos)
        else:
            bp, sp = bb, ba
            if pos > 50:
                bp -= 1
                sp -= 1
            elif pos < -50:
                bp += 1
                sp += 1
            if m14b and not m14s:
                bp += 1
            elif m14s and not m14b:
                sp -= 1
            if m38b and not m38s:
                sp -= 1
            elif m38s and not m38b:
                bp += 1
            bp = min(bp, ba - 1)
            sp = max(sp, bb + 1)
            bq = min(mm, HYDRO_LIMIT - pos)
            sq = min(mm, HYDRO_LIMIT + pos)
            if bq > 0 and bp < sp:
                o.append(Order(HYDROGEL_PACK, int(bp), bq))
            if sq > 0 and bp < sp:
                o.append(Order(HYDROGEL_PACK, int(sp), -sq))
        return _one_symbol_result(o)


class HydroTightMMTrader:
    """Sweeps + unwind + bb+2/ba-2 in neutral when spread allows (tighter than default)."""

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        if HYDROGEL_PACK not in state.order_depths:
            return _one_symbol_result([])
        d = state.order_depths[HYDROGEL_PACK]
        if not d.buy_orders or not d.sell_orders:
            return _one_symbol_result([])
        mid = _mid(d)
        if mid is None:
            return _one_symbol_result([])
        p = HYDRO_PARAMS
        buy_t = int(p["buy_thresh"])
        sell_t = int(p["sell_thresh"])
        uw = int(p["unwind_long"])
        mm = int(p.get("mm_size", 15))
        pos = int(state.position.get(HYDROGEL_PACK, 0))
        bb = max(d.buy_orders.keys())
        ba = min(d.sell_orders.keys())
        m14b, m14s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 14")
        m38b, m38s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 38")
        o: List[Order] = []
        if mid > sell_t and pos > -HYDRO_LIMIT:
            o = _sweep_sell(d, pos, bb - 10)
        elif mid < buy_t and pos < HYDRO_LIMIT:
            o = _sweep_buy(d, pos, ba + 10)
        elif mid > uw and pos > 0:
            o = _unwind_long(d, pos)
        else:
            if ba - bb >= 4:
                bp, sp = bb + 2, ba - 2
            else:
                bp, sp = bb + 1, ba - 1
            if pos > 50:
                bp -= 1
                sp -= 1
            elif pos < -50:
                bp += 1
                sp += 1
            if m14b and not m14s:
                bp += 1
            elif m14s and not m14b:
                sp -= 1
            if m38b and not m38s:
                sp -= 1
            elif m38s and not m38b:
                bp += 1
            bp = min(bp, ba - 1)
            sp = max(sp, bb + 1)
            bq = min(mm, HYDRO_LIMIT - pos)
            sq = min(mm, HYDRO_LIMIT + pos)
            if bq > 0 and bp < sp:
                o.append(Order(HYDROGEL_PACK, int(bp), bq))
            if sq > 0 and bp < sp:
                o.append(Order(HYDROGEL_PACK, int(sp), -sq))
        return _one_symbol_result(o)


class HydroDonchianR3CalibratedTrader:
    """
    Buy/sell sweep thresholds = lo/hi from R3 day0 mid percentiles (passed in);
    other layers same as production (unwind, marks, inner MM).
    """

    def __init__(self, buy_thresh: int, sell_thresh: int) -> None:
        self._buy = int(buy_thresh)
        self._sell = int(sell_thresh)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        if HYDROGEL_PACK not in state.order_depths:
            return _one_symbol_result([])
        d = state.order_depths[HYDROGEL_PACK]
        if not d.buy_orders or not d.sell_orders:
            return _one_symbol_result([])
        mid = _mid(d)
        if mid is None:
            return _one_symbol_result([])
        p = HYDRO_PARAMS
        uw = int(p["unwind_long"])
        mm = int(p.get("mm_size", 15))
        pos = int(state.position.get(HYDROGEL_PACK, 0))
        bb = max(d.buy_orders.keys())
        ba = min(d.sell_orders.keys())
        m14b, m14s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 14")
        m38b, m38s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 38")
        buy_t, sell_t = self._buy, self._sell
        o: List[Order] = []
        if mid > sell_t and pos > -HYDRO_LIMIT:
            o = _sweep_sell(d, pos, bb - 10)
        elif mid < buy_t and pos < HYDRO_LIMIT:
            o = _sweep_buy(d, pos, ba + 10)
        elif mid > uw and pos > 0:
            o = _unwind_long(d, pos)
        else:
            bp, sp = _mm_quotes(
                bb, ba, pos, mm, m14b, m14s, m38b, m38s, "follow", "fade"
            )
            bq = min(mm, HYDRO_LIMIT - pos)
            sq = min(mm, HYDRO_LIMIT + pos)
            if bq > 0:
                o.append(Order(HYDROGEL_PACK, int(bp), bq))
            if sq > 0:
                o.append(Order(HYDROGEL_PACK, int(sp), -sq))
        return _one_symbol_result(o)


def load_full_file_trader() -> Type:
    """Full r4_m14_67_tilt_minseoks option strat (all products) — for parity check."""
    # .../misc/hydro_research/strategies/thisfile -> repo root
    repo = Path(__file__).resolve().parent.parent.parent.parent
    path = repo / "trading_algos" / "ROUND4" / "r4_m14_67_tilt_minseoks_option_strat.py"
    spec = importlib.util.spec_from_file_location("mhs", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Trader


# Registry: id -> (label, class or 0-ary factory returning class)
RESEARCH_TRADERS: List[Tuple[str, str, Callable[[], Type]]] = [
    ("prod_core", "Production_HYDRO_core_M14_fadeM38", lambda: HydroProductionHydroCoreTrader),
    ("prod_file", "Production_full_file_all_symbols", load_full_file_trader),
    ("naive_mm", "Naive_MM_only", lambda: HydroNaiveMMTrader),
    ("thr_mr", "Threshold_MR_sweeps_unwind_no_marks", lambda: HydroThresholdMRTrader),
    ("mm_m14", "MM_sweeps_M14_no_M38", lambda: HydroMMNoMarksTrader),
    ("follow_m38", "Production_with_follow_M38", lambda: HydroFollowM38Trader),
    ("unwind_only", "Unwind_long_only", lambda: HydroUnwindOnlyTrader),
    ("wide_mm", "Production_touch_quotes", lambda: HydroWideMMTrader),
    ("tight_mm", "Production_bb2_ba2", lambda: HydroTightMMTrader),
]

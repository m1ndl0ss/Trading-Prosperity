"""
Round 4: HYDRO + VELVET = r4_main_m14_67_tilt (follow M14, vs M38 on HYDRO;
follow M67, vs M55 on VELVET).  VEV = Friends_higher_pnl_algo v11 options only
(deep ITM 4000/4500, ATM+drift 5000-5200, sell-biased 5300-5500 w/ dynamic T,
MM 6000/6500).  Sell-biased v11 K/S smile (same as profitable 497154 log).
No Friends HYDRO/VELVET thresholds or M67+8 sell premium.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ── HYDRO / VELVET (r4_main_m14_67_tilt) ─────────────────────────────────────
HYDROGEL_PACK = "HYDROGEL_PACK"
HYDRO_PARAMS: Dict[str, Any] = {
    "buy_thresh":  9935,
    "sell_thresh": 10030,
    "unwind_long": 9995,
    "mm_size":     15,
}
HYDRO_LIMIT = 200

VELVET = "VELVETFRUIT_EXTRACT"
VELVET_PARAMS: Dict[str, Any] = {
    "buy_thresh":  5215,
    "sell_thresh": 5275,
    "unwind_long": 5272,
    "mm_size":     25,
}
VELVET_LIMIT = 200

# ── Friends: BS / smiles / T / VEV structure (unchanged from Friends file) ─
_SQRT2  = 1.4142135623730951
_T      = 0.016
_S_REF  = 5248.0
_T_INIT     = 2.0 / 252.0
_T_PER_DAY  = 1.0 / 252.0
_MAX_TIMESTAMP = 99900.0

_IV_A =  7.867
_IV_B = -15.462
_IV_C =  7.922
_NEW_IV_A =  14.535
_NEW_IV_B = -29.732
_NEW_IV_C =  15.424

_EMA_ALPHA       = 0.01
_DRIFT_THRESHOLD = 15.0
_VEV_LIMIT = 300
_VEV_MM    = 15

_VEV_SELL_BIASED: Dict[int, Dict] = {
    5300: {"sell_margin": 2.0,  "buy_margin": 12.0},
    5400: {"sell_margin": 3.0,  "buy_margin": 10.0},
    5500: {"sell_margin": 2.5,  "buy_margin": 8.0},
}

_VEV_DEEP_ITM: Dict[int, Dict] = {
    4000: {"buy": 1246.0, "sell": 1276.0},
    4500: {"buy":  749.0, "sell":  773.0},
}
_VEV_ATM: Dict[int, Dict] = {
    5000: {"buy": 240.0, "sell": 264.0, "drift": True},
    5100: {"buy": 151.0, "sell": 173.0, "drift": True},
    5200: {"buy":  89.3, "sell": 109.7, "drift": False},
}
_VEV_FV_REF_ATM: Dict[int, float] = {}

_VEV_MM_ONLY: Tuple[int, ...] = (6000, 6500)

_ALL_SYMBOLS: Tuple[str, ...] = (
    HYDROGEL_PACK,
    VELVET,
    *(f"VEV_{k}" for k in (
        list(_VEV_DEEP_ITM) + list(_VEV_ATM) +
        list(_VEV_SELL_BIASED) + list(_VEV_MM_ONLY)
    )),
)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return max(S - K, 0.0)
    sq = sigma * math.sqrt(T)
    if sq < 1e-9:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sq
    return S * _ncdf(d1) - K * _ncdf(d1 - sq)


def _old_smile_fv(K: float, S: float) -> float:
    m  = K / S
    iv = max(_IV_A * m * m + _IV_B * m + _IV_C, 0.05)
    return _bs_call(S, K, _T, iv)


def _new_smile_fv(K: float, S: float, T: float) -> float:
    m = K / S
    iv = max(_NEW_IV_A * m * m + _NEW_IV_B * m + _NEW_IV_C, 0.05)
    return _bs_call(S, K, T, iv)


# deferred init for _VEV_FV_REF_ATM (must run after _old_smile_fv is defined)
def _init_fv_ref() -> None:
    global _VEV_FV_REF_ATM
    if not _VEV_FV_REF_ATM:
        _VEV_FV_REF_ATM = {
            K: _old_smile_fv(float(K), _S_REF) for K in _VEV_ATM
        }


_init_fv_ref()


def _mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


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


def _hydro_sweep_buy(depth: OrderDepth, pos: int, max_price: int) -> List[Order]:
    want = HYDRO_LIMIT - pos
    if want <= 0 or not depth.sell_orders:
        return []
    if min(depth.sell_orders.keys()) > max_price:
        return []
    return [Order(HYDROGEL_PACK, max_price, want)]


def _hydro_sweep_sell(depth: OrderDepth, pos: int, min_price: int) -> List[Order]:
    want = HYDRO_LIMIT + pos
    if want <= 0 or not depth.buy_orders:
        return []
    if max(depth.buy_orders.keys()) < min_price:
        return []
    return [Order(HYDROGEL_PACK, min_price, -want)]


def _hydro_unwind_long(depth: OrderDepth, pos: int) -> List[Order]:
    if pos <= 0 or not depth.buy_orders:
        return []
    return [Order(HYDROGEL_PACK, max(depth.buy_orders.keys()), -pos)]


def _velvet_sweep_buy(depth: OrderDepth, pos: int, max_price: int) -> List[Order]:
    want = VELVET_LIMIT - pos
    if want <= 0 or not depth.sell_orders:
        return []
    if min(depth.sell_orders.keys()) > max_price:
        return []
    return [Order(VELVET, max_price, want)]


def _velvet_sweep_sell(depth: OrderDepth, pos: int, min_price: int) -> List[Order]:
    want = VELVET_LIMIT + pos
    if want <= 0 or not depth.buy_orders:
        return []
    if max(depth.buy_orders.keys()) < min_price:
        return []
    return [Order(VELVET, min_price, -want)]


_VEV_EVERY: Tuple[int, ...] = (
    *tuple(_VEV_DEEP_ITM),
    *tuple(_VEV_ATM),
    *tuple(_VEV_SELL_BIASED),
    *_VEV_MM_ONLY,
)


class Trader:
    LIMIT: Dict[str, int] = {
        HYDROGEL_PACK: HYDRO_LIMIT,
        VELVET: VELVET_LIMIT,
        **{f"VEV_{k}": _VEV_LIMIT for k in _VEV_EVERY},
    }

    def __init__(self) -> None:
        self.hydro_p  = dict(HYDRO_PARAMS)
        self.velvet_p = dict(VELVET_PARAMS)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                td = {}

        result: Dict[str, List[Order]] = {sym: [] for sym in _ALL_SYMBOLS}

        m14b, m14s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 14")
        m38b, m38s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 38")
        m67b, m67s = _last_tick_mark_sides(state, VELVET, "Mark 67")
        m55b, m55s = _last_tick_mark_sides(state, VELVET, "Mark 55")

        S_now: Optional[float] = None

        # 1) HYDROGEL
        if HYDROGEL_PACK in state.order_depths:
            d = state.order_depths[HYDROGEL_PACK]
            if d.buy_orders and d.sell_orders:
                mid = _mid(d)
                if mid is not None:
                    p = self.hydro_p
                    buy_t = int(p["buy_thresh"])
                    sell_t = int(p["sell_thresh"])
                    uw = int(p["unwind_long"])
                    mm = int(p.get("mm_size", 15))
                    pos = int(state.position.get(HYDROGEL_PACK, 0))
                    bb = max(d.buy_orders.keys())
                    ba = min(d.sell_orders.keys())
                    o: List[Order] = []

                    if mid > sell_t and pos > -HYDRO_LIMIT:
                        o = _hydro_sweep_sell(d, pos, bb - 10)
                    elif mid < buy_t and pos < HYDRO_LIMIT:
                        o = _hydro_sweep_buy(d, pos, ba + 10)
                    elif mid > uw and pos > 0:
                        o = _hydro_unwind_long(d, pos)
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
                        if bq > 0:
                            o.append(Order(HYDROGEL_PACK, int(bp), bq))
                        if sq > 0:
                            o.append(Order(HYDROGEL_PACK, int(sp), -sq))
                    result[HYDROGEL_PACK] = o

        # 2) VELVET
        if VELVET in state.order_depths:
            d = state.order_depths[VELVET]
            if d.buy_orders and d.sell_orders:
                mid = _mid(d)
                if mid is not None:
                    S_now = mid
                    p = self.velvet_p
                    buy_t = int(p["buy_thresh"])
                    sell_t = int(p["sell_thresh"])
                    uw_l = int(p.get("unwind_long", 99999))
                    mm = int(p.get("mm_size", 25))
                    pos = int(state.position.get(VELVET, 0))
                    bb = max(d.buy_orders.keys())
                    ba = min(d.sell_orders.keys())
                    o: List[Order] = []

                    if mid < buy_t:
                        o = _velvet_sweep_buy(d, pos, ba + 4)
                    elif mid > sell_t:
                        o = _velvet_sweep_sell(d, pos, bb - 4)
                    elif mid > uw_l and pos > 0:
                        o = [Order(VELVET, bb, -pos)]
                    elif abs(pos) < mm * 4:
                        bp, sp = bb + 1, ba - 1
                        if pos > 50:
                            bp -= 1
                            sp -= 1
                        elif pos < -50:
                            bp += 1
                            sp += 1
                        if m67b and not m67s:
                            bp += 1
                        elif m67s and not m67b:
                            sp -= 1
                        if m55b and not m55s:
                            sp -= 1
                        elif m55s and not m55b:
                            bp += 1
                        bp = min(bp, ba - 1)
                        sp = max(sp, bb + 1)
                        if bp < sp:
                            bq = min(mm, VELVET_LIMIT - pos)
                            sq = min(mm, VELVET_LIMIT + pos)
                            if bq > 0:
                                o.append(Order(VELVET, int(bp), bq))
                            if sq > 0:
                                o.append(Order(VELVET, int(sp), -sq))
                    result[VELVET] = o

        # EMA + dynamic T (Friends) — for VEV
        ema_S: float = float(td.get("ema_S", _S_REF))
        if S_now is not None:
            ema_S = _EMA_ALPHA * S_now + (1.0 - _EMA_ALPHA) * ema_S
        td["ema_S"] = ema_S

        day_count: int = int(td.get("day_count", 0))
        last_ts: int = int(td.get("last_ts", state.timestamp))
        if state.timestamp < last_ts:
            day_count += 1
        td["day_count"] = day_count
        td["last_ts"] = state.timestamp
        day_frac = float(state.timestamp) / _MAX_TIMESTAMP
        t_dyn = max(
            _T_INIT - day_count * _T_PER_DAY - day_frac * _T_PER_DAY,
            1e-5,
        )

        # 3a) Deep ITM
        for K, cfg in _VEV_DEEP_ITM.items():
            sym = f"VEV_{K}"
            d = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders:
                continue
            midv = _mid(d)
            if midv is None:
                continue
            pos = int(state.position.get(sym, 0))
            bb = max(d.buy_orders.keys())
            ba = min(d.sell_orders.keys())
            o: List[Order] = []
            buy_t, sell_t = cfg["buy"], cfg["sell"]
            if midv < buy_t:
                want = _VEV_LIMIT - pos
                if want > 0:
                    o.append(Order(sym, ba + 4, want))
            elif midv > sell_t:
                want = _VEV_LIMIT + pos
                if want > 0:
                    o.append(Order(sym, bb - 4, -want))
            else:
                bp, sp = bb + 1, ba - 1
                if pos > _VEV_LIMIT // 4:
                    bp -= 1
                    sp -= 1
                elif pos < -(_VEV_LIMIT // 4):
                    bp += 1
                    sp += 1
                bp = min(bp, ba - 1)
                sp = max(sp, bb + 1)
                if bp < sp:
                    bq = min(_VEV_MM, _VEV_LIMIT - pos)
                    sq = min(_VEV_MM, _VEV_LIMIT + pos)
                    if bq > 0:
                        o.append(Order(sym, int(bp), bq))
                    if sq > 0:
                        o.append(Order(sym, int(sp), -sq))
            result[sym] = o

        # 3b) ATM/OTM core
        drift = ema_S - _S_REF
        apply_bs = abs(drift) >= _DRIFT_THRESHOLD

        for K, cfg in _VEV_ATM.items():
            sym = f"VEV_{K}"
            d = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders:
                continue
            midv = _mid(d)
            if midv is None:
                continue
            pos = int(state.position.get(sym, 0))
            bb = max(d.buy_orders.keys())
            ba = min(d.sell_orders.keys())
            o = []
            use_drift = apply_bs and cfg.get("drift", True)
            fv_d = (
                (_old_smile_fv(float(K), ema_S) - _VEV_FV_REF_ATM[K]) if use_drift else 0.0
            )
            buy_t = cfg["buy"] + fv_d
            sell_t = cfg["sell"] + fv_d
            if midv < buy_t:
                want = _VEV_LIMIT - pos
                if want > 0:
                    o.append(Order(sym, ba + 4, want))
            elif midv > sell_t:
                want = _VEV_LIMIT + pos
                if want > 0:
                    o.append(Order(sym, bb - 4, -want))
            else:
                bp, sp = bb + 1, ba - 1
                if pos > _VEV_LIMIT // 4:
                    bp -= 1
                    sp -= 1
                elif pos < -(_VEV_LIMIT // 4):
                    bp += 1
                    sp += 1
                bp = min(bp, ba - 1)
                sp = max(sp, bb + 1)
                if bp < sp:
                    bq = min(_VEV_MM, _VEV_LIMIT - pos)
                    sq = min(_VEV_MM, _VEV_LIMIT + pos)
                    if bq > 0:
                        o.append(Order(sym, int(bp), bq))
                    if sq > 0:
                        o.append(Order(sym, int(sp), -sq))
            result[sym] = o

        # 3c) Sell-biased
        for K_sb, cfg_sb in _VEV_SELL_BIASED.items():
            sym_sb = f"VEV_{K_sb}"
            d_sb = state.order_depths.get(sym_sb)
            if not d_sb or not d_sb.buy_orders or not d_sb.sell_orders:
                continue
            mid_sb = _mid(d_sb)
            if mid_sb is None:
                continue
            pos_sb = int(state.position.get(sym_sb, 0))
            bb_sb = max(d_sb.buy_orders.keys())
            ba_sb = min(d_sb.sell_orders.keys())
            fv_sb = _new_smile_fv(float(K_sb), ema_S, t_dyn)
            sell_t = fv_sb + cfg_sb["sell_margin"]
            buy_t = fv_sb - cfg_sb["buy_margin"]
            o_sb: List[Order] = []
            if mid_sb > sell_t and pos_sb > -_VEV_LIMIT:
                want = _VEV_LIMIT + pos_sb
                if want > 0:
                    o_sb.append(Order(sym_sb, bb_sb - 2, -want))
            elif mid_sb < buy_t and pos_sb < _VEV_LIMIT:
                want = _VEV_LIMIT - pos_sb
                if want > 0:
                    o_sb.append(Order(sym_sb, ba_sb + 2, want))
            else:
                bp_sb = bb_sb + 1
                sp_sb = ba_sb - 1
                if pos_sb > _VEV_LIMIT // 4:
                    bp_sb -= 1
                    sp_sb -= 1
                bp_sb = min(bp_sb, ba_sb - 1)
                sp_sb = max(sp_sb, bb_sb + 1)
                if bp_sb < sp_sb:
                    bq_sb = min(_VEV_MM, _VEV_LIMIT - pos_sb)
                    sq_sb = min(_VEV_MM, _VEV_LIMIT + pos_sb)
                    if bq_sb > 0:
                        o_sb.append(Order(sym_sb, int(bp_sb), bq_sb))
                    if sq_sb > 0:
                        o_sb.append(Order(sym_sb, int(sp_sb), -sq_sb))
            result[sym_sb] = o_sb

        # 4) MM 6000 / 6500
        for K in _VEV_MM_ONLY:
            sym = f"VEV_{K}"
            d = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders:
                continue
            pos = int(state.position.get(sym, 0))
            bb = max(d.buy_orders.keys())
            ba = min(d.sell_orders.keys())
            o: List[Order] = []
            if (ba - bb) >= 3:
                bp, sp = bb + 1, ba - 1
                if pos > _VEV_LIMIT // 4:
                    bp -= 1
                    sp -= 1
                elif pos < -(_VEV_LIMIT // 4):
                    bp += 1
                    sp += 1
                bp = min(bp, ba - 1)
                sp = max(sp, bb + 1)
                if bp < sp:
                    bq = min(_VEV_MM, _VEV_LIMIT - pos)
                    sq = min(_VEV_MM, _VEV_LIMIT + pos)
                    if bq > 0:
                        o.append(Order(sym, int(bp), bq))
                    if sq > 0:
                        o.append(Order(sym, int(sp), -sq))
            result[sym] = o

        return result, 0, json.dumps(td, separators=(",", ":"))

from math import log, sqrt
from statistics import NormalDist
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
import json

_N = NormalDist()

# ── Volatility smile parabola (K=4000/4500 excluded from fit) ──
A, B, C = 7.867122, -15.462142, 7.922102

# ── Per-strike config ──
# adj_fair = bs_fair + mean_bias  (the "expected" market execution price)
# Only K=5100 and K=5500 are kept:
#   K=5000 → REMOVED: avg spread 6.2 credits, execution edge always negative
#   K=5100 → day 2 sells have +0.14 avg edge at execution (36% positive ticks)
#   K=5500 → day 0 buys +2.15 (88%), day 2 sells +2.03 (99%) — the key product
STRIKE_PARAMS = {
    5100: {'mean_bias':  0.08, 'pos_limit': 60},
    5500: {'mean_bias': -1.67, 'pos_limit': 150},
}

# ── Minimum edge at execution price vs adj_fair to actually trade ──
# Advisor: "volume should reflect strength of conviction" → scale with edge
MIN_EDGE = 0.5        # credits — below this, skip entirely
MIN_VOUCHER_MID = 3.0 # skip near-worthless OTM options

# ── VELVETFRUIT_EXTRACT mean reversion ──
VEV_LIMIT    = 200
VEV_EMA_SPAN = 50
VEV_EDGE     = 4.0
VEV_QTY      = 8


# ── Math ─────────────────────────────────────────────────────────────────────

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2)


def smile_sigma(S: float, K: int) -> float:
    m = K / S
    return max(A * m ** 2 + B * m + C, 0.01)


def fair_price(S: float, K: int, T_years: float) -> float:
    return bs_call(S, K, T_years, smile_sigma(S, K))


def calibrate_tte(S: float, order_depths: dict) -> Optional[float]:
    """Back-solve TTE from near-ATM option market price via bisection."""
    for K in [5100, 5000, 5200, 5300]:
        depth = order_depths.get(f'VEV_{K}')
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            continue
        mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0
        if mid < 10:
            continue
        sigma = smile_sigma(S, K)
        lo, hi = 0.3 / 365.0, 10.0 / 365.0
        for _ in range(30):
            t_mid = (lo + hi) / 2.0
            if bs_call(S, K, t_mid, sigma) > mid:
                hi = t_mid
            else:
                lo = t_mid
        tte_days = t_mid * 365.0
        if 0.5 <= tte_days <= 8.5:
            return round(tte_days * 2) / 2.0
    return None


def qty_from_edge(edge: float) -> int:
    """
    Advisor: scale volume proportionally to conviction (deviation magnitude).
    Larger edge = stronger signal = more units.
    """
    if edge < MIN_EDGE:
        return 0
    elif edge < 1.5:
        return 3
    elif edge < 3.0:
        return 7
    elif edge < 5.0:
        return 12
    else:
        return 18


def get_mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        # ── 1. Restore persisted state ───────────────────────────────────────
        ema_vev   = None
        tte_start = None
        prev_ts   = -1
        day_count = 0

        if state.traderData:
            try:
                d = json.loads(state.traderData)
                ema_vev   = d.get('ema_vev')
                tte_start = d.get('tte_start')
                prev_ts   = d.get('prev_ts', -1)
                day_count = d.get('day_count', 0)
            except Exception:
                pass

        # ── 2. Detect day rollover ───────────────────────────────────────────
        if prev_ts > 0 and state.timestamp < prev_ts:
            day_count += 1
            if tte_start is not None:
                tte_start -= 1.0
        prev_ts = state.timestamp

        # ── 3. Get underlying price ──────────────────────────────────────────
        vev_depth = state.order_depths.get('VELVETFRUIT_EXTRACT')
        S = get_mid(vev_depth) if vev_depth else None
        if S is None:
            return {}, 0, json.dumps({'ema_vev': ema_vev, 'tte_start': tte_start,
                                      'prev_ts': prev_ts, 'day_count': day_count})

        # ── 4. Calibrate TTE on first tick ───────────────────────────────────
        if tte_start is None:
            tte_start = calibrate_tte(S, state.order_depths)
            if tte_start is None:
                tte_start = 5.0

        T_years = max((tte_start - state.timestamp / 1_000_000.0) / 365.0, 1e-6)

        # ── 5. Update EMA ────────────────────────────────────────────────────
        alpha = 2.0 / (VEV_EMA_SPAN + 1)
        ema_vev = S if ema_vev is None else alpha * S + (1 - alpha) * ema_vev

        result: Dict[str, List[Order]] = {}

        # ── 6. IV Scalping — compare EXECUTION PRICE vs adj_fair ─────────────
        # Advisor: buy underpriced vol, sell overpriced vol.
        # Critical fix: we execute at bid (sells) or ask (buys), NOT at mid.
        # Only trade when the actual execution price beats the adjusted fair value.
        for K, params in STRIKE_PARAMS.items():
            symbol = f'VEV_{K}'
            depth  = state.order_depths.get(symbol)
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue

            mid = get_mid(depth)
            if mid is None or mid < MIN_VOUCHER_MID:
                continue

            fp       = fair_price(S, K, T_years)
            adj_fair = fp + params['mean_bias']
            pos      = state.position.get(symbol, 0)
            limit    = params['pos_limit']
            bid      = max(depth.buy_orders)
            ask      = min(depth.sell_orders)
            orders   = []

            # SELL: bid must be above adj_fair (we execute above fair value)
            sell_edge = bid - adj_fair
            if sell_edge >= MIN_EDGE and pos > -limit:
                qty = min(qty_from_edge(sell_edge), limit + pos)
                if qty > 0:
                    orders.append(Order(symbol, bid, -qty))

            # BUY: ask must be below adj_fair (we execute below fair value)
            buy_edge = adj_fair - ask
            if buy_edge >= MIN_EDGE and pos < limit:
                qty = min(qty_from_edge(buy_edge), limit - pos)
                if qty > 0:
                    orders.append(Order(symbol, ask, qty))

            if orders:
                result[symbol] = orders

        # ── 7. VELVETFRUIT_EXTRACT mean reversion ────────────────────────────
        ema_dev    = S - ema_vev
        vev_pos    = state.position.get('VELVETFRUIT_EXTRACT', 0)
        vev_orders = []

        if ema_dev > VEV_EDGE and vev_pos > -VEV_LIMIT:
            bid = max(vev_depth.buy_orders) if vev_depth.buy_orders else None
            if bid:
                qty = min(VEV_QTY, VEV_LIMIT + vev_pos)
                if qty > 0:
                    vev_orders.append(Order('VELVETFRUIT_EXTRACT', bid, -qty))

        elif ema_dev < -VEV_EDGE and vev_pos < VEV_LIMIT:
            ask = min(vev_depth.sell_orders) if vev_depth.sell_orders else None
            if ask:
                qty = min(VEV_QTY, VEV_LIMIT - vev_pos)
                if qty > 0:
                    vev_orders.append(Order('VELVETFRUIT_EXTRACT', ask, qty))

        if vev_orders:
            result['VELVETFRUIT_EXTRACT'] = vev_orders

        # ── 8. Persist state ─────────────────────────────────────────────────
        return result, 0, json.dumps({
            'ema_vev': ema_vev, 'tte_start': tte_start,
            'prev_ts': prev_ts, 'day_count': day_count,
        })

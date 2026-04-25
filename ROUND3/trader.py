from math import log, sqrt
from statistics import NormalDist
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
import json

_N = NormalDist()

# ── Volatility smile parabola (corrected: K=4000/4500 excluded from fit) ──
A, B, C = 7.867122, -15.462142, 7.922102

# ── Active strikes with per-strike mean bias and edge threshold ──
# Only trade strikes with tight std (reliable signal)
# Mean bias = typical (market - fair); edge = min deviation to trade
STRIKE_PARAMS = {
    5000: {'mean_bias': -0.37, 'edge': 1.8},
    5100: {'mean_bias':  0.08, 'edge': 2.2},
    5500: {'mean_bias': -1.67, 'edge': 2.8},
}

# ── Safety filters ──
MIN_VOUCHER_MID  = 5.0   # skip near-worthless deep OTM options
MAX_TRADE_QTY    = 10    # units per order (scale up once confident)
VEV_LIMIT        = 200
VEV_EMA_SPAN     = 50    # slow EMA → filters tick noise
VEV_EDGE         = 4.0   # ~3 std from EMA before trading
VEV_QTY          = 8


# ── Math helpers ──────────────────────────────────────────────────────────────

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2)


def smile_sigma(S: float, K: int) -> float:
    m = K / S
    return max(A * m ** 2 + B * m + C, 0.01)


def fair_price(S: float, K: int, T: float) -> float:
    return bs_call(S, K, T, smile_sigma(S, K))


def calibrate_tte(S: float, order_depths: dict) -> Optional[float]:
    """
    Back-solve TTE (in days) from near-ATM option market price.
    Uses bisection: find T such that bs_call(S, K, T, sigma) == market_mid.
    Falls back through strikes until one works.
    """
    for K in [5100, 5000, 5200, 5300]:
        depth = order_depths.get(f'VEV_{K}')
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            continue
        mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0
        if mid < 10:   # need enough time value to be a useful anchor
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
        # Sanity check: should be between 1 and 8 days
        if 0.5 <= tte_days <= 8.5:
            return round(tte_days * 2) / 2.0   # round to nearest 0.5 day
    return None


def get_mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        # ── 1. Load persisted state ──────────────────────────────────────────
        ema_vev       = None
        tte_start     = None   # TTE in days at timestamp=0 of current day
        prev_ts       = -1
        day_count     = 0

        if state.traderData:
            try:
                d = json.loads(state.traderData)
                ema_vev   = d.get('ema_vev')
                tte_start = d.get('tte_start')
                prev_ts   = d.get('prev_ts', -1)
                day_count = d.get('day_count', 0)
            except Exception:
                pass

        # ── 2. Detect day rollover (timestamp resets to 0 between days) ─────
        if prev_ts > 0 and state.timestamp < prev_ts:
            day_count  += 1
            tte_start  -= 1.0   # one full day consumed
        prev_ts = state.timestamp

        # ── 3. Get underlying price ──────────────────────────────────────────
        vev_depth = state.order_depths.get('VELVETFRUIT_EXTRACT')
        S = get_mid(vev_depth) if vev_depth else None
        if S is None:
            return {}, 0, json.dumps({'ema_vev': ema_vev, 'tte_start': tte_start,
                                       'prev_ts': prev_ts, 'day_count': day_count})

        # ── 4. Calibrate TTE on first tick (robust to backtester starting mid-run) ──
        if tte_start is None:
            tte_start = calibrate_tte(S, state.order_depths)
            if tte_start is None:
                tte_start = 5.0   # safe default (overestimates, but better than crashing)

        # Current time to expiry in years
        T_years = max((tte_start - state.timestamp / 1_000_000.0) / 365.0, 1e-6)

        # ── 5. Update EMA on underlying ─────────────────────────────────────
        alpha = 2.0 / (VEV_EMA_SPAN + 1)
        ema_vev = S if ema_vev is None else alpha * S + (1 - alpha) * ema_vev

        result: Dict[str, List[Order]] = {}

        # ── 6. IV Scalping across active strikes ─────────────────────────────
        for K, params in STRIKE_PARAMS.items():
            symbol = f'VEV_{K}'
            depth  = state.order_depths.get(symbol)
            if depth is None:
                continue

            mid = get_mid(depth)
            if mid is None or mid < MIN_VOUCHER_MID:
                continue   # skip near-worthless / illiquid options

            fp       = fair_price(S, K, T_years)
            centered = (mid - fp) - params['mean_bias']
            edge     = params['edge']
            pos      = state.position.get(symbol, 0)
            orders   = []

            if centered > edge and pos > -300:
                # Overpriced → sell at best bid
                bid = max(depth.buy_orders) if depth.buy_orders else None
                if bid is not None:
                    qty = min(MAX_TRADE_QTY, 300 + pos)
                    if qty > 0:
                        orders.append(Order(symbol, bid, -qty))

            elif centered < -edge and pos < 300:
                # Underpriced → buy at best ask
                ask = min(depth.sell_orders) if depth.sell_orders else None
                if ask is not None:
                    qty = min(MAX_TRADE_QTY, 300 - pos)
                    if qty > 0:
                        orders.append(Order(symbol, ask, qty))

            if orders:
                result[symbol] = orders

        # ── 7. Mean reversion on VELVETFRUIT_EXTRACT ────────────────────────
        ema_dev  = S - ema_vev
        vev_pos  = state.position.get('VELVETFRUIT_EXTRACT', 0)
        vev_orders = []

        if ema_dev > VEV_EDGE and vev_pos > -VEV_LIMIT:
            bid = max(vev_depth.buy_orders) if vev_depth.buy_orders else None
            if bid is not None:
                qty = min(VEV_QTY, VEV_LIMIT + vev_pos)
                if qty > 0:
                    vev_orders.append(Order('VELVETFRUIT_EXTRACT', bid, -qty))

        elif ema_dev < -VEV_EDGE and vev_pos < VEV_LIMIT:
            ask = min(vev_depth.sell_orders) if vev_depth.sell_orders else None
            if ask is not None:
                qty = min(VEV_QTY, VEV_LIMIT - vev_pos)
                if qty > 0:
                    vev_orders.append(Order('VELVETFRUIT_EXTRACT', ask, qty))

        if vev_orders:
            result['VELVETFRUIT_EXTRACT'] = vev_orders

        # ── 8. Persist state ─────────────────────────────────────────────────
        trader_data = json.dumps({
            'ema_vev':   ema_vev,
            'tte_start': tte_start,
            'prev_ts':   prev_ts,
            'day_count': day_count,
        })

        return result, 0, trader_data

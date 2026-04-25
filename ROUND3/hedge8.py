from math import log, sqrt
from statistics import NormalDist
from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

_N = NormalDist()

# ── Calibrated from 6aPlot (corrected, excluding K=4000/4500) ──
A, B, C = 7.867122, -15.462142, 7.922102

# ── Per-strike thresholds: (mean_bias, entry_threshold) ──
# Entry when |deviation - mean_bias| > entry_threshold
STRIKE_PARAMS = {
    5000: {'mean': -0.37, 'edge': 1.5},
    5100: {'mean':  0.08, 'edge': 2.0},
    5500: {'mean': -1.67, 'edge': 2.5},
    6000: {'mean': -0.11, 'edge': 0.6},
    6500: {'mean': -0.49, 'edge': 1.0},
}

# ── Round 3: TTE starts at 5 days, 1 tick = 1/1,000,000 day ──
TTE_START_DAYS = 5.0

def bs_call_price(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2)

def fair_price(S, K, T_years):
    m = K / S
    v_hat = A * m**2 + B * m + C
    v_hat = max(v_hat, 0.01)          # floor at 1% vol
    return bs_call_price(S, K, T_years, v_hat)


class Trader:
    def __init__(self):
        self.ema = None                # for VELVETFRUIT_EXTRACT mean reversion
        self.EMA_SPAN = 10            # fast EMA, same as Frankfurt

    def run(self, state: TradingState):
        result = {}
        
        # ── 1. Get VELVETFRUIT_EXTRACT mid price ──
        vev_depth = state.order_depths.get('VELVETFRUIT_EXTRACT', None)
        if vev_depth is None:
            return result, 0, ""
        
        S = self._mid(vev_depth)
        if S is None:
            return result, 0, ""
        
        # ── 2. Update EMA ──
        alpha = 2 / (self.EMA_SPAN + 1)
        self.ema = S if self.ema is None else alpha * S + (1 - alpha) * self.ema
        
        # ── 3. Time to expiry ──
        T_years = (TTE_START_DAYS - state.timestamp / 1_000_000) / 365.0
        T_years = max(T_years, 1e-6)
        
        # ── 4. IV Scalp each active strike ──
        for K, params in STRIKE_PARAMS.items():
            symbol = f'VEV_{K}'
            depth = state.order_depths.get(symbol)
            if depth is None:
                continue
            
            market_mid = self._mid(depth)
            if market_mid is None:
                continue
            
            fp = fair_price(S, K, T_years)
            deviation = market_mid - fp          # market minus model
            centered = deviation - params['mean'] # remove structural bias
            edge = params['edge']
            pos = state.position.get(symbol, 0)
            limit = 300
            
            orders = []
            if centered > edge and pos > -limit:   # overpriced → sell
                qty = min(10, limit + pos)
                best_bid = max(depth.buy_orders) if depth.buy_orders else None
                if best_bid:
                    orders.append(Order(symbol, best_bid, -qty))
            
            elif centered < -edge and pos < limit: # underpriced → buy
                qty = min(10, limit - pos)
                best_ask = min(depth.sell_orders) if depth.sell_orders else None
                if best_ask:
                    orders.append(Order(symbol, best_ask, qty))
            
            if orders:
                result[symbol] = orders
        
        # ── 5. Mean reversion on VELVETFRUIT_EXTRACT ──
        ema_deviation = S - self.ema
        vev_pos = state.position.get('VELVETFRUIT_EXTRACT', 0)
        vev_limit = 200
        vev_orders = []
        EMA_EDGE = 2.0   # credits — ~1.5 std from EMA

        if ema_deviation > EMA_EDGE and vev_pos > -vev_limit:
            qty = min(15, vev_limit + vev_pos)
            best_bid = max(vev_depth.buy_orders) if vev_depth.buy_orders else None
            if best_bid:
                vev_orders.append(Order('VELVETFRUIT_EXTRACT', best_bid, -qty))

        elif ema_deviation < -EMA_EDGE and vev_pos < vev_limit:
            qty = min(15, vev_limit - vev_pos)
            best_ask = min(vev_depth.sell_orders) if vev_depth.sell_orders else None
            if best_ask:
                vev_orders.append(Order('VELVETFRUIT_EXTRACT', best_ask, qty))

        if vev_orders:
            result['VELVETFRUIT_EXTRACT'] = vev_orders
        
        # Persist EMA across ticks via trader_data
        trader_data = json.dumps({'ema': self.ema})
        return result, 0, trader_data

    def _mid(self, depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2
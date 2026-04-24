import math
import statistics
from statistics import NormalDist
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

_N = NormalDist()

# --- CONSTANTS ---
HYDROGEL_PACK = "HYDROGEL_PACK"
VELVETFRUIT_EXTRACT = "VELVETFRUIT_EXTRACT"
VOUCHER_STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
OPTION_SYMBOLS = tuple(f"VEV_{k}" for k in VOUCHER_STRIKES)

POS_LIMITS = {HYDROGEL_PACK: 200, VELVETFRUIT_EXTRACT: 200, **{os: 300 for os in OPTION_SYMBOLS}}

# --- UPDATED HYPERPARAMETERS ---
DAY = 5.0  # Round 3 starts at 5
EDGE_TAKE = 3.0   # Require a 3-credit edge to cross the spread (be more picky!)
EDGE_MAKE = 0.5   # Target 0.5 credits for limit orders
DEFAULT_VOL = 0.02 # 2% Annualized (Velvetfruit is very stable)

def bs_call(S: float, K: float, T: float, sigma: float) -> Tuple[float, float]:
    # Standard BS formula with T in years
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0)
    d1 = (math.log(S/K) + (0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2), _N.cdf(d1)

def find_iv(C: float, S: float, K: float, T: float) -> float:
    low, high = 0.0001, 1.0
    for _ in range(12):
        mid = (low + high) / 2
        price, _ = bs_call(S, K, T, mid)
        if price > C: high = mid
        else: low = mid
    return (low + high) / 2

class ProductTrader:
    def __init__(self, name: str, state: TradingState):
        self.name = name
        self.orders = []
        self.pos = state.position.get(name, 0)
        self.limit = POS_LIMITS[name]
        depth = state.order_depths.get(name)
        self.buys = sorted(depth.buy_orders.items(), key=lambda x: x[0], reverse=True) if depth else []
        self.sells = sorted(depth.sell_orders.items(), key=lambda x: x[0]) if depth else []
        self.best_bid = self.buys[0][0] if self.buys else None
        self.best_ask = self.sells[0][0] if self.sells else None
        self.mid = (self.best_bid + self.best_ask) / 2.0 if (self.best_bid and self.best_ask) else None

    def buy(self, price, vol):
        room = self.limit - self.pos
        if room > 0:
            q = min(abs(vol), room)
            self.orders.append(Order(self.name, int(price), q))
            self.pos += q

    def sell(self, price, vol):
        room = self.limit + self.pos
        if room > 0:
            q = min(abs(vol), room)
            self.orders.append(Order(self.name, int(price), -q))
            self.pos -= q

class Trader:
    def run(self, state: TradingState):
        result = {}
        extract = ProductTrader(VELVETFRUIT_EXTRACT, state)
        vouchers = [ProductTrader(s, state) for s in OPTION_SYMBOLS]
        
        # TTE in years (assuming 365 days per year)
        tte_days = max(1e-5, DAY - (state.timestamp / 1_000_000.0))
        tte_years = tte_days / 365.0
        
        # Calculate Fair IV from liquid strikes
        market_ivs = []
        if extract.mid:
            for v in vouchers:
                strike = int(v.name.split('_')[-1])
                if v.mid and 4500 <= strike <= 5500 and v.mid > 2.0:
                    market_ivs.append(find_iv(v.mid, extract.mid, strike, tte_years))
        
        fair_iv = statistics.median(market_ivs) if market_ivs else DEFAULT_VOL

        total_delta = 0.0
        for v in vouchers:
            if not v.mid or not extract.mid: continue
            strike = int(v.name.split('_')[-1])
            fair_px, delta = bs_call(extract.mid, strike, tte_years, fair_iv)
            total_delta += v.pos * delta
            
            # Arb Logic
            if v.best_ask < fair_px - EDGE_TAKE:
                v.buy(v.best_ask, 300)
            elif v.best_bid > fair_px + EDGE_TAKE:
                v.sell(v.best_bid, 300)
            
            # MM Logic
            v.buy(fair_px - EDGE_MAKE, 300)
            v.sell(fair_px + EDGE_MAKE, 300)
            result[v.name] = v.orders

        # Delta Hedging
        if extract.mid:
            target_pos = max(-200, min(200, -int(total_delta)))
            diff = target_pos - extract.pos
            if abs(diff) >= 5: # Hedge threshold
                if diff > 0: extract.buy(extract.best_ask, diff)
                else: extract.sell(extract.best_bid, abs(diff))
            result[extract.name] = extract.orders

        # Hydrogel MM
        hydro = ProductTrader(HYDROGEL_PACK, state)
        if hydro.mid:
            hydro.buy(hydro.mid - 1, 200)
            hydro.sell(hydro.mid + 1, 200)
            result[hydro.name] = hydro.orders

        return result, 0, ""
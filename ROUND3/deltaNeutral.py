"""
Elaborated VELVETFRUIT_EXTRACT & Options Strategy
Integrates fixed-level mean-reversion with Delta-Neutral Options Market Making.
"""

from __future__ import annotations
import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

PRODUCT = "VELVETFRUIT_EXTRACT"

# Define the options and their strike prices based on the environment
OPTIONS = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300
}

PARAMS: Dict[str, Any] = {
    "buy_thresh":  5244,  
    "sell_thresh": 5274,  
    "mm_size": 25,
    "opt_edge": 2.0,      # Edge to capture on options (bid FV - edge, ask FV + edge)
    "volatility": 0.15,   # Implied volatility (TUNE THIS based on market observations)
    "days_to_expiry": 250 # Time to expiry in days (TUNE THIS based on round rules)
}

LIMIT_UNDERLYING = 200
LIMIT_OPTIONS = 100 # Assuming a limit of 100 for options

# --- Black-Scholes Math Utilities ---

def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_call(S: float, K: float, T: float, r: float, vol: float) -> Tuple[float, float]:
    """
    Calculates the Black-Scholes price and Delta for a European Call Option.
    Returns: (Price, Delta)
    """
    if T <= 0:
        return max(0.0, S - K), 1.0 if S > K else 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    
    price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    delta = norm_cdf(d1)
    
    return price, delta

# --- Order Utilities ---

def _mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0

def _sweep_buy(depth: OrderDepth, pos: int, max_price: int, limit: int) -> List[Order]:
    want = limit - pos
    if want <= 0 or not depth.sell_orders: return []
    if min(depth.sell_orders) > max_price: return []
    return [Order(PRODUCT, max_price, want)]

def _sweep_sell(depth: OrderDepth, pos: int, min_price: int, limit: int) -> List[Order]:
    want = limit + pos
    if want <= 0 or not depth.buy_orders: return []
    if max(depth.buy_orders) < min_price: return []
    return [Order(PRODUCT, min_price, -want)]

class Trader:
    def __init__(self) -> None:
        self.params = dict(PARAMS)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        
        # 1. Establish the "Spot" price for our underlying
        spot_price = 5250.0 # Default fair value fallback
        if PRODUCT in state.order_depths:
            mid = _mid(state.order_depths[PRODUCT])
            if mid is not None:
                spot_price = mid
                
        # 2. Process Options: Quote around Fair Value and calculate Total Delta
        total_options_delta = 0.0
        r = 0.0 # Assuming 0 risk-free rate
        T = self.params["days_to_expiry"] / 365.0
        vol = self.params["volatility"]
        edge = self.params["opt_edge"]
        
        for opt_sym, K in OPTIONS.items():
            if opt_sym not in state.order_depths:
                continue
                
            opt_depth = state.order_depths[opt_sym]
            opt_pos = int(state.position.get(opt_sym, 0))
            
            # Calculate Fair Value and Delta
            fv, delta = bs_call(spot_price, K, T, r, vol)
            
            # Accumulate the delta based on our current position in this option
            total_options_delta += (opt_pos * delta)
            
            # Market Make the option
            buy_px = math.floor(fv - edge)
            sell_px = math.ceil(fv + edge)
            
            opt_orders: List[Order] = []
            
            # Skew quotes based on inventory to avoid breaching limits
            if opt_pos > 50: buy_px -= 1
            if opt_pos < -50: sell_px += 1
            
            bq = min(20, LIMIT_OPTIONS - opt_pos) # Post small chunks
            sq = min(20, LIMIT_OPTIONS + opt_pos)
            
            if bq > 0: opt_orders.append(Order(opt_sym, int(buy_px), bq))
            if sq > 0: opt_orders.append(Order(opt_sym, int(sell_px), -sq))
                
            result[opt_sym] = opt_orders

        # 3. Process Underlying: Mean Reversion + Delta Hedging
        if PRODUCT in state.order_depths:
            depth = state.order_depths[PRODUCT]
            mid = _mid(depth)
            pos = int(state.position.get(PRODUCT, 0))
            orders: List[Order] = []
            
            if mid is not None:
                best_bid = max(depth.buy_orders)
                best_ask = min(depth.sell_orders)
                
                # Combine our directional mean-reversion with our delta hedge.
                # If total_options_delta is +50, we want our underlying pos to be -50.
                target_hedge_pos = -int(total_options_delta)
                effective_pos = pos - target_hedge_pos 
                
                buy_thresh  = int(self.params["buy_thresh"])
                sell_thresh = int(self.params["sell_thresh"])
                mm_size     = int(self.params["mm_size"])

                if mid < buy_thresh:
                    # Price cheap — aggressively sweep asks
                    orders = _sweep_buy(depth, pos, best_ask + 4, LIMIT_UNDERLYING)
                elif mid > sell_thresh:
                    # Price rich — aggressively sweep bids
                    orders = _sweep_sell(depth, pos, best_bid - 4, LIMIT_UNDERLYING)
                else:
                    # Neutral zone: Passive MM, but skewed by effective_pos to acquire our hedge
                    buy_px  = best_bid + 1
                    sell_px = best_ask - 1
                    
                    if effective_pos > 30: # We are too long compared to our ideal delta hedge
                        buy_px  -= 1
                        sell_px -= 1
                    elif effective_pos < -30: # We are too short compared to our ideal delta hedge
                        buy_px  += 1
                        sell_px += 1

                    buy_px  = min(buy_px,  best_ask - 1)
                    sell_px = max(sell_px, best_bid + 1)

                    if buy_px < sell_px:
                        bq = min(mm_size, LIMIT_UNDERLYING - pos)
                        sq = min(mm_size, LIMIT_UNDERLYING + pos)
                        if bq > 0: orders.append(Order(PRODUCT, int(buy_px), bq))
                        if sq > 0: orders.append(Order(PRODUCT, int(sell_px), -sq))

            result[PRODUCT] = orders

        return result, 0, ""
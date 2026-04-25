import math
from statistics import NormalDist
from typing import Dict, List
from datamodel import Order, TradingState, OrderDepth

# --- 1. THE HEDGEHOG MATH ENGINE ---
_N = NormalDist()

def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """Calculates the theoretical fair price of a European Call option."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2)

class Trader:
    def __init__(self):
        # Your specific 3-day Volatility Smile Coefficients
        self.A = 4.6520
        self.B = -8.3052
        self.C = 3.9732
        
        # Position limits for Round 3
        self.LIMIT_VOUCHER = 300

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        
        # --- 2. GET UNDERLYING ANCHOR ---
        extract_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not extract_depth or not extract_depth.buy_orders or not extract_depth.sell_orders:
            # If we don't have the underlying price, we can't calculate moneyness. Abort tick.
            return result, 0, ""

        s_best_bid = max(extract_depth.buy_orders.keys())
        s_best_ask = min(extract_depth.sell_orders.keys())
        s_mid = (s_best_bid + s_best_ask) / 2.0

        # Calculate Time to Expiry (T) in years.
        # Round 3 starts at 5 days. state.timestamp is in milliseconds.
        days_passed = state.timestamp / 1_000_000.0
        tte_years = max(0.0, (5.0 - days_passed) / 365.0)

        # --- 3. THE SCALPING LOOP ---
        strikes = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
        
        for strike in strikes:
            symbol = f"VEV_{strike}"
            depth = state.order_depths.get(symbol)
            
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            # A. Calculate Fair Volatility (Figure 6a/6b Logic)
            m = strike / s_mid
            v_hat = self.A * (m**2) + self.B * m + self.C
            
            # Prevent negative volatility edge cases from math
            v_hat = max(0.0001, v_hat) 

            # B. Calculate Fair Price (Figure 7a Logic)
            fair_px = bs_call_price(s_mid, strike, tte_years, v_hat)
            
            # C. Dynamic Execution Thresholds (Figure 6c Logic)
            # We tighten the edge for ATM options (high volume, safe) 
            # and widen it for deep OTM/ITM options (low liquidity, noisy).
            if 4800 <= strike <= 5300:
                edge = 1.8  # ATM/NTM: Tighter spread, trade more often
            else:
                edge = 3.5  # ITM/OTM: Wider spread, only trade massive errors

            # Current inventory for this specific voucher
            current_pos = state.position.get(symbol, 0)
            orders = []
            
            # Get best market prices
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())

            # D. The "Normalized" Trade Triggers (Figure 7b Logic)
            
            # Undervalued: Market is selling it cheap -> WE BUY
            if best_ask < fair_px - edge:
                # Calculate how much we can buy without breaching the 300 limit
                buy_vol = self.LIMIT_VOUCHER - current_pos
                if buy_vol > 0:
                    orders.append(Order(symbol, best_ask, buy_vol))
            
            # Overvalued: Market is buying it high -> WE SELL
            elif best_bid > fair_px + edge:
                # Calculate how much we can sell without breaching the -300 limit
                sell_vol = -self.LIMIT_VOUCHER - current_pos
                if abs(sell_vol) > 0:
                    orders.append(Order(symbol, best_bid, sell_vol))
                    
            result[symbol] = orders

        return result, 0, "Hedgehog IV Scalper Executing"

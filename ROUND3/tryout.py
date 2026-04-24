import math
from statistics import NormalDist
from typing import Dict, List
from datamodel import Order, TradingState, OrderDepth

# --- 1. REGRESSION COEFFICIENTS (From your 3-day analysis) ---
A, B, C = 4.6520, -8.3052, 3.9732

# --- 2. HYPERPARAMETERS ---
SCALP_THRESHOLD = 2.0      # Credits of mispricing to trigger a trade
EMA_WINDOW = 20            # "Fast" EMA for mean reversion
EMA_THRESHOLD = 4.0        # Credits away from EMA to trade underlying
MIN_EXTRINSIC = 1.0        # Disregard options with too little "time value"

_N = NormalDist()

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, S - K), (1.0 if S > K else 0.0)
    d1 = (math.log(S / K) + (0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2), _N.cdf(d1)

class Trader:
    def __init__(self):
        self.ema_extract = None
        self.alpha = 2 / (EMA_WINDOW + 1)

    def run(self, state: TradingState):
        result = {}
        
        # --- 3. PREPARE UNDERLYING ---
        extract_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not extract_depth or not extract_depth.buy_orders: return {}, 0, ""
        
        s_best_bid = max(extract_depth.buy_orders.keys())
        s_best_ask = min(extract_depth.sell_orders.keys())
        s_mid = (s_best_bid + s_best_ask) / 2.0
        
        # Update EMA for Mean Reversion
        if self.ema_extract is None: self.ema_extract = s_mid
        else: self.ema_extract = self.alpha * s_mid + (1 - self.alpha) * self.ema_extract
        
        tte_years = (5.0 - (state.timestamp / 1_000_000.0)) / 365.0

        # --- 4. IV SCALPING (The Parabola Alpha) ---
        for strike in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
            symbol = f"VEV_{strike}"
            depth = state.order_depths.get(symbol)
            if not depth or not depth.buy_orders: continue

            # Filter for Extrinsic Value (Hedgehog logic: disregard bottom-left outliers)
            intrinsic = max(0, s_mid - strike)
            market_mid = (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0
            if (market_mid - intrinsic) < MIN_EXTRINSIC: continue

            # A. Detrending: Get Fair IV from your fitted parabola
            m = strike / s_mid
            v_hat = A * (m**2) + B * m + C
            
            # B. Theoretical Price
            fair_px, _ = bs_call(s_mid, strike, tte_years, v_hat)
            
            # C. Scalping Signal (Figure 6c logic: Price Deviations)
            orders = []
            pos = state.position.get(symbol, 0)
            
            if min(depth.sell_orders.keys()) < fair_px - SCALP_THRESHOLD:
                vol = 300 - pos
                if vol > 0: orders.append(Order(symbol, min(depth.sell_orders.keys()), vol))
            elif max(depth.buy_orders.keys()) > fair_px + SCALP_THRESHOLD:
                vol = -300 - pos
                if abs(vol) > 0: orders.append(Order(symbol, max(depth.buy_orders.keys()), vol))
            
            result[symbol] = orders

        # --- 5. MEAN REVERSION (The "Hedge against bad luck") ---
        # Instead of strict delta hedging, we trade the underlying's own alpha
        extract_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
        extract_orders = []
        
        if s_mid < self.ema_extract - EMA_THRESHOLD:
            # Price is too low compared to EMA -> BUY
            vol = 200 - extract_pos
            if vol > 0: extract_orders.append(Order("VELVETFRUIT_EXTRACT", s_best_ask, vol))
        elif s_mid > self.ema_extract + EMA_THRESHOLD:
            # Price is too high compared to EMA -> SELL
            vol = -200 - extract_pos
            if abs(vol) > 0: extract_orders.append(Order("VELVETFRUIT_EXTRACT", s_best_bid, vol))
            
        result["VELVETFRUIT_EXTRACT"] = extract_orders

        return result, 0, ""
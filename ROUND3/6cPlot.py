import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log, sqrt
from statistics import NormalDist

# --- 1. CONFIGURATION & COEFFICIENTS ---
# Your specific coefficients from the 3-day Universal Smile
A, B, C = 4.6520, -8.3052, 3.9732

# Files paths (excluding ROUND3/)
FILES = [
    'prices_round_3_day_0.csv',
    'prices_round_3_day_1.csv',
    'prices_round_3_day_2.csv'
]
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
START_TTE_MAP = {0: 5.0, 1: 4.0, 2: 3.0}

_N = NormalDist()

# --- 2. MATH HELPERS (Standard Library) ---
def bs_call_price(S, K, T, sigma):
    """Calculates Fair Price using Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0: 
        return max(0.0, S - K)
    d1 = (log(S/K) + (0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2)

# --- 3. MULTI-STRIKE PRICE DETRENDING ---
print("Transforming IV deviations into Price space (Credits) for all strikes...")

plt.figure(figsize=(16, 9))
colors = plt.cm.get_cmap('tab10', len(STRIKES))

# A list to store medians to help analyze stationarity
strike_medians = []

for idx, K in enumerate(STRIKES):
    strike_results = []
    symbol = f"VEV_{K}"
    
    for day_idx, file_path in enumerate(FILES):
        if not os.path.exists(file_path): continue
        
        # Prosperity CSVs use semicolons
        df = pd.read_csv(file_path, sep=';')
        
        # Load Extract (Underlying) and Option prices
        extract = df[df['product'] == 'VELVETFRUIT_EXTRACT'][['timestamp', 'mid_price']].rename(columns={'mid_price': 'S'})
        option = df[df['product'] == symbol][['timestamp', 'mid_price']].rename(columns={'mid_price': 'market_px'})
        
        # Merge on timestamp to analyze tick-by-tick deviations
        merged = pd.merge(extract, option, on='timestamp').dropna()
        start_tte = START_TTE_MAP[day_idx]
        
        # Sample every 100 ticks for a clearer "spaghetti" visual
        for _, row in merged.iloc[::100].iterrows():
            T_years = (start_tte - (row['timestamp'] / 1_000_000.0)) / 365.0
            moneyness = K / row['S']
            
            # HEDGEHOG METHOD:
            # 1. Detrending: Get Fair IV from your Parabolas
            v_hat = A*(moneyness**2) + B*moneyness + C
            
            # 2. Transformation: Convert Fair IV back to Price
            # This is the expected "fair price" based on the market-average smile
            fair_px = bs_call_price(row['S'], K, T_years, v_hat)
            
            strike_results.append({
                'global_ts': (day_idx * 1_000_000) + row['timestamp'],
                'price_deviation': row['market_px'] - fair_px # Figure 6c Metric
            })
            
    res_df = pd.DataFrame(strike_results)
    if not res_df.empty:
        # Thick lines for improved visuals
        plt.plot(res_df['global_ts'], res_df['price_deviation'], 
                 label=symbol, 
                 color=colors(idx), 
                 lw=2.0, 
                 alpha=0.7)
        strike_medians.append(res_df['price_deviation'].median())

# --- 4. TRADING VISUALS & FORMATTING ---
plt.axhline(0, color='black', lw=2.5, zorder=1) # The "Fair Value" axis

# Add Arbitrage Thresholds (These are visual aid for setting your EDGE_TAKE)
plt.axhline(1.0, color='blue', linestyle=':', lw=1.5, alpha=0.5, label='Aggressive Entry (+1.0)')
plt.axhline(-1.0, color='blue', linestyle=':', lw=1.5, alpha=0.5)
plt.axhline(2.5, color='red', linestyle='--', lw=2.0, alpha=0.6, label='High-Confidence Entry (+2.5)')
plt.axhline(-2.5, color='red', linestyle='--', lw=2.0, alpha=0.6)

# Titles and Labels using Bold/Raw strings for math safety
plt.title(r'$\mathbf{Figure\ 6c:\ Multi-Strike\ Price\ Deviations\ (Credits)}$', fontsize=18, pad=20)
plt.xlabel('Combined Timestamp (Day 0 $\\rightarrow$ Day 2)', fontsize=14, fontweight='bold')
plt.ylabel('Price Deviation (Market Price - Model Price)', fontsize=14, fontweight='bold')

plt.legend(title="Voucher Strike", loc='upper left', bbox_to_anchor=(1, 1), fontsize=11, frameon=True)
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.tight_layout()

plt.show()

# Quick Analysis print-out
print(f"Combined Median Deviation (Should be close to 0): {np.median(strike_medians):.4f}")
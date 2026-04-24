import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log, sqrt
from statistics import NormalDist

# --- 1. SETTINGS & COEFFICIENTS ---
A, B, C = 4.6520, -8.3052, 3.9732

FILES = [
    'prices_round_3_day_0.csv',
    'prices_round_3_day_1.csv',
    'prices_round_3_day_2.csv'
]
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
START_TTE_MAP = {0: 5.0, 1: 4.0, 2: 3.0}

_N = NormalDist()

def bs_call_price(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return max(0.0, S - K)
    d1 = (log(S/K) + (0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2)

def get_iv(market_price, S, K, T):
    low, high = 0.0001, 2.0
    for _ in range(12): 
        mid = (low + high) / 2
        if bs_call_price(S, K, T, mid) > market_price: high = mid
        else: low = mid
    return mid

# --- 2. DATA PROCESSING ---
plt.figure(figsize=(16, 9)) # Larger figure for better detail

# Use a distinct, high-contrast color map
colors = plt.cm.get_cmap('tab10', len(STRIKES))

for idx, K in enumerate(STRIKES):
    strike_results = []
    symbol = f"VEV_{K}"
    
    for day_idx, file_path in enumerate(FILES):
        if not os.path.exists(file_path): continue
        
        df = pd.read_csv(file_path, sep=';')
        extract = df[df['product'] == 'VELVETFRUIT_EXTRACT'][['timestamp', 'mid_price']].rename(columns={'mid_price': 'S'})
        option = df[df['product'] == symbol][['timestamp', 'mid_price']].rename(columns={'mid_price': 'market_px'})
        merged = pd.merge(extract, option, on='timestamp').dropna()
        
        start_tte = START_TTE_MAP[day_idx]
        
        # Sampling every 100 ticks to make lines thicker and clearer (less "fuzz")
        for _, row in merged.iloc[::100].iterrows():
            T_years = (start_tte - (row['timestamp'] / 1_000_000.0)) / 365.0
            v_market = get_iv(row['market_px'], row['S'], K, T_years)
            v_hat = A*( (K/row['S'])**2 ) + B*(K/row['S']) + C
            
            strike_results.append({
                'global_ts': (day_idx * 1_000_000) + row['timestamp'],
                'deviation': v_market - v_hat
            })
            
    res_df = pd.DataFrame(strike_results)
    if not res_df.empty:
        # THICKER LINES (lw=2.0) and smoother look
        plt.plot(res_df['global_ts'], res_df['deviation'], 
                 label=symbol, 
                 color=colors(idx), 
                 lw=2.0, 
                 alpha=0.85)

# --- 3. IMPROVED STYLING ---
plt.axhline(0, color='black', lw=2.5, zorder=1) # Thicker center line

# Bold fonts and clearer labels
plt.title(r'$\mathbf{Figure\ 6b:\ Multi-Strike\ IV\ Deviations\ (v_t - \hat{v}_t)}$', fontsize=18, pad=20)
plt.xlabel('Combined Timestamp (Day 0 $\\rightarrow$ Day 2)', fontsize=14, fontweight='bold')
plt.ylabel('IV Deviation (Market - Parabola)', fontsize=14, fontweight='bold')

# Better legend placement
plt.legend(title="Voucher Strike", loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, frameon=True, shadow=True)

# Cleaner Grid
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log, sqrt
from statistics import NormalDist

# --- 1. CONFIGURATION & COEFFICIENTS ---
A, B, C = 4.6520, -8.3052, 3.9732

FILES = [
    'prices_round_3_day_0.csv',
    'prices_round_3_day_1.csv',
    'prices_round_3_day_2.csv'
]
TARGET_VOUCHER = 'VEV_5000'
STRIKE = 5000
START_TTE_MAP = {0: 5.0, 1: 4.0, 2: 3.0}

_N = NormalDist()

# --- 2. BLACK-SCHOLES MATH ---
def bs_call_price(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: 
        return max(0.0, S - K)
    d1 = (log(S/K) + (0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2)

# --- 3. DATA PROCESSING ---
all_plot_data = []

print(f"Generating Normalized Fluctuations (7b) for {TARGET_VOUCHER}...")

for i, file_path in enumerate(FILES):
    if not os.path.exists(file_path): continue
    
    df = pd.read_csv(file_path, sep=';')
    extract = df[df['product'] == 'VELVETFRUIT_EXTRACT'][['timestamp', 'mid_price']].rename(columns={'mid_price': 'S'})
    option = df[df['product'] == TARGET_VOUCHER][['timestamp', 'mid_price']].rename(columns={'mid_price': 'market_mid'})
    merged = pd.merge(extract, option, on='timestamp').dropna()
    
    start_tte = START_TTE_MAP[i]
    
    for _, row in merged.iterrows():
        T_years = (start_tte - (row['timestamp'] / 1_000_000.0)) / 365.0
        m = STRIKE / row['S']
        
        # Fair IV from your fitted parabola
        v_hat = A*(m**2) + B*m + C
        
        # Fair Price using that IV
        theory_px = bs_call_price(row['S'], STRIKE, T_years, v_hat)
        
        all_plot_data.append({
            'global_ts': (i * 1_000_000) + row['timestamp'],
            'normalized_diff': row['market_mid'] - theory_px
        })

plot_df = pd.DataFrame(all_plot_data)

# --- 4. PLOTTING ---
plt.figure(figsize=(16, 6))

# The Normalized Signal (The Alpha)
plt.plot(plot_df['global_ts'], plot_df['normalized_diff'], 
         color='#1f77b4', lw=1.0, label='Price Deviation (Market - Theory)')

# Baseline (The Target)
plt.axhline(0, color='orange', lw=2.5, label='Theoretical Baseline')

# Execution Thresholds
plt.axhline(2.0, color='red', linestyle='--', alpha=0.6, label='Sell Threshold (+2.0)')
plt.axhline(-2.0, color='green', linestyle='--', alpha=0.6, label='Buy Threshold (-2.0)')

# Styling with Raw Strings for LaTeX safety
plt.title(r'$\mathbf{Figure\ 7b:\ Normalized\ VEV\_5000\ Price\ Fluctuations}$', fontsize=16)
plt.xlabel(r'Combined Timestamp (Day 0 $\rightarrow$ Day 2)', fontsize=12)
plt.ylabel('Deviation from Fair Value (Credits)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
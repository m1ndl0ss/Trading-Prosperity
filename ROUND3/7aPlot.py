import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log, sqrt
from statistics import NormalDist

# --- 1. CONFIGURATION & COEFFICIENTS ---
A, B, C = 4.6520, -8.3052, 3.9732

# Paths updated to exclude ROUND3/ as requested
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

print(f"Generating Price Fluctuation data for {TARGET_VOUCHER}...")

for i, file_path in enumerate(FILES):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found, skipping.")
        continue
    
    df = pd.read_csv(file_path, sep=';')
    extract = df[df['product'] == 'VELVETFRUIT_EXTRACT'][['timestamp', 'mid_price']].rename(columns={'mid_price': 'S'})
    option = df[df['product'] == TARGET_VOUCHER][['timestamp', 'mid_price']].rename(columns={'mid_price': 'market_mid'})
    merged = pd.merge(extract, option, on='timestamp').dropna()
    
    start_tte = START_TTE_MAP[i]
    
    # Sampling for clarity (every 50 ticks)
    for _, row in merged.iloc[::50].iterrows():
        T_years = (start_tte - (row['timestamp'] / 1_000_000.0)) / 365.0
        m = STRIKE / row['S']
        
        # Fair IV from your fitted parabola
        v_hat = A*(m**2) + B*m + C
        
        # Fair Price using that IV
        theoretical_px = bs_call_price(row['S'], STRIKE, T_years, v_hat)
        
        all_plot_data.append({
            'global_ts': (i * 1_000_000) + row['timestamp'],
            'market_px': row['market_mid'],
            'theory_px': theoretical_px
        })

plot_df = pd.DataFrame(all_plot_data)

# --- 4. PLOTTING ---
plt.figure(figsize=(16, 8))

# Actual Market Price
plt.plot(plot_df['global_ts'], plot_df['market_px'], 
         label='Market Mid Price (VEV_5000)', color='#1f77b4', lw=1.2, alpha=0.8)

# Your Theoretical Model Price
plt.plot(plot_df['global_ts'], plot_df['theory_px'], 
         label='Theoretical Fair Price (Hedgehog Smile Model)', color='#ff7f0e', lw=2.0)

# RAW STRINGS (r'') are used here to fix the ParseException
plt.title(r'$\mathbf{Figure\ 7a:\ VEV\_5000\ Price\ Fluctuations\ vs.\ Theoretical\ Value}$', fontsize=16)
plt.xlabel(r'Combined Timestamp (Day 0 $\rightarrow$ Day 2)', fontsize=12)
plt.ylabel('Price (Credits)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
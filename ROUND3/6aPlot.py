import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log, sqrt
from statistics import NormalDist

# --- 1. CONFIGURATION ---
# Paths to your uploaded files
FILES = [
    'prices_round_3_day_0.csv',
    'prices_round_3_day_1.csv',
    'prices_round_3_day_2.csv'
]
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# TTE Logic: Day 0 (5 days), Day 1 (4 days), Day 2 (3 days)
START_TTE_MAP = {0: 5.0, 1: 4.0, 2: 3.0}

# --- 2. BLACK-SCHOLES & IV SOLVER ---
_N = NormalDist()

def bs_call_price(S, K, T, sigma):
    """Fair price using standard library for compatibility."""
    if T <= 0 or sigma <= 0 or S <= 0: return max(0.0, S - K)
    d1 = (log(S/K) + (0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2)

def find_iv(market_price, S, K, T):
    """Bisection method to find Implied Volatility."""
    low, high = 0.0001, 2.0
    for _ in range(15):
        mid = (low + high) / 2
        if bs_call_price(S, K, T, mid) > market_price: high = mid
        else: low = mid
    return mid

# --- 3. DATA PROCESSING ---
all_medians = []
raw_scatter_m = []
raw_scatter_v = []

print("Processing files and calculating IVs...")

for i, file_path in enumerate(FILES):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping...")
        continue
    
    df = pd.read_csv(file_path, sep=';')
    extract = df[df['product'] == 'VELVETFRUIT_EXTRACT'][['timestamp', 'mid_price']].rename(columns={'mid_price': 'S'})
    day_start_tte = START_TTE_MAP[i]
    
    for K in STRIKES:
        option_data = df[df['product'] == f'VEV_{K}'][['timestamp', 'mid_price']]
        merged = pd.merge(extract, option_data, on='timestamp').dropna()
        
        if merged.empty: continue
        
        # Sample every 100 ticks to keep processing fast and reduce noise
        sample = merged.iloc[::100]
        ivs = []
        moneyness = []
        
        for _, row in sample.iterrows():
            # Time to expiry in years
            T_years = (day_start_tte - (row['timestamp'] / 1_000_000.0)) / 365.0
            iv = find_iv(row['mid_price'], row['S'], K, T_years)
            m = K / row['S']
            
            ivs.append(iv)
            moneyness.append(m)
            raw_scatter_m.append(m)
            raw_scatter_v.append(iv)
        
        # Store median for the strike/day to define the "Clean Smile"
        all_medians.append({'m': np.median(moneyness), 'v': np.median(ivs)})

# Create DataFrame for clean points
clean_points = pd.DataFrame(all_medians)

# --- 4. QUADRATIC REGRESSION ---
# Fit the curve: y = ax^2 + bx + c
coeffs = np.polyfit(clean_points['m'], clean_points['v'], 2)
poly_func = np.poly1d(coeffs)

# --- 5. PLOTTING ---
plt.figure(figsize=(12, 7))

# Plot raw data as a light background shadow (the "noise")
plt.scatter(raw_scatter_m, raw_scatter_v, alpha=0.05, s=2, color='gray', label='Raw Ticks (Day 0-2)')

# Plot the median points (the "signal")
plt.scatter(clean_points['m'], clean_points['v'], color='red', s=50, label='Strike Medians', edgecolors='black', zorder=5)

# Plot the regression line
x_range = np.linspace(min(raw_scatter_m), max(raw_scatter_m), 100)
plt.plot(x_range, poly_func(x_range), color='blue', lw=3, label='Hedgehog Regression Fit')

# Formatting
plt.title('Hedgehog Volatility Smile Regression (Combined Day 0, 1, 2)', fontsize=14)
plt.xlabel('Moneyness ($m = K/S$)', fontsize=12)
plt.ylabel('Implied Volatility ($v$)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.0) # IV is usually between 0 and 100%

plt.tight_layout()
plt.show()

# Print coefficients for your Trader code
print("\n--- REGRESSION RESULTS ---")
print(f"a = {coeffs[0]:.6f}")
print(f"b = {coeffs[1]:.6f}")
print(f"c = {coeffs[2]:.6f}")
print(f"Formula: Fair_IV = {coeffs[0]:.4f}m² + {coeffs[1]:.4f}m + {coeffs[2]:.4f}")
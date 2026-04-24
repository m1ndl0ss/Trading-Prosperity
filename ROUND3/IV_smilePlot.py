import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log, sqrt, exp
from statistics import NormalDist

# --- 1. CONFIGURATION ---
CSV_FILE = 'prices_round_3_day_0.csv'
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
TTE_START_DAYS = 5.0

# --- 2. BLACK-SCHOLES & IV SEARCH (Standard Library Only) ---
_N = NormalDist()

def bs_call_price(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return max(0, S - K)
    d1 = (log(S / K) + (0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2)

def find_iv(market_price, S, K, T):
    low, high = 0.0001, 2.0  # Search range for Volatility (0.01% to 200%)
    for _ in range(15): # Bisection search
        mid = (low + high) / 2
        if bs_call_price(S, K, T, mid) > market_price: high = mid
        else: low = mid
    return mid

# --- 3. DATA LOADING ---
if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found. Please check your ROUND3 folder.")
else:
    df = pd.read_csv(CSV_FILE, sep=';')
    
    # Extract underlying prices
    underlying = df[df['product'] == 'VELVETFRUIT_EXTRACT'][['timestamp', 'mid_price']]
    underlying = underlying.rename(columns={'mid_price': 'S'})

    all_points = []

    print("Calculating Implied Volatilities (this may take a few seconds)...")

    # Iterate through each strike
    for K in STRIKES:
        option_data = df[df['product'] == f'VEV_{K}'][['timestamp', 'mid_price']]
        merged = pd.merge(underlying, option_data, on='timestamp').dropna()
        
        # We sample every 10th row to keep the plot clean and fast
        for i, row in merged.iloc[::10].iterrows():
            S = row['S']
            market_price = row['mid_price']
            
            # Calculate Time to Expiry in Years
            tte_years = (TTE_START_DAYS - (row['timestamp'] / 1_000_000.0)) / 365.0
            
            # Calculate Moneyness (m_t = K / S)
            mt = K / S
            
            # Calculate Implied Volatility (v_t)
            vt = find_iv(market_price, S, K, tte_years)
            
            # Hedgehogs filter: only keep points where IV is "reasonable"
            if 0.001 < vt < 1.5:
                all_points.append({'strike': K, 'mt': mt, 'vt': vt})

    plot_df = pd.DataFrame(all_points)

    # --- 4. PLOTTING ---
    plt.figure(figsize=(12, 7))
    
    # Color-code by strike so you can see the clusters
    for K in STRIKES:
        subset = plot_df[plot_df['strike'] == K]
        plt.scatter(subset['mt'], subset['vt'], label=f'Strike {K}', alpha=0.5, s=10)

    plt.title('Implied Volatility ($v_t$) vs. Moneyness ($m_t$)', fontsize=14)
    plt.xlabel('Moneyness ($m_t = K/S_t$)', fontsize=12)
    plt.ylabel('Implied Volatility ($v_t$)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Bonus: Show the coefficients for the quadratic fit (Hedgehog Alpha)
    z = np.polyfit(plot_df['mt'], plot_df['vt'], 2)
    print(f"\nYour Quadratic Fit Coefficients (a, b, c):")
    print(f"a: {z[0]:.6f}, b: {z[1]:.6f}, c: {z[2]:.6f}")
    print(f"Formula: Fair_IV = {z[0]:.4f} * (m^2) + {z[1]:.4f} * m + {z[2]:.4f}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION ---
FILES = [
    'prices_round_3_day_0.csv',
    'prices_round_3_day_1.csv',
    'prices_round_3_day_2.csv'
]
UNDERLYING = 'VELVETFRUIT_EXTRACT'
WINDOW = 1000  # The rolling window size (1000 ticks = roughly 10% of a trading day)

# --- 2. DATA PROCESSING ---
all_extract_returns = []

print(f"Calculating Rolling Autocorrelation for {UNDERLYING}...")

for i, file_path in enumerate(FILES):
    if not os.path.exists(file_path): continue
    
    df = pd.read_csv(file_path, sep=';')
    extract = df[df['product'] == UNDERLYING].copy()
    
    # Create continuous timestamp across 3 days
    extract['global_ts'] = (i * 1_000_000) + extract['timestamp']
    
    # Calculate tick-by-tick returns
    extract['returns'] = extract['mid_price'].pct_change()
    extract = extract.dropna()
    
    all_extract_returns.append(extract)

if all_extract_returns:
    combined_df = pd.concat(all_extract_returns, ignore_index=True)
    
    # --- A. CALCULATE ACTUAL ASSET AUTOCORRELATION ---
    # Shift returns by 1 to compare current tick with previous tick (Lag 1)
    combined_df['lag1_returns'] = combined_df['returns'].shift(1)
    
    # Calculate rolling correlation
    combined_df['rolling_autocorr'] = combined_df['returns'].rolling(window=WINDOW).corr(combined_df['lag1_returns'])
    
    # --- B. CALCULATE RANDOM BASELINE (The Hedgehog Method) ---
    # Generate a perfectly random sequence with the same standard deviation
    np.random.seed(42)
    random_returns = np.random.normal(0, combined_df['returns'].std(), len(combined_df))
    random_df = pd.DataFrame({'rand_returns': random_returns})
    random_df['rand_lag1'] = random_df['rand_returns'].shift(1)
    
    # Calculate rolling correlation for the fake random asset
    combined_df['random_autocorr'] = random_df['rand_returns'].rolling(window=WINDOW).corr(random_df['rand_lag1'])
    
    # --- 3. PLOTTING FIGURE 8 (Hedgehog Style) ---
    plt.figure(figsize=(16, 7))
    
    # Plot the Random Noise Baseline (Red)
    plt.plot(combined_df['global_ts'], combined_df['random_autocorr'], 
             color='#d62728', alpha=0.6, lw=1.5, label='Purely Random Sequence (Baseline)')
    
    # Plot Actual Velvetfruit Extract (Blue)
    plt.plot(combined_df['global_ts'], combined_df['rolling_autocorr'], 
             color='#1f77b4', lw=2.5, label=f'Actual {UNDERLYING}')
    
    # Formatting
    plt.axhline(0, color='black', lw=2, linestyle='--')
    plt.title(r'$\mathbf{Figure\ 8:\ Rolling\ 1-Lag\ Autocorrelation\ (Mean\ Reversion)}$', fontsize=16)
    plt.xlabel(r'Combined Timestamp (Day 0 $\rightarrow$ Day 2)', fontsize=12)
    plt.ylabel(f'{WINDOW}-Tick Rolling Autocorrelation', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
else:
    print("No data found to plot. Check your CSV files.")
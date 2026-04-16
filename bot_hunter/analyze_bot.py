# -*- coding: utf-8 -*-
"""
Bot Hunter Analysis - IMC Prosperity Round 1
=============================================
Goal: Identify a mystery bot that consistently trades just before price moves,
      using trade fingerprints (price offset, size, timing, direction vs future move).

Since buyer/seller fields are empty in the data, we infer trade direction from
the trade price relative to the bid/ask at that timestamp:
  - price >= ask_price_1  -> buy aggressor (expects price UP)
  - price <= bid_price_1  -> sell aggressor (expects price DOWN)
  - between bid/ask       -> ambiguous (kept but flagged)

We then check: did the price actually move in the predicted direction?
A bot that is "always right" will show high predictive accuracy.
"""

import os
import sys
import io
import warnings

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripted runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(SCRIPT_DIR, "..", "Data")
ROUND1_DIR   = os.path.join(DATA_ROOT, "ROUND1")
TUTORIAL_DIR = os.path.join(DATA_ROOT, "TUTORIAL")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Future-look horizons (in price-row steps, each step = 100 ms)
FUTURE_STEPS = [5, 10, 20]

# Isolation Forest contamination — fraction of trades we suspect are "bot"
IF_CONTAMINATION = 0.08

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_round1_data():
    """
    Load all three days of Round 1 prices and trades.
    Returns two DataFrames: prices, trades.
    Each row carries a 'day' column for traceability.
    """
    price_frames = []
    trade_frames = []

    for day in [-2, -1, 0]:
        price_file = os.path.join(ROUND1_DIR, f"prices_round_1_day_{day}.csv")
        trade_file = os.path.join(ROUND1_DIR, f"trades_round_1_day_{day}.csv")

        if not os.path.exists(price_file):
            print(f"  [WARN] Missing price file: {price_file}")
            continue
        if not os.path.exists(trade_file):
            print(f"  [WARN] Missing trade file: {trade_file}")
            continue

        p = pd.read_csv(price_file, sep=";")
        t = pd.read_csv(trade_file, sep=";")

        # Ensure the day column exists (it is already present in prices)
        if "day" not in t.columns:
            t["day"] = day

        price_frames.append(p)
        trade_frames.append(t)
        print(f"  Loaded day {day:+d}: {len(p):,} price rows, {len(t):,} trade rows")

    if not price_frames:
        sys.exit("ERROR: No data files found. Check DATA_ROOT path.")

    prices = pd.concat(price_frames, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True)

    # Normalise column names
    trades.rename(columns={"symbol": "product"}, inplace=True)

    return prices, trades


# ---------------------------------------------------------------------------
# 2. Merge trades with price snapshots
# ---------------------------------------------------------------------------

def build_price_lookup(prices):
    """
    Build a nested dict: product → day → sorted (timestamps, mid_price array)
    for fast nearest-timestamp lookup.
    Also stores bid_price_1 and ask_price_1 for direction inference.
    """
    lookup = {}
    for (product, day), grp in prices.groupby(["product", "day"]):
        grp_sorted = grp.sort_values("timestamp").reset_index(drop=True)
        lookup.setdefault(product, {})[day] = {
            "timestamps": grp_sorted["timestamp"].values,
            "mid_price":  grp_sorted["mid_price"].values,
            "bid1":       grp_sorted["bid_price_1"].values,
            "ask1":       grp_sorted["ask_price_1"].values,
            "row_index":  grp_sorted.index.values,   # positional index for future-step lookup
        }
    return lookup


def get_price_at(lookup, product, day, timestamp, field="mid_price"):
    """Return the price-field value at the given timestamp (exact match expected)."""
    if product not in lookup or day not in lookup[product]:
        return np.nan
    rec = lookup[product][day]
    idx = np.searchsorted(rec["timestamps"], timestamp, side="left")
    if idx >= len(rec["timestamps"]) or rec["timestamps"][idx] != timestamp:
        # Fall back to last known price before timestamp
        idx = max(0, np.searchsorted(rec["timestamps"], timestamp, side="right") - 1)
    return rec[field][idx]


def get_future_mid(lookup, product, day, timestamp, steps):
    """
    Return mid_price N *steps* into the future (each step = one price row = 100 ms).
    Returns NaN if out of range.
    """
    if product not in lookup or day not in lookup[product]:
        return np.nan
    rec = lookup[product][day]
    idx = np.searchsorted(rec["timestamps"], timestamp, side="left")
    if idx >= len(rec["timestamps"]):
        return np.nan
    future_idx = idx + steps
    if future_idx >= len(rec["timestamps"]):
        return np.nan
    return rec["mid_price"][future_idx]


def merge_features(trades, prices):
    """
    Join every trade with contemporaneous price data and compute raw features:
      - mid_price, bid1, ask1 at trade timestamp
      - price_offset = trade_price − mid_price
      - direction:  +1 = buy aggressor (trade >= ask), -1 = sell aggressor (trade <= bid), 0 = ambiguous
      - future_mid_5/10/20 and future_change_5/10/20
      - correct_5/10/20:  1 if direction * future_change > 0 (prediction was right)
    """
    lookup = build_price_lookup(prices)

    rows = []
    for _, tr in trades.iterrows():
        product   = tr["product"]
        day       = tr["day"]
        ts        = tr["timestamp"]
        tprice    = tr["price"]
        qty       = tr["quantity"]

        mid  = get_price_at(lookup, product, day, ts, "mid_price")
        bid1 = get_price_at(lookup, product, day, ts, "bid1")
        ask1 = get_price_at(lookup, product, day, ts, "ask1")

        if np.isnan(mid):
            continue

        offset = tprice - mid

        # Infer direction from price vs bid/ask
        if not np.isnan(ask1) and tprice >= ask1:
            direction = +1   # buy aggressor
        elif not np.isnan(bid1) and tprice <= bid1:
            direction = -1   # sell aggressor
        else:
            direction = 0    # ambiguous / crossed spread

        # Future mid prices and changes
        futures = {}
        for n in FUTURE_STEPS:
            fmid = get_future_mid(lookup, product, day, ts, n)
            fchg = (fmid - mid) if not np.isnan(fmid) else np.nan
            futures[f"future_mid_{n}"]    = fmid
            futures[f"future_change_{n}"] = fchg
            if direction != 0 and not np.isnan(fchg):
                futures[f"correct_{n}"] = int(direction * fchg > 0)
            else:
                futures[f"correct_{n}"] = np.nan

        rows.append({
            "day":       day,
            "timestamp": ts,
            "product":   product,
            "trade_price": tprice,
            "quantity":  qty,
            "mid_price": mid,
            "bid1":      bid1,
            "ask1":      ask1,
            "spread":    (ask1 - bid1) if (not np.isnan(ask1) and not np.isnan(bid1)) else np.nan,
            "price_offset": offset,
            "direction": direction,
            **futures,
        })

    df = pd.DataFrame(rows)

    # Time-since-last-trade per product per day (in ms)
    df.sort_values(["product", "day", "timestamp"], inplace=True)
    df["time_since_last"] = df.groupby(["product", "day"])["timestamp"].diff().fillna(9999)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Analysis functions
# ---------------------------------------------------------------------------

def analysis_offset_accuracy(df, horizon=10):
    """
    Analysis 1: Predictive accuracy by price_offset bucket.
    Groups trades into offset bins and computes mean correctness.
    """
    print("\n" + "="*70)
    print(f"ANALYSIS 1 — Predictive accuracy by price_offset (horizon={horizon} steps)")
    print("="*70)

    col = f"correct_{horizon}"
    sub = df[df["direction"] != 0].dropna(subset=[col, "price_offset"]).copy()

    # Bin offsets into buckets of width 2
    sub["offset_bin"] = (sub["price_offset"] / 2).round() * 2

    grouped = (
        sub.groupby(["product", "offset_bin"])[col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_trades"})
        .reset_index()
    )
    grouped = grouped[grouped["n_trades"] >= 5]   # min sample threshold
    grouped.sort_values("accuracy", ascending=False, inplace=True)

    for product in grouped["product"].unique():
        sub_prod = grouped[grouped["product"] == product]
        print(f"\n  Product: {product}")
        print(f"  {'Offset Bin':>12} | {'Accuracy':>9} | {'N Trades':>9}")
        print(f"  {'-'*37}")
        for _, row in sub_prod.iterrows():
            marker = " <<< TOP" if row["accuracy"] >= 0.65 else ""
            print(f"  {row['offset_bin']:>12.1f} | {row['accuracy']:>9.1%} | {row['n_trades']:>9.0f}{marker}")

    return grouped


def analysis_sequential_patterns(df, window_ms=1000, horizon=10):
    """
    Analysis 2: Find bursts of trades within a short window that precede a directional move.
    A 'burst' is 3+ trades within `window_ms` milliseconds on the same product/day.
    """
    print("\n" + "="*70)
    print(f"ANALYSIS 2 — Sequential pattern mining (burst window={window_ms}ms, horizon={horizon})")
    print("="*70)

    results = []
    col = f"correct_{horizon}"

    for (product, day), grp in df.groupby(["product", "day"]):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        timestamps = grp["timestamp"].values

        for i in range(len(timestamps)):
            window_end = timestamps[i] + window_ms
            in_window  = np.where(timestamps <= window_end)[0]
            in_window  = in_window[in_window >= i]
            if len(in_window) < 3:
                continue

            burst = grp.loc[in_window]
            burst_correct = burst[col].dropna()
            if len(burst_correct) < 2:
                continue

            results.append({
                "product":        product,
                "day":            day,
                "burst_start_ts": int(timestamps[i]),
                "burst_size":     len(in_window),
                "burst_accuracy": burst_correct.mean(),
                "mean_qty":       burst["quantity"].mean(),
            })

    if not results:
        print("  No bursts found.")
        return pd.DataFrame()

    res_df = pd.DataFrame(results).drop_duplicates(subset=["product","day","burst_start_ts"])
    high_acc = res_df[res_df["burst_accuracy"] >= 0.70].sort_values("burst_accuracy", ascending=False)

    print(f"  Total bursts found: {len(res_df)}")
    print(f"  High-accuracy bursts (>=70%): {len(high_acc)}")
    if not high_acc.empty:
        print(f"\n  Top 15 high-accuracy bursts:")
        print(f"  {'Product':25} | {'Day':>4} | {'Start TS':>9} | {'Size':>5} | {'Acc':>7} | {'Avg Qty':>8}")
        print(f"  {'-'*70}")
        for _, row in high_acc.head(15).iterrows():
            print(f"  {row['product']:25} | {int(row['day']):>4} | {int(row['burst_start_ts']):>9} | "
                  f"{int(row['burst_size']):>5} | {row['burst_accuracy']:>7.1%} | {row['mean_qty']:>8.2f}")

    return res_df


def analysis_qty_accuracy(df, horizon=10):
    """
    Analysis 3: Does a specific quantity correlate with higher predictive accuracy?
    """
    print("\n" + "="*70)
    print(f"ANALYSIS 3 — Trade size vs predictive accuracy (horizon={horizon})")
    print("="*70)

    col = f"correct_{horizon}"
    sub = df[df["direction"] != 0].dropna(subset=[col]).copy()

    grouped = (
        sub.groupby(["product", "quantity"])[col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= 5].sort_values(["product", "accuracy"], ascending=[True, False])

    for product in grouped["product"].unique():
        sub_p = grouped[grouped["product"] == product]
        print(f"\n  Product: {product}")
        print(f"  {'Qty':>5} | {'Accuracy':>9} | {'N':>6}")
        print(f"  {'-'*27}")
        for _, row in sub_p.iterrows():
            marker = " <<< TOP" if row["accuracy"] >= 0.60 else ""
            print(f"  {int(row['quantity']):>5} | {row['accuracy']:>9.1%} | {int(row['n']):>6}{marker}")

    return grouped


def analysis_price_deviation(df, horizon=10):
    """
    Analysis 4: Which exact price deviation from mid has highest hit rate?
    Focuses on common offset values.
    """
    print("\n" + "="*70)
    print(f"ANALYSIS 4 — Price deviation from mid vs hit rate (horizon={horizon})")
    print("="*70)

    col = f"correct_{horizon}"
    sub = df[df["direction"] != 0].dropna(subset=[col, "price_offset"]).copy()

    grouped = (
        sub.groupby(["product", "price_offset"])[col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= 3].sort_values(["product", "accuracy"], ascending=[True, False])

    for product in grouped["product"].unique():
        sub_p = grouped[grouped["product"] == product].head(15)
        print(f"\n  Product: {product} — top 15 offsets by accuracy")
        print(f"  {'Offset':>8} | {'Accuracy':>9} | {'N':>6}")
        print(f"  {'-'*30}")
        for _, row in sub_p.iterrows():
            marker = " <<< HOT" if row["accuracy"] >= 0.65 else ""
            print(f"  {row['price_offset']:>8.1f} | {row['accuracy']:>9.1%} | {int(row['n']):>6}{marker}")

    return grouped


def analysis_temporal_clustering(df, horizon=10, move_threshold=3.0):
    """
    Analysis 5: Find timestamps where a trade is followed by a consistent price move.
    We define "temporal hotspot" as a timestamp where the absolute future price change
    is above move_threshold AND direction was correct.
    """
    print("\n" + "="*70)
    print(f"ANALYSIS 5 — Temporal clustering of profitable trades (horizon={horizon})")
    print("="*70)

    col_chg = f"future_change_{horizon}"
    col_cor = f"correct_{horizon}"
    sub = df[df["direction"] != 0].dropna(subset=[col_chg, col_cor]).copy()
    sub["big_move"]  = sub[col_chg].abs() >= move_threshold
    sub["hot_trade"] = (sub[col_cor] == 1) & sub["big_move"]

    hot = sub[sub["hot_trade"]].copy()

    print(f"  Total directional trades: {len(sub)}")
    print(f"  Big-move trades (|mid_change| >= {move_threshold}): {sub['big_move'].sum()}")
    print(f"  Correctly predicted big moves: {len(hot)}")
    print(f"  Hit rate on big moves: {(hot['hot_trade'].sum() / sub['big_move'].sum())*100:.1f}%")

    for product in sub["product"].unique():
        sub_p  = sub[sub["product"] == product]
        hot_p  = hot[hot["product"] == product]
        print(f"\n  Product: {product}")
        print(f"    Hot trades: {len(hot_p)}/{len(sub_p)}")
        if not hot_p.empty:
            print(f"    Qty distribution in hot trades: {dict(hot_p['quantity'].value_counts().sort_index())}")
            print(f"    Offset distribution in hot trades: {dict(hot_p['price_offset'].round(1).value_counts().sort_index())}")

    return hot


def analysis_isolation_forest(df, horizon=10):
    """
    Analysis 6: Use Isolation Forest to find anomalously profitable trades.
    Features: price_offset, quantity, time_since_last, future_change, correct.
    Anomalies = trades the model thinks don't fit the "normal random noise" profile.
    """
    print("\n" + "="*70)
    print(f"ANALYSIS 6 — Isolation Forest anomaly detection (horizon={horizon})")
    print("="*70)

    col_chg = f"future_change_{horizon}"
    col_cor = f"correct_{horizon}"

    features = ["price_offset", "quantity", "time_since_last", col_chg]
    sub = df[df["direction"] != 0].dropna(subset=features + [col_cor]).copy()

    if len(sub) < 20:
        print("  Not enough data for Isolation Forest.")
        return sub

    X = sub[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iforest = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    sub = sub.copy()
    sub["anomaly_score"] = iforest.fit_predict(X_scaled)   # -1 = anomaly, +1 = normal
    sub["if_score_raw"]  = iforest.score_samples(X_scaled) # lower = more anomalous

    anomalies = sub[sub["anomaly_score"] == -1].copy()
    normal    = sub[sub["anomaly_score"] == +1].copy()

    print(f"  Trades analysed: {len(sub)}")
    print(f"  Anomalies flagged: {len(anomalies)} ({len(anomalies)/len(sub)*100:.1f}%)")
    print()
    print(f"  Accuracy comparison:")
    print(f"    Anomalies  — correct@{horizon}: {anomalies[col_cor].mean()*100:.1f}%  (n={len(anomalies)})")
    print(f"    Normal     — correct@{horizon}: {normal[col_cor].mean()*100:.1f}%  (n={len(normal)})")
    print()

    # Profile the anomalies
    print(f"  Anomaly profile:")
    for product in anomalies["product"].unique():
        a_p = anomalies[anomalies["product"] == product]
        print(f"    Product: {product}  ({len(a_p)} anomalies)")
        print(f"      price_offset : mean={a_p['price_offset'].mean():.2f}, "
              f"std={a_p['price_offset'].std():.2f}, "
              f"values={sorted(a_p['price_offset'].round(1).unique().tolist())}")
        print(f"      quantity     : mean={a_p['quantity'].mean():.2f}, "
              f"mode={int(a_p['quantity'].mode().iloc[0])}, "
              f"values={sorted(a_p['quantity'].unique().tolist())}")
        print(f"      future_chg   : mean={a_p[col_chg].mean():.2f}, "
              f"std={a_p[col_chg].std():.2f}")
        print(f"      correct@{horizon}   : {a_p[col_cor].mean()*100:.1f}%")
        print(f"      direction +1 (buy): {(a_p['direction']==1).sum()} | "
              f"direction -1 (sell): {(a_p['direction']==-1).sum()}")

    return sub


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

def plot_offset_accuracy(offset_df, horizon=10):
    """Bar chart: price_offset bin vs predictive accuracy per product."""
    products = offset_df["product"].unique()
    fig, axes = plt.subplots(1, len(products), figsize=(7 * len(products), 5), squeeze=False)

    for ax, product in zip(axes[0], products):
        sub = offset_df[offset_df["product"] == product].sort_values("offset_bin")
        bars = ax.bar(sub["offset_bin"], sub["accuracy"], width=1.5,
                      color=["#e74c3c" if a >= 0.65 else "#3498db" for a in sub["accuracy"]],
                      edgecolor="white", linewidth=0.5)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="50% baseline")
        ax.set_title(f"{product}\nOffset vs Accuracy (h={horizon})", fontsize=10)
        ax.set_xlabel("Price Offset from Mid")
        ax.set_ylabel("Predictive Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        # Annotate n
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"n={int(row['n_trades'])}", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"offset_accuracy_h{horizon}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [Plot saved] {out}")


def plot_qty_accuracy(qty_df, horizon=10):
    """Bar chart: quantity vs predictive accuracy per product."""
    products = qty_df["product"].unique()
    fig, axes = plt.subplots(1, len(products), figsize=(6 * len(products), 4), squeeze=False)

    for ax, product in zip(axes[0], products):
        sub = qty_df[qty_df["product"] == product].sort_values("quantity")
        colors = ["#e74c3c" if a >= 0.60 else "#3498db" for a in sub["accuracy"]]
        ax.bar(sub["quantity"].astype(int), sub["accuracy"], color=colors, edgecolor="white")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="50% baseline")
        ax.set_title(f"{product}\nQuantity vs Accuracy (h={horizon})", fontsize=10)
        ax.set_xlabel("Trade Quantity")
        ax.set_ylabel("Predictive Accuracy")
        ax.set_ylim(0, 1)
        ax.set_xticks(sub["quantity"].astype(int).values)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"qty_accuracy_h{horizon}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [Plot saved] {out}")


def plot_isolation_forest(if_df, horizon=10):
    """Scatter: price_offset vs future_change, coloured by anomaly status."""
    col_chg = f"future_change_{horizon}"
    products = if_df["product"].unique()

    fig, axes = plt.subplots(1, len(products), figsize=(7 * len(products), 5), squeeze=False)

    for ax, product in zip(axes[0], products):
        sub = if_df[if_df["product"] == product].dropna(subset=[col_chg])
        normal   = sub[sub["anomaly_score"] == +1]
        anomaly  = sub[sub["anomaly_score"] == -1]

        ax.scatter(normal["price_offset"],  normal[col_chg],
                   c="#3498db", alpha=0.4, s=20, label="Normal")
        ax.scatter(anomaly["price_offset"], anomaly[col_chg],
                   c="#e74c3c", alpha=0.8, s=40, marker="D", label="Anomaly (potential bot)")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{product}\nIsolation Forest (h={horizon})", fontsize=10)
        ax.set_xlabel("Price Offset from Mid")
        ax.set_ylabel(f"Future Mid Change (+{horizon} steps)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"isolation_forest_h{horizon}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [Plot saved] {out}")


def plot_mid_price_with_trades(df, prices, product, day, horizon=10):
    """
    Line chart of mid_price over time with trades overlaid.
    Correct predictions shown in green, wrong in red, anomalies starred.
    """
    col_cor = f"correct_{horizon}"

    p_sub = prices[(prices["product"] == product) & (prices["day"] == day)].sort_values("timestamp")
    t_sub = df[(df["product"] == product) & (df["day"] == day)].dropna(subset=[col_cor]).copy()

    if p_sub.empty or t_sub.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(p_sub["timestamp"], p_sub["mid_price"], color="#2c3e50", linewidth=0.8,
            label="Mid Price", zorder=1)

    correct_mask   = t_sub[col_cor] == 1
    incorrect_mask = t_sub[col_cor] == 0

    ax.scatter(t_sub.loc[correct_mask,   "timestamp"],
               t_sub.loc[correct_mask,   "trade_price"],
               c="#27ae60", s=30, zorder=3, alpha=0.8, label="Correct prediction")
    ax.scatter(t_sub.loc[incorrect_mask, "timestamp"],
               t_sub.loc[incorrect_mask, "trade_price"],
               c="#e74c3c", s=30, zorder=3, alpha=0.8, label="Wrong prediction")

    # Highlight anomalies if column present
    if "anomaly_score" in t_sub.columns:
        anom_mask = t_sub["anomaly_score"] == -1
        ax.scatter(t_sub.loc[anom_mask, "timestamp"],
                   t_sub.loc[anom_mask, "trade_price"],
                   c="#f39c12", s=80, zorder=4, marker="*", label="IF Anomaly")

    ax.set_title(f"{product} — Day {day} — Trades vs Mid Price (h={horizon})", fontsize=11)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()

    safe_product = product.replace(" ", "_")
    out = os.path.join(PLOTS_DIR, f"timeseries_{safe_product}_day{day}_h{horizon}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [Plot saved] {out}")


def plot_accuracy_by_direction(df, horizon=10):
    """Grouped bar: accuracy for buy vs sell aggressor trades per product."""
    col = f"correct_{horizon}"
    sub = df[df["direction"] != 0].dropna(subset=[col]).copy()
    sub["dir_label"] = sub["direction"].map({1: "Buy aggressor", -1: "Sell aggressor"})

    grouped = (
        sub.groupby(["product", "dir_label"])[col]
        .agg(["mean", "count"])
        .reset_index()
    )

    products = grouped["product"].unique()
    fig, axes = plt.subplots(1, len(products), figsize=(5 * len(products), 4), squeeze=False)

    for ax, product in zip(axes[0], products):
        sub_p = grouped[grouped["product"] == product]
        bars = ax.bar(sub_p["dir_label"], sub_p["mean"],
                      color=["#27ae60", "#e74c3c"], edgecolor="white")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{product}\nAccuracy by Direction (h={horizon})", fontsize=10)
        ax.set_ylabel("Predictive Accuracy")
        ax.set_ylim(0, 1)
        for bar, (_, row) in zip(bars, sub_p.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{row['mean']*100:.0f}%\nn={int(row['count'])}", ha="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"accuracy_by_direction_h{horizon}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [Plot saved] {out}")


def plot_qty_distribution(df):
    """Histogram of trade quantities per product."""
    products = df["product"].unique()
    fig, axes = plt.subplots(1, len(products), figsize=(5 * len(products), 4), squeeze=False)

    for ax, product in zip(axes[0], products):
        sub = df[df["product"] == product]
        qtys = sub["quantity"].value_counts().sort_index()
        ax.bar(qtys.index.astype(int), qtys.values, color="#3498db", edgecolor="white")
        ax.set_title(f"{product}\nQuantity Distribution", fontsize=10)
        ax.set_xlabel("Quantity")
        ax.set_ylabel("Trade Count")
        ax.set_xticks(qtys.index.astype(int))

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "qty_distribution.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [Plot saved] {out}")


def plot_time_since_last_vs_accuracy(df, horizon=10):
    """Scatter: time_since_last_trade vs correctness, to see if timing matters."""
    col = f"correct_{horizon}"
    sub = df[df["direction"] != 0].dropna(subset=[col, "time_since_last"]).copy()
    # Clip extreme gaps for readability
    sub["time_clipped"] = sub["time_since_last"].clip(upper=10000)

    products = sub["product"].unique()
    fig, axes = plt.subplots(1, len(products), figsize=(7 * len(products), 4), squeeze=False)

    for ax, product in zip(axes[0], products):
        sub_p = sub[sub["product"] == product]
        colors = sub_p[col].map({1: "#27ae60", 0: "#e74c3c"})
        ax.scatter(sub_p["time_clipped"], sub_p["price_offset"].abs(),
                   c=colors, alpha=0.5, s=25)
        ax.set_title(f"{product}\nTime Gap vs |Price Offset| (h={horizon})\nGreen=correct, Red=wrong",
                     fontsize=9)
        ax.set_xlabel("Time Since Last Trade (ms, capped at 10k)")
        ax.set_ylabel("|Price Offset from Mid|")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"timing_vs_accuracy_h{horizon}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [Plot saved] {out}")


# ---------------------------------------------------------------------------
# 5. Fingerprint summary
# ---------------------------------------------------------------------------

def compute_bot_fingerprint(df, if_df, horizon=10):
    """
    Synthesise findings into a bot fingerprint summary.
    Looks for the offset/qty/direction combination with highest accuracy.
    """
    col_cor = f"correct_{horizon}"
    col_chg = f"future_change_{horizon}"

    # Overall accuracy
    directional = df[df["direction"] != 0].dropna(subset=[col_cor])
    overall_acc = directional[col_cor].mean()

    fingerprints = {}

    for product in df["product"].unique():
        sub = directional[directional["product"] == product].copy()
        if sub.empty:
            continue

        # Best offset
        off_acc = (
            sub.groupby("price_offset")[col_cor]
            .agg(["mean", "count"])
            .reset_index()
        )
        off_acc = off_acc[off_acc["count"] >= 5]
        best_offset_row = off_acc.loc[off_acc["mean"].idxmax()] if not off_acc.empty else None

        # Best qty
        qty_acc = (
            sub.groupby("quantity")[col_cor]
            .agg(["mean", "count"])
            .reset_index()
        )
        qty_acc = qty_acc[qty_acc["count"] >= 5]
        best_qty_row = qty_acc.loc[qty_acc["mean"].idxmax()] if not qty_acc.empty else None

        # Direction balance
        buy_count  = (sub["direction"] == +1).sum()
        sell_count = (sub["direction"] == -1).sum()

        # Most common offset
        common_offset = sub["price_offset"].mode().iloc[0] if not sub.empty else None

        # Most common qty
        common_qty = sub["quantity"].mode().iloc[0] if not sub.empty else None

        # Avg time between trades
        avg_gap = sub["time_since_last"].median()

        # From isolation forest anomalies
        if if_df is not None and "anomaly_score" in if_df.columns:
            if_sub = if_df[(if_df["product"] == product) & (if_df["anomaly_score"] == -1)]
            anom_acc  = if_sub[col_cor].mean() if not if_sub.empty else np.nan
            anom_qty  = if_sub["quantity"].mode().iloc[0] if not if_sub.empty else None
            anom_off  = if_sub["price_offset"].mode().iloc[0] if not if_sub.empty else None
            anom_n    = len(if_sub)
        else:
            anom_acc, anom_qty, anom_off, anom_n = np.nan, None, None, 0

        fingerprints[product] = {
            "overall_accuracy":    overall_acc,
            "product_accuracy":    sub[col_cor].mean(),
            "n_trades":            len(sub),
            "best_offset":         best_offset_row["price_offset"] if best_offset_row is not None else None,
            "best_offset_accuracy":best_offset_row["mean"] if best_offset_row is not None else None,
            "most_common_offset":  common_offset,
            "best_qty":            int(best_qty_row["quantity"]) if best_qty_row is not None else None,
            "best_qty_accuracy":   best_qty_row["mean"] if best_qty_row is not None else None,
            "most_common_qty":     int(common_qty) if common_qty is not None else None,
            "buy_aggressor_pct":   buy_count / len(sub) * 100,
            "sell_aggressor_pct":  sell_count / len(sub) * 100,
            "median_trade_gap_ms": avg_gap,
            "if_anomaly_count":    anom_n,
            "if_anomaly_accuracy": anom_acc,
            "if_anomaly_qty":      int(anom_qty) if anom_qty is not None else None,
            "if_anomaly_offset":   anom_off,
        }

    return fingerprints


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  BOT HUNTER -- IMC Prosperity Round 1 Analysis")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/7] Loading data...")
    prices, trades = load_round1_data()
    print(f"  Total price rows: {len(prices):,}")
    print(f"  Total trade rows: {len(trades):,}")
    print(f"  Products in prices: {sorted(prices['product'].unique())}")
    print(f"  Products in trades: {sorted(trades['product'].unique())}")

    # --- Merge and compute features ---
    print("\n[2/7] Merging trades with prices & computing features...")
    df = merge_features(trades, prices)
    print(f"  Feature-enriched trades: {len(df):,}")

    directional = df[df["direction"] != 0]
    print(f"  Directional trades (buy/sell aggressor): {len(directional):,} "
          f"({len(directional)/len(df)*100:.1f}%)")
    print(f"  Ambiguous (price between bid/ask):       {(df['direction']==0).sum():,}")

    for h in FUTURE_STEPS:
        col = f"correct_{h}"
        valid = directional.dropna(subset=[col])
        print(f"  Overall accuracy @{h:>2} steps: {valid[col].mean()*100:.1f}%  (n={len(valid)})")

    # --- Run analyses ---
    HORIZON = 10     # primary horizon used for most analyses

    print("\n[3/7] Running analyses...")

    offset_df  = analysis_offset_accuracy(df, horizon=HORIZON)
    burst_df   = analysis_sequential_patterns(df, window_ms=1000, horizon=HORIZON)
    qty_df     = analysis_qty_accuracy(df, horizon=HORIZON)
    dev_df     = analysis_price_deviation(df, horizon=HORIZON)
    hot_df     = analysis_temporal_clustering(df, horizon=HORIZON, move_threshold=3.0)
    if_df      = analysis_isolation_forest(df, horizon=HORIZON)

    # --- Plots ---
    print("\n[4/7] Generating plots...")
    plot_qty_distribution(df)
    plot_offset_accuracy(offset_df, horizon=HORIZON)
    plot_qty_accuracy(qty_df, horizon=HORIZON)
    plot_isolation_forest(if_df, horizon=HORIZON)
    plot_accuracy_by_direction(df, horizon=HORIZON)
    plot_time_since_last_vs_accuracy(df, horizon=HORIZON)

    # Merge anomaly column back into df for timeseries plots
    if "anomaly_score" in if_df.columns:
        df = df.merge(
            if_df[["day", "timestamp", "product", "anomaly_score"]].drop_duplicates(),
            on=["day", "timestamp", "product"],
            how="left",
        )

    for product in df["product"].unique():
        for day in df["day"].unique():
            plot_mid_price_with_trades(df, prices, product, day, horizon=HORIZON)

    # --- Multi-horizon accuracy table ---
    print("\n[5/7] Multi-horizon accuracy summary:")
    print(f"  {'Product':25} | {'h=5':>8} | {'h=10':>8} | {'h=20':>8}")
    print(f"  {'-'*55}")
    for product in sorted(df["product"].unique()):
        row_vals = []
        for h in FUTURE_STEPS:
            col = f"correct_{h}"
            sub = df[(df["product"] == product) & (df["direction"] != 0)].dropna(subset=[col])
            row_vals.append(f"{sub[col].mean()*100:.1f}%" if not sub.empty else "  N/A")
        print(f"  {product:25} | {row_vals[0]:>8} | {row_vals[1]:>8} | {row_vals[2]:>8}")

    # --- Fingerprint ---
    print("\n[6/7] Computing bot fingerprint...")
    fingerprints = compute_bot_fingerprint(df, if_df, horizon=HORIZON)

    # --- Final Conclusion ---
    print("\n" + "=" * 70)
    print("[7/7] FINAL CONCLUSION")
    print("=" * 70)

    for product, fp in fingerprints.items():
        print(f"\n{'─'*70}")
        print(f"BOT FINGERPRINT — {product}")
        print(f"{'─'*70}")
        print(f"  Trades analysed         : {fp['n_trades']}")
        print(f"  Overall directional acc : {fp['product_accuracy']*100:.1f}%")
        print()
        print(f"  Direction mix:")
        print(f"    Buy  aggressor  : {fp['buy_aggressor_pct']:.1f}%")
        print(f"    Sell aggressor  : {fp['sell_aggressor_pct']:.1f}%")
        print()
        print(f"  Best price offset (highest acc): {fp['best_offset']} "
              f"(acc={fp['best_offset_accuracy']*100:.1f}%  if set)")
        print(f"  Most frequent offset            : {fp['most_common_offset']}")
        print()
        print(f"  Best quantity (highest acc)     : qty={fp['best_qty']} "
              f"(acc={fp['best_qty_accuracy']*100:.1f}%  if set)")
        print(f"  Most frequent quantity          : qty={fp['most_common_qty']}")
        print()
        print(f"  Median time between trades      : {fp['median_trade_gap_ms']:.0f} ms")
        print()
        print(f"  Isolation Forest anomalies      : {fp['if_anomaly_count']} trades")
        if fp['if_anomaly_count'] > 0:
            acc_str = f"{fp['if_anomaly_accuracy']*100:.1f}%" if not np.isnan(fp['if_anomaly_accuracy']) else "N/A"
            print(f"    Anomaly accuracy (@{HORIZON} steps) : {acc_str}")
            print(f"    Anomaly typical qty             : {fp['if_anomaly_qty']}")
            print(f"    Anomaly typical offset          : {fp['if_anomaly_offset']}")

    print(f"\n{'='*70}")

    # Compose the one-line fingerprint for each product
    for product, fp in fingerprints.items():
        acc_pct = fp['product_accuracy'] * 100
        best_off = fp['best_offset']
        best_acc = fp['best_offset_accuracy'] * 100 if fp['best_offset_accuracy'] else 0
        best_qty = fp['best_qty']
        gap      = fp['median_trade_gap_ms']
        anom_acc = fp['if_anomaly_accuracy'] * 100 if not np.isnan(fp.get('if_anomaly_accuracy', np.nan)) else 0
        anom_qty = fp['if_anomaly_qty']

        dominant_dir = "BUY-aggressor" if fp['buy_aggressor_pct'] > 55 else (
                       "SELL-aggressor" if fp['sell_aggressor_pct'] > 55 else "MIXED")

        print(f"\nBOT FINGERPRINT [{product}]: "
              f"overall_accuracy={acc_pct:.1f}% | "
              f"best_offset={best_off} (acc={best_acc:.1f}%) | "
              f"best_qty={best_qty} | "
              f"dominant_direction={dominant_dir} | "
              f"median_trade_gap={gap:.0f}ms | "
              f"IF_anomaly_accuracy={anom_acc:.1f}% (n={fp['if_anomaly_count']}, "
              f"typical_qty={anom_qty})")

    print(f"\n{'='*70}")
    print("Analysis complete. Plots saved to:", PLOTS_DIR)
    print("="*70)


if __name__ == "__main__":
    main()

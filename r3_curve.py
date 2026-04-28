import json, io
import pandas as pd
import numpy as np

with open(r'LOGS/ROUND4/r3 Results/486918.json', encoding='utf-8') as f:
    data = json.load(f)

print("profit:", data.get('profit'))
log_str = ''.join(data.get('activitiesLog', []))
df = pd.read_csv(io.StringIO(log_str), sep=';')

# ── Total equity curve by timestamp ──────────────────────────────────────────
equity = df.groupby('timestamp')['profit_and_loss'].sum().reset_index()
equity.columns = ['ts', 'pnl']

peak_idx = equity['pnl'].idxmax()
trough_idx = equity['pnl'].idxmin()
peak_pnl = equity.loc[peak_idx, 'pnl']
trough_pnl = equity.loc[trough_idx, 'pnl']
peak_ts = equity.loc[peak_idx, 'ts']
trough_ts = equity.loc[trough_idx, 'ts']
end_pnl = equity['pnl'].iloc[-1]

print(f"\nEquity stats:")
print(f"  Peak:   {peak_pnl:>10,.0f}  at ts={peak_ts:,}")
print(f"  Trough: {trough_pnl:>10,.0f}  at ts={trough_ts:,}")
print(f"  End:    {end_pnl:>10,.0f}")
print(f"  Max drawdown from start to trough: {trough_pnl:>+,.0f}")
print(f"  Peak-to-trough:  {trough_pnl - peak_pnl:>+,.0f}")

# ── Per-product PnL at key timestamps ────────────────────────────────────────
print("\n=== Per-product final PnL ===")
final = df.groupby('product')['profit_and_loss'].last().sort_values(ascending=False)
for p, v in final[final.abs() > 10].items():
    print(f"  {p:30s}: {v:>10,.2f}")

# ── Who causes the drawdown? Compare at peak vs trough ────────────────────────
print("\n=== Product PnL at peak ts vs trough ts vs end ===")
products = df['product'].unique()
rows = []
for prod in products:
    sub = df[df['product']==prod].sort_values('timestamp').reset_index(drop=True)
    at_peak   = sub[sub['timestamp'] <= peak_ts]['profit_and_loss']
    at_trough = sub[sub['timestamp'] <= trough_ts]['profit_and_loss']
    at_end    = sub['profit_and_loss']
    v_peak   = at_peak.iloc[-1]   if len(at_peak)   > 0 else 0.0
    v_trough = at_trough.iloc[-1] if len(at_trough) > 0 else 0.0
    v_end    = at_end.iloc[-1]    if len(at_end)    > 0 else 0.0
    delta_drop    = v_trough - v_peak
    delta_recover = v_end - v_trough
    if abs(v_end) > 50 or abs(delta_drop) > 500:
        rows.append((prod, v_peak, v_trough, v_end, delta_drop, delta_recover))

rows.sort(key=lambda x: x[4])  # sort by drop, worst first
print(f"  {'product':30s}  {'at_peak':>10}  {'at_trough':>10}  {'at_end':>10}  {'drop':>10}  {'recover':>10}")
for prod, p, tr, e, dr, rc in rows:
    flag = " <<<" if abs(dr) > 3000 else ""
    print(f"  {prod:30s}  {p:>10,.0f}  {tr:>10,.0f}  {e:>10,.0f}  {dr:>+10,.0f}  {rc:>+10,.0f}{flag}")

# ── Equity curve checkpoints ─────────────────────────────────────────────────
print("\n=== Equity checkpoints every 50k timestamps ===")
for ts_target in range(0, 1000001, 50000):
    near = equity[equity['ts'] <= ts_target]
    if near.empty: continue
    pnl_v = near.iloc[-1]['pnl']
    ts_v  = near.iloc[-1]['ts']
    print(f"  ts~{ts_target:>7,}: pnl={pnl_v:>10,.0f}")

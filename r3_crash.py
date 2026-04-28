import json, io
import pandas as pd
import numpy as np

with open(r'LOGS/ROUND4/r3 Results/486918.json', encoding='utf-8') as f:
    data = json.load(f)

log_str = ''.join(data.get('activitiesLog', []))
df = pd.read_csv(io.StringIO(log_str), sep=';')

# ── Zoom into the crash window ───────────────────────────────────────────────
crash_start, crash_end = 380_000, 520_000
window = df[(df['timestamp'] >= crash_start) & (df['timestamp'] <= crash_end)]

print("=== Per-product PnL change inside crash window ===")
print(f"Window: ts {crash_start:,} -> {crash_end:,}")
print()

for prod in sorted(df['product'].unique()):
    sub = df[df['product'] == prod].sort_values('timestamp')
    before = sub[sub['timestamp'] <= crash_start]['profit_and_loss']
    after  = sub[sub['timestamp'] <= crash_end  ]['profit_and_loss']
    v_before = before.iloc[-1] if len(before) > 0 else 0.0
    v_after  = after.iloc[-1]  if len(after)  > 0 else 0.0
    delta = v_after - v_before
    if abs(delta) > 200:
        print(f"  {prod:30s}: {v_before:>10,.0f} -> {v_after:>10,.0f}  delta={delta:>+10,.0f}")

# ── Position held during crash ───────────────────────────────────────────────
print("\n=== Position & mid-price at peak (ts~400k) and trough (ts~470k) ===")
for prod in ['HYDROGEL_PACK', 'VELVETFRUIT_EXTRACT', 'VEV_4000', 'VEV_4500',
             'VEV_5000', 'VEV_5100', 'VEV_5200']:
    sub = df[df['product']==prod].sort_values('timestamp')
    for ts_check, label in [(400_000,'@400k'), (450_000,'@450k'), (470_000,'@470k')]:
        near = sub[sub['timestamp'] <= ts_check]
        if near.empty: continue
        row = near.iloc[-1]
        print(f"  {prod:30s} {label}: pos={row.get('position',0):>+5.0f}  "
              f"mid={(row.get('bid_price_1',0)+row.get('ask_price_1',0))/2:>8.1f}  "
              f"pnl={row['profit_and_loss']:>10,.0f}")
    print()

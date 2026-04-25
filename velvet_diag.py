import json, io
import pandas as pd
import numpy as np

# ---- Official run log ----
with open(r'LOGS/CLAUDE LOGS/velvet - +2.3k/401656.json', encoding='utf-8') as f:
    data = json.load(f)
print("profit:", data.get('profit'))
log_str = ''.join(data.get('activitiesLog', []))
df = pd.read_csv(io.StringIO(log_str), sep=';')
vf = df[df['product'] == 'VELVETFRUIT_EXTRACT'].copy().reset_index(drop=True)

mid = vf['mid_price'].values
pnl = vf['profit_and_loss'].values
ts  = vf['timestamp'].values
bid1 = vf['bid_price_1'].values
ask1 = vf['ask_price_1'].values
days_col = vf['day'].values

print("Days:", sorted(set(days_col.tolist())))
print("Ticks:", len(vf))
print("Mid range: {:.1f} - {:.1f}  mean: {:.2f}".format(mid.min(), mid.max(), mid.mean()))
print("PnL: start={:.2f}  min={:.2f}  max={:.2f}  end={:.2f}".format(
    pnl[0], pnl.min(), pnl.max(), pnl[-1]))
print("Ticks mid < 5230:", (mid < 5230).sum(), "  Ticks mid > 5275:", (mid > 5275).sum())
print("Ticks mid < 5240:", (mid < 5240).sum(), "  Ticks mid > 5270:", (mid > 5270).sum())

print("\nPnL checkpoints every 100 ticks:")
for i in range(0, len(vf), 100):
    print("  tick {:4d}  ts={:6d}  mid={:.1f}  pnl={:.2f}".format(
        i, int(ts[i]), mid[i], pnl[i]))

print("\nPosition trace (big PnL shifts):")
pnl_diff = np.diff(pnl)
BUY_T, SELL_T = 5230, 5275
pos, cash = 0, 0.0
prev_pos = 0
for i in range(len(mid)):
    m, a, b = mid[i], ask1[i], bid1[i]
    if m > SELL_T and pos > -200:
        q = 200 + pos; cash += q*b; pos -= q
    elif m < BUY_T and pos < 200:
        q = 200 - pos; cash -= q*a; pos += q
    if pos != prev_pos:
        equity = cash + pos*m
        print("  ts={:6d}  mid={:.1f}  pos: {:+d}->{:+d}  equity={:,.2f}".format(
            int(ts[i]), m, prev_pos, pos, equity))
        prev_pos = pos
print("  ts={:6d} (EOD) mid={:.1f}  pos={}  sim_pnl={:,.2f}".format(
    int(ts[-1]), mid[-1], pos, cash + pos*mid[-1]))

# ---- Training data intraday analysis ----
print("\n=== Training data — what actually happens intraday ===")
frames = []
for day in [0, 1, 2]:
    df2 = pd.read_csv('Data/ROUND3/prices_round_3_day_{}.csv'.format(day), sep=';')
    df2['day'] = day
    frames.append(df2)
all_p = pd.concat(frames)
tr = all_p[all_p['product'] == 'VELVETFRUIT_EXTRACT'].sort_values(['day','timestamp']).reset_index(drop=True)

for d in [0, 1, 2]:
    sub = tr[tr['day'] == d]['mid_price'].values
    n = len(sub)
    print("\nDay {} ({} ticks):".format(d, n))
    print("  min={:.1f}  max={:.1f}  mean={:.2f}  start={:.1f}  end={:.1f}".format(
        sub.min(), sub.max(), sub.mean(), sub[0], sub[-1]))
    # Decile path
    seg = n // 10
    for i in range(0, 10, 2):
        chunk = sub[i*seg:(i+2)*seg]
        print("  ticks {:5d}-{:5d}: mean={:.1f}  min={:.1f}  max={:.1f}".format(
            i*seg, (i+2)*seg, chunk.mean(), chunk.min(), chunk.max()))

# How much time does price spend above 5275 vs below 5230 — and WHEN?
print("\n=== Timing of threshold crossings (training) ===")
for d in [0, 1, 2]:
    sub = tr[tr['day'] == d][['timestamp','mid_price']].reset_index(drop=True)
    m = sub['mid_price'].values
    above = sub[m > 5275]['timestamp'].values
    below = sub[m < 5230]['timestamp'].values
    print("Day {}: above5275 ticks={:4d} ({:.1f}%)  below5230 ticks={:4d} ({:.1f}%)".format(
        d, len(above), 100*len(above)/len(m), len(below), 100*len(below)/len(m)))
    if len(above) > 0:
        print("  above5275: first_ts={:7d}  last_ts={:7d}".format(int(above[0]), int(above[-1])))
    if len(below) > 0:
        print("  below5230: first_ts={:7d}  last_ts={:7d}".format(int(below[0]), int(below[-1])))

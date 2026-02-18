"""
WHSDSC 2026 — Generate League Table (Phase 1a Prerequisite)
============================================================
Lightweight script: reads raw data, builds W/L/OTL/PTS table.
"""
import pandas as pd
import numpy as np

ROOT = "/Users/frankbisignano/Library/CloudStorage/GoogleDrive-frankabisignano@gmail.com/My Drive/WHSDSC_2026/"
df = pd.read_csv(ROOT + "whl_2025.csv")
SPEC = {'PP_up', 'PP_kill_dwn', 'PP_kill_up', 'empty_net_line'}

print(f"Loaded {len(df):,} records | {df['game_id'].nunique():,} games | {df['home_team'].nunique()} teams")

# ── Game aggregation ─────────────────────────────────────────
def agg_game(g):
    row = {}
    for s in ['home', 'away']:
        o = 'away' if s == 'home' else 'home'
        row[f'{s}_goals'] = g[f'{s}_goals'].sum()
        row[f'{s}_xg']    = g[f'{s}_xg'].sum()
        row[f'{s}_shots'] = g[f'{s}_shots'].sum()
        row[f'{s}_pen']   = g[f'{s}_penalties_committed'].sum()
        pp = g[g[f'{s}_off_line'] == 'PP_up']
        row[f'{s}_pp_xg'] = pp[f'{s}_xg'].sum()
        pk = g[g[f'{s}_off_line'].isin({'PP_kill_dwn', 'PP_kill_up'})]
        row[f'{s}_pk_xga'] = pk[f'{o}_xg'].sum()
    row['went_ot']   = g['went_ot'].iloc[0]
    row['home_team'] = g['home_team'].iloc[0]
    row['away_team'] = g['away_team'].iloc[0]
    return pd.Series(row)

print("Aggregating to game level...")
games = df.groupby('game_id').apply(agg_game).reset_index()
games['home_win'] = (games['home_goals'] > games['away_goals']).astype(int)
print(f"Games: {len(games)}")

# ── Build league table ───────────────────────────────────────
rows = []
for _, g in games.iterrows():
    hw = g['home_win']
    ot = g['went_ot']
    for side, opp_side, is_home in [('home', 'away', True), ('away', 'home', False)]:
        w = hw if is_home else 1 - hw
        rows.append({
            'team': g[f'{side}_team'],
            'gf': g[f'{side}_goals'],
            'ga': g[f'{opp_side}_goals'],
            'xgf': g[f'{side}_xg'],
            'xga': g[f'{opp_side}_xg'],
            'sf': g[f'{side}_shots'],
            'sa': g[f'{opp_side}_shots'],
            'win': w,
            'loss': 1 - w,
            'ot_loss': 1 if (w == 0 and ot == 1) else 0,
            'pp_xg': g[f'{side}_pp_xg'],
            'pk_xga': g[f'{side}_pk_xga'],
            'pen': g[f'{side}_pen'],
        })

tg = pd.DataFrame(rows)
tg['pts'] = tg['win'] * 2 + tg['ot_loss'] * 1

lt = tg.groupby('team').agg(
    GP=('win', 'count'),
    W=('win', 'sum'),
    L=('loss', 'sum'),
    OTL=('ot_loss', 'sum'),
    PTS=('pts', 'sum'),
    GF=('gf', 'sum'),
    GA=('ga', 'sum'),
    xGF=('xgf', 'sum'),
    xGA=('xga', 'sum'),
    SF=('sf', 'sum'),
    SA=('sa', 'sum'),
    PP_xGF=('pp_xg', 'sum'),
    PK_xGA=('pk_xga', 'sum'),
    PEN=('pen', 'sum'),
).reset_index()

lt['GD']  = lt['GF'] - lt['GA']
lt['xGD'] = lt['xGF'] - lt['xGA']
lt['xG%'] = (lt['xGF'] / (lt['xGF'] + lt['xGA'])).round(4)
lt['Pts/GP'] = (lt['PTS'] / lt['GP']).round(3)

lt = lt.sort_values('PTS', ascending=False).reset_index(drop=True)
lt.index += 1

print("\nLEAGUE TABLE:")
print(lt[['team', 'GP', 'W', 'L', 'OTL', 'PTS', 'Pts/GP', 'GF', 'GA', 'GD', 'xGF', 'xGA', 'xGD', 'xG%']].to_string())

# Save
out = ROOT + "league_table.csv"
lt.to_csv(out, index_label='rank')
print(f"\nSaved: {out}")

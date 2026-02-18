"""
WHSDSC 2026 — Maximum AUC Hockey Prediction Model
===================================================
Architecture:
  - ELO ratings (goals-based + xG-based)
  - Multi-window rolling stats (7, 15, 30 games)
  - Goalie GSAx (goals saved above expected)
  - Line quality features (first vs second line xG/60)
  - Stacked ensemble: GB × 2 + RF + ET + LR  →  LR meta

Outputs:
  - power_rankings.csv      (Phase 1a)
  - matchup_predictions.csv (Phase 1b win probabilities)
  - line_disparity.csv      (Phase 1c top 10 disparity teams)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, StackingClassifier
)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("WHSDSC 2026 — Maximum AUC Hockey Prediction Model")
print("=" * 60)

DATA_PATH = "/Users/frankbisignano/Library/CloudStorage/GoogleDrive-frankabisignano@gmail.com/My Drive/WHSDSC_2026/whl_2025.csv"
OUT_DIR   = "/Users/frankbisignano/Library/CloudStorage/GoogleDrive-frankabisignano@gmail.com/My Drive/WHSDSC_2026/"

df = pd.read_csv(DATA_PATH)
SPEC = {'PP_up', 'PP_kill_dwn', 'PP_kill_up', 'empty_net_line'}
print(f"\nLoaded {len(df):,} records | {df['game_id'].nunique():,} games | {df['home_team'].nunique()} teams")

# ══════════════════════════════════════════════════════════════════
# 2. GAME-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════
def agg_game(g):
    """Collapse line-level records into one game row."""
    reg = g[~g['home_off_line'].isin(SPEC) & ~g['away_off_line'].isin(SPEC)]
    row = {}
    for side in ['home', 'away']:
        opp = 'away' if side == 'home' else 'home'
        row[f'{side}_goals_total']   = g[f'{side}_goals'].sum()
        row[f'{side}_xg_total']      = g[f'{side}_xg'].sum()
        row[f'{side}_shots_total']   = g[f'{side}_shots'].sum()
        row[f'{side}_pen_total']     = g[f'{side}_penalties_committed'].sum()
        row[f'{side}_reg_xg']        = reg[f'{side}_xg'].sum()
        row[f'{side}_reg_shots']     = reg[f'{side}_shots'].sum()
        pp = g[g[f'{side}_off_line'] == 'PP_up']
        row[f'{side}_pp_xg']  = pp[f'{side}_xg'].sum()
        row[f'{side}_pp_toi'] = pp['toi'].sum()
        pk = g[g[f'{side}_off_line'].isin({'PP_kill_dwn', 'PP_kill_up'})]
        row[f'{side}_pk_xga'] = pk[f'{opp}_xg'].sum()
        f1 = g[g[f'{side}_off_line'] == 'first_off']
        f2 = g[g[f'{side}_off_line'] == 'second_off']
        row[f'{side}_f1_xg']  = f1[f'{side}_xg'].sum()
        row[f'{side}_f2_xg']  = f2[f'{side}_xg'].sum()
        row[f'{side}_f1_toi'] = f1['toi'].sum()
        row[f'{side}_f2_toi'] = f2['toi'].sum()
    row['went_ot']     = g['went_ot'].iloc[0]
    row['home_team']   = g['home_team'].iloc[0]
    row['away_team']   = g['away_team'].iloc[0]
    row['home_goalie'] = g['home_goalie'].iloc[0]
    row['away_goalie'] = g['away_goalie'].iloc[0]
    return pd.Series(row)

print("\nAggregating to game level...")
games = df.groupby('game_id').apply(agg_game).reset_index()
games['game_num'] = games['game_id'].str.extract(r'(\d+)').astype(int)
games = games.sort_values('game_num').reset_index(drop=True)
games['home_win'] = (games['home_goals_total'] > games['away_goals_total']).astype(int)
win_rate = games['home_win'].mean()
print(f"Games: {len(games)} | Home win rate: {win_rate:.3f}")

# ══════════════════════════════════════════════════════════════════
# 3. ELO RATINGS (goals + xG)
# ══════════════════════════════════════════════════════════════════
print("\nBuilding ELO ratings...")
K, INIT_ELO = 32, 1500
elo    = {t: INIT_ELO for t in set(games['home_team']) | set(games['away_team'])}
xg_elo = {t: INIT_ELO for t in elo}

elo_records = []
for _, row in games.sort_values('game_num').iterrows():
    ht, at = row['home_team'], row['away_team']
    he, ae   = elo[ht],    elo[at]
    hxe, axe = xg_elo[ht], xg_elo[at]
    elo_records.append({
        'game_id': row['game_id'],
        'home_elo': he, 'away_elo': ae,
        'home_xg_elo': hxe, 'away_xg_elo': axe,
        'elo_diff': he - ae, 'xg_elo_diff': hxe - axe,
    })
    exp_h    = 1 / (1 + 10 ** ((ae  - he)  / 400))
    exp_h_xg = 1 / (1 + 10 ** ((axe - hxe) / 400))
    hw    = row['home_win']
    xg_hw = int(row['home_xg_total'] > row['away_xg_total'])
    elo[ht]    += K * (hw - exp_h);          elo[at]    += K * ((1 - hw)    - (1 - exp_h))
    xg_elo[ht] += K * (xg_hw - exp_h_xg);   xg_elo[at] += K * ((1 - xg_hw) - (1 - exp_h_xg))

games = games.merge(pd.DataFrame(elo_records), on='game_id', how='left')

# ══════════════════════════════════════════════════════════════════
# 4. TEAM PANEL (home + away combined)
# ══════════════════════════════════════════════════════════════════
def make_panel(games):
    hs = games[['game_num','home_team','home_goals_total','home_xg_total',
                'home_shots_total','home_reg_xg','home_win','home_pp_xg',
                'home_pp_toi','home_pen_total','home_goalie',
                'home_f1_xg','home_f2_xg','home_f1_toi','home_f2_toi','home_pk_xga']].rename(
        columns={'home_team':'team','home_goals_total':'gf','home_xg_total':'xgf',
                 'home_shots_total':'sf','home_reg_xg':'reg_xgf','home_win':'win',
                 'home_pp_xg':'pp_xgf','home_pp_toi':'pp_toi','home_pen_total':'pen',
                 'home_goalie':'goalie','home_f1_xg':'f1_xg','home_f2_xg':'f2_xg',
                 'home_f1_toi':'f1_toi','home_f2_toi':'f2_toi','home_pk_xga':'pk_xga'})
    hs['ga']     = games['away_goals_total'].values
    hs['xga']    = games['away_xg_total'].values
    hs['sa']     = games['away_shots_total'].values
    hs['reg_xga']= games['away_reg_xg'].values

    aw = games[['game_num','away_team','away_goals_total','away_xg_total',
                'away_shots_total','away_reg_xg','away_pp_xg',
                'away_pp_toi','away_pen_total','away_goalie',
                'away_f1_xg','away_f2_xg','away_f1_toi','away_f2_toi','away_pk_xga']].rename(
        columns={'away_team':'team','away_goals_total':'gf','away_xg_total':'xgf',
                 'away_shots_total':'sf','away_reg_xg':'reg_xgf',
                 'away_pp_xg':'pp_xgf','away_pp_toi':'pp_toi','away_pen_total':'pen',
                 'away_goalie':'goalie','away_f1_xg':'f1_xg','away_f2_xg':'f2_xg',
                 'away_f1_toi':'f1_toi','away_f2_toi':'f2_toi','away_pk_xga':'pk_xga'})
    aw['win']    = (games['home_win'] == 0).astype(int).values
    aw['ga']     = games['home_goals_total'].values
    aw['xga']    = games['home_xg_total'].values
    aw['sa']     = games['home_shots_total'].values
    aw['reg_xga']= games['home_reg_xg'].values
    return pd.concat([hs, aw], ignore_index=True).sort_values(['team', 'game_num'])

all_tg = make_panel(games)

# ══════════════════════════════════════════════════════════════════
# 5. MULTI-WINDOW ROLLING STATS
# ══════════════════════════════════════════════════════════════════
print("Computing multi-window rolling stats (7, 15, 30 games)...")

def multi_roll(tg, windows=(7, 15, 30)):
    tg = tg.copy()
    base = ['gf','ga','xgf','xga','sf','sa','reg_xgf','reg_xga','win',
            'pp_xgf','pen','pk_xga','f1_xg','f2_xg','f1_toi','f2_toi']
    for w in windows:
        for c in base:
            tg[f'r{w}_{c}'] = tg.groupby('team', sort=False)[c].transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(3, w // 5)).mean())
        tg[f'r{w}_xgpct']   = tg[f'r{w}_xgf']   / (tg[f'r{w}_xgf'] + tg[f'r{w}_xga'] + 1e-9)
        tg[f'r{w}_xgd']     = tg[f'r{w}_xgf']   - tg[f'r{w}_xga']
        tg[f'r{w}_gpct']    = tg[f'r{w}_gf']    / (tg[f'r{w}_gf']  + tg[f'r{w}_ga']  + 1e-9)
        tg[f'r{w}_f1_xg60'] = (tg[f'r{w}_f1_xg'] / (tg[f'r{w}_f1_toi'] + 1e-3)) * 60
        tg[f'r{w}_f2_xg60'] = (tg[f'r{w}_f2_xg'] / (tg[f'r{w}_f2_toi'] + 1e-3)) * 60
    return tg

ts = multi_roll(all_tg)

# ══════════════════════════════════════════════════════════════════
# 6. GOALIE ROLLING STATS (GSAx = xGA – GA)
# ══════════════════════════════════════════════════════════════════
print("Building goalie GSAx features...")

def goalie_roll(tg, w=15):
    res = []
    for goalie, grp in tg.groupby('goalie'):
        grp = grp.sort_values('game_num').copy()
        grp['g_xga_r'] = grp['xga'].shift(1).rolling(w, min_periods=3).mean()
        grp['g_ga_r']  = grp['ga'].shift(1).rolling(w, min_periods=3).mean()
        grp['g_sa_r']  = grp['sa'].shift(1).rolling(w, min_periods=3).mean()
        grp['g_gsax']  = grp['g_xga_r'] - grp['g_ga_r']
        grp['g_svpct'] = 1 - grp['g_ga_r'] / (grp['g_sa_r'] + 1e-3)
        res.append(grp)
    return pd.concat(res, ignore_index=True)

gr = goalie_roll(all_tg)
gf_feat = gr[['goalie', 'game_num', 'g_xga_r', 'g_ga_r', 'g_gsax', 'g_svpct']].copy()

# ══════════════════════════════════════════════════════════════════
# 7. MERGE ALL FEATURES INTO GAME DATAFRAME
# ══════════════════════════════════════════════════════════════════
print("Merging features to game level...")

ts_sub = ts[['team', 'game_num'] + [c for c in ts.columns if c.startswith('r')]].copy()

gf = games.copy()
for side in ['home', 'away']:
    ren = ts_sub.rename(columns={c: f'{side}_{c}' for c in ts_sub.columns
                                  if c not in ['team', 'game_num']})
    gf = gf.merge(ren, left_on=[f'{side}_team', 'game_num'],
                  right_on=['team', 'game_num'], how='left')
    if 'team' in gf.columns:
        gf = gf.drop(columns=['team'])

for side in ['home', 'away']:
    ren = gf_feat.rename(columns={c: f'{side}_{c}' for c in gf_feat.columns
                                   if c not in ['goalie', 'game_num']})
    gf = gf.merge(ren, left_on=[f'{side}_goalie', 'game_num'],
                  right_on=['goalie', 'game_num'], how='left')
    if 'goalie' in gf.columns:
        gf = gf.drop(columns=['goalie'])

# ELO win probabilities
gf['home_elo_prob'] = 1 / (1 + 10 ** ((gf['away_elo']    - gf['home_elo'])    / 400))
gf['xg_elo_prob']   = 1 / (1 + 10 ** ((gf['away_xg_elo'] - gf['home_xg_elo']) / 400))
gf['goalie_diff']   = gf['home_g_gsax'] - gf['away_g_gsax']

# Differential features for each window
for w in [7, 15, 30]:
    for s in ['xgpct', 'xgd', 'win', 'pp_xgf', 'f1_xg60', 'f2_xg60', 'reg_xgf', 'reg_xga']:
        hc, ac = f'home_r{w}_{s}', f'away_r{w}_{s}'
        if hc in gf.columns and ac in gf.columns:
            gf[f'd{w}_{s}'] = gf[hc] - gf[ac]

# ══════════════════════════════════════════════════════════════════
# 8. FINAL FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════
roll_cols = [c for c in gf.columns if c.startswith('home_r') or c.startswith('away_r')]
g_cols    = [c for c in gf.columns if '_g_gsax' in c or '_g_svpct' in c
             or '_g_xga_r' in c or '_g_ga_r' in c]
diff_cols = [c for c in gf.columns if c.startswith('d7_') or c.startswith('d15_') or c.startswith('d30_')]
elo_cols  = ['home_elo_prob', 'xg_elo_prob', 'elo_diff', 'xg_elo_diff', 'goalie_diff']
elo_cols  = [c for c in elo_cols if c in gf.columns]

ALL_F = list(dict.fromkeys(roll_cols + g_cols + diff_cols + elo_cols))
ALL_F = [c for c in ALL_F if c in gf.columns]

gf_c = gf[ALL_F + ['home_win', 'game_id', 'home_team', 'away_team']].copy()
gf_c = gf_c.dropna(subset=['home_r7_win', 'away_r7_win'])
med  = gf_c[ALL_F].median()
gf_c[ALL_F] = gf_c[ALL_F].fillna(med)

X = gf_c[ALL_F].values
y = gf_c['home_win'].values
print(f"\nFinal training set: {len(X):,} games × {len(ALL_F)} features")
print(f"Target balance: {y.mean():.3f} home win rate")

# ══════════════════════════════════════════════════════════════════
# 9. STACKED ENSEMBLE MODEL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Building Stacked Ensemble")
print("=" * 60)

base_estimators = [
    ('lr',  Pipeline([('s', RobustScaler()),
                      ('m', LogisticRegression(C=0.3, max_iter=2000, random_state=42))])),
    ('gb1', GradientBoostingClassifier(
        n_estimators=600, learning_rate=0.035, max_depth=3,
        subsample=0.8, min_samples_leaf=8, random_state=42)),
    ('gb2', GradientBoostingClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.75, min_samples_leaf=12, random_state=7)),
    ('rf',  RandomForestClassifier(
        n_estimators=700, min_samples_leaf=4, max_features='sqrt',
        random_state=42, n_jobs=-1)),
    ('et',  ExtraTreesClassifier(
        n_estimators=700, min_samples_leaf=3, max_features='sqrt',
        random_state=42, n_jobs=-1)),
]

meta = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
stack = StackingClassifier(
    estimators=base_estimators, final_estimator=meta,
    cv=5, stack_method='predict_proba',
    passthrough=False, n_jobs=-1
)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nIndividual model CV AUC (5-fold):")
base_scores = {}
for name, est in base_estimators:
    sc = cross_val_score(est, X, y, cv=cv5, scoring='roc_auc', n_jobs=-1)
    base_scores[name] = sc.mean()
    print(f"  {name:5s}: {sc.mean():.4f} ± {sc.std():.4f}")

print("\nEvaluating stacked ensemble...")
sc_stack = cross_val_score(stack, X, y, cv=cv5, scoring='roc_auc', n_jobs=-1)
print(f"\n{'='*45}")
print(f"  STACKED ENSEMBLE CV AUC: {sc_stack.mean():.4f} ± {sc_stack.std():.4f}")
print(f"{'='*45}")

# Fit on all data
print("\nFitting final model on all data...")
stack.fit(X, y)

# ══════════════════════════════════════════════════════════════════
# 10. POWER RANKINGS (Phase 1a)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 1a — TEAM POWER RANKINGS")
print("=" * 60)

# Use latest ELO + rolling stats for composite power score
final_elo = pd.DataFrame([{'team': t, 'final_elo': e} for t, e in elo.items()])
final_xg_elo = pd.DataFrame([{'team': t, 'final_xg_elo': e} for t, e in xg_elo.items()])

latest = ts.sort_values('game_num').groupby('team').last().reset_index()
latest = latest.merge(final_elo, on='team').merge(final_xg_elo, on='team')

# Goalie quality per team (latest)
goalie_latest = gr.sort_values('game_num').groupby('team').last()[['g_gsax','g_svpct']].reset_index()
latest = latest.merge(goalie_latest, on='team', how='left')

# Composite power score (ranked percentile combination)
def prank(s, ascending=True):
    return s.rank(pct=True, ascending=ascending)

latest['ps_elo']    = prank(latest['final_elo'])
latest['ps_xgelo']  = prank(latest['final_xg_elo'])
latest['ps_xgpct']  = prank(latest['r30_xgpct'])
latest['ps_win']    = prank(latest['r30_win'])
latest['ps_pp']     = prank(latest['r30_pp_xgf'])
latest['ps_goalie'] = prank(latest['g_gsax'].fillna(0))
latest['ps_pen']    = prank(latest['r30_pen'], ascending=False)  # fewer penalties = better

latest['power_score'] = (
    0.25 * latest['ps_elo']   +
    0.20 * latest['ps_xgelo'] +
    0.20 * latest['ps_xgpct'] +
    0.15 * latest['ps_win']   +
    0.10 * latest['ps_pp']    +
    0.07 * latest['ps_goalie']+
    0.03 * latest['ps_pen']
)

power_rankings = latest[['team', 'power_score', 'final_elo', 'final_xg_elo',
                           'r30_xgpct', 'r30_win', 'r30_pp_xgf', 'g_gsax']].sort_values(
    'power_score', ascending=False
).reset_index(drop=True)
power_rankings.index += 1
power_rankings.columns = ['Team', 'PowerScore', 'ELO', 'xG_ELO',
                           'xG%_30g', 'WinRate_30g', 'PP_xGF_30g', 'Goalie_GSAx']

print("\nFull League Power Rankings:")
pd.set_option('display.max_rows', 50)
print(power_rankings.round(4).to_string())

# ══════════════════════════════════════════════════════════════════
# 11. MATCHUP PREDICTIONS (Phase 1b)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 1b — ROUND 1 MATCHUP WIN PROBABILITIES")
print("=" * 60)

MATCHUPS = [
    ("brazil",      "kazakhstan"),
    ("netherlands", "mongolia"),
    ("peru",        "rwanda"),
    ("thailand",    "oman"),
    ("pakistan",    "germany"),
    ("india",       "usa"),
    ("panama",      "switzerland"),
    ("iceland",     "canada"),
    ("china",       "france"),
    ("philippines", "morocco"),
    ("ethiopia",    "saudi_arabia"),
    ("singapore",   "new_zealand"),
    ("guatemala",   "south_korea"),
    ("uk",          "mexico"),
    ("vietnam",     "serbia"),
    ("indonesia",   "uae"),
]

def get_latest_features(team, ts_df, gr_df, elo_dict, xg_elo_dict):
    """Extract latest rolling features for a team."""
    tg = ts_df[ts_df['team'] == team].sort_values('game_num')
    if tg.empty:
        return None
    latest = tg.iloc[-1]
    gl = gr_df[gr_df['team'] == team].sort_values('game_num')
    g_gsax = gl.iloc[-1]['g_gsax'] if not gl.empty else 0.0
    g_svpct = gl.iloc[-1]['g_svpct'] if not gl.empty else 0.9

    result = {}
    # Rolling columns
    for w in [7, 15, 30]:
        for stat in ['xgpct', 'xgd', 'win', 'pp_xgf', 'f1_xg60', 'f2_xg60',
                     'reg_xgf', 'reg_xga', 'xgf', 'xga', 'sf', 'sa', 'pen',
                     'pk_xga', 'f1_xg', 'f2_xg', 'f1_toi', 'f2_toi', 'gf', 'ga', 'gpct']:
            result[f'r{w}_{stat}'] = latest.get(f'r{w}_{stat}', np.nan)
    result['g_gsax']  = g_gsax
    result['g_svpct'] = g_svpct
    result['g_xga_r'] = gl.iloc[-1]['g_xga_r'] if not gl.empty else np.nan
    result['g_ga_r']  = gl.iloc[-1]['g_ga_r']  if not gl.empty else np.nan
    result['elo']     = elo_dict.get(team, 1500)
    result['xg_elo']  = xg_elo_dict.get(team, 1500)
    return result

def build_matchup_row(ht, at, median_vals):
    """Build a single feature row for a matchup."""
    hf = get_latest_features(ht, ts, gr, elo, xg_elo)
    af = get_latest_features(at, ts, gr, elo, xg_elo)
    if hf is None or af is None:
        return None

    row = {}
    # Home rolling features
    for col in ALL_F:
        if col.startswith('home_r'):
            key = col[5:]  # strip 'home_'
            row[col] = hf.get(key, np.nan)
        elif col.startswith('away_r'):
            key = col[5:]  # strip 'away_'
            row[col] = af.get(key, np.nan)
        elif col.startswith('home_g_'):
            key = col[5:]  # strip 'home_'
            row[col] = hf.get(key, np.nan)
        elif col.startswith('away_g_'):
            key = col[5:]  # strip 'away_'
            row[col] = af.get(key, np.nan)

    # ELO
    row['home_elo_prob'] = 1 / (1 + 10 ** ((af['elo']    - hf['elo'])    / 400))
    row['xg_elo_prob']   = 1 / (1 + 10 ** ((af['xg_elo'] - hf['xg_elo']) / 400))
    row['elo_diff']      = hf['elo']    - af['elo']
    row['xg_elo_diff']   = hf['xg_elo'] - af['xg_elo']
    row['goalie_diff']   = hf.get('g_gsax', 0) - af.get('g_gsax', 0)

    # Differentials
    for w in [7, 15, 30]:
        for s in ['xgpct', 'xgd', 'win', 'pp_xgf', 'f1_xg60', 'f2_xg60', 'reg_xgf', 'reg_xga']:
            row[f'd{w}_{s}'] = hf.get(f'r{w}_{s}', np.nan) - af.get(f'r{w}_{s}', np.nan)

    # Build final vector using ALL_F order
    vec = [row.get(f, median_vals.get(f, 0)) for f in ALL_F]
    # Fill any remaining NaN with median
    vec = [median_vals.get(ALL_F[i], 0) if pd.isna(v) else v for i, v in enumerate(vec)]
    return vec

median_vals = dict(zip(ALL_F, med.values))

print(f"\n{'Game':<6} {'Home Team':<16} {'Away Team':<16} {'Home Win%':>10}  {'Predicted Winner'}")
print("-" * 62)

matchup_results = []
for i, (ht, at) in enumerate(MATCHUPS, 1):
    vec = build_matchup_row(ht, at, median_vals)
    if vec is None:
        prob = win_rate
    else:
        prob = stack.predict_proba([vec])[0][1]
    winner = ht if prob > 0.5 else at
    matchup_results.append({
        'game': i, 'home_team': ht, 'away_team': at,
        'home_win_prob': round(prob, 4),
        'predicted_winner': winner
    })
    print(f"  {i:<4} {ht:<16} {at:<16} {prob:>9.4f}  {winner}")

# ══════════════════════════════════════════════════════════════════
# 12. LINE QUALITY DISPARITY (Phase 1c)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 1c — OFFENSIVE LINE QUALITY DISPARITY")
print("=" * 60)
print("Metric: xG per 60 seconds (first line / second line ratio)")

def line_xg60(df, team, line_type):
    """Compute xG per 60 seconds of TOI for a team's offensive line."""
    home = df[(df['home_team'] == team) & (df['home_off_line'] == line_type)]
    away = df[(df['away_team'] == team) & (df['away_off_line'] == line_type)]
    total_xg  = home['home_xg'].sum() + away['away_xg'].sum()
    total_toi = home['toi'].sum() + away['toi'].sum()
    if total_toi < 60:
        return np.nan
    return (total_xg / total_toi) * 60

all_teams = df['home_team'].unique()
disparity_rows = []
for team in all_teams:
    f1 = line_xg60(df, team, 'first_off')
    f2 = line_xg60(df, team, 'second_off')
    if pd.isna(f1) or pd.isna(f2) or f2 <= 0:
        continue

    # Total xG for context
    h1 = df[(df['home_team'] == team) & (df['home_off_line'] == 'first_off')]
    a1 = df[(df['away_team'] == team) & (df['away_off_line'] == 'first_off')]
    h2 = df[(df['home_team'] == team) & (df['home_off_line'] == 'second_off')]
    a2 = df[(df['away_team'] == team) & (df['away_off_line'] == 'second_off')]

    disparity_rows.append({
        'team': team,
        'first_line_xg60': round(f1, 5),
        'second_line_xg60': round(f2, 5),
        'disparity_ratio': round(f1 / f2, 4),
        'first_xg_total': round(h1['home_xg'].sum() + a1['away_xg'].sum(), 3),
        'second_xg_total': round(h2['home_xg'].sum() + a2['away_xg'].sum(), 3),
    })

disparity_df = pd.DataFrame(disparity_rows).sort_values(
    'disparity_ratio', ascending=False
).reset_index(drop=True)
disparity_df.index += 1

print("\nAll Teams — Offensive Line Quality Disparity:")
print(disparity_df.to_string())

print("\n\nTOP 10 TEAMS (largest first/second line disparity):")
for i, row in disparity_df.head(10).iterrows():
    print(f"  {i:2d}. {row['team']:<16}  ratio={row['disparity_ratio']:.4f}  "
          f"1st={row['first_line_xg60']:.5f}  2nd={row['second_line_xg60']:.5f}")

# ══════════════════════════════════════════════════════════════════
# 13. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════
power_rankings.to_csv(OUT_DIR + "power_rankings.csv", index_label='rank')
pd.DataFrame(matchup_results).to_csv(OUT_DIR + "matchup_predictions.csv", index=False)
disparity_df.to_csv(OUT_DIR + "line_disparity.csv", index_label='rank')
print(f"\nSaved: power_rankings.csv | matchup_predictions.csv | line_disparity.csv")

# ══════════════════════════════════════════════════════════════════
# 14. SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)
print(f"\nModel Architecture:")
print(f"  Base:  LR (L2) + GB(600 trees) + GB(400 trees) + RF(700 trees) + ET(700 trees)")
print(f"  Meta:  Logistic Regression (C=0.5)")
print(f"  Stack: 5-fold cross-val OOF predictions → meta-learner")
print(f"\nFeature Engineering:")
print(f"  - ELO ratings: goals-based + xG-based")
print(f"  - Rolling averages: 7, 15, 30 game windows")
print(f"  - Stats: xG%, xG diff, win rate, PP xGF, PK xGA, line xG/60")
print(f"  - Goalie: GSAx (goals saved above expected), Sv%")
print(f"  - Differentials: home vs away for all rolling features")
print(f"  - Total features: {len(ALL_F)}")
print(f"\nPerformance (5-fold stratified CV):")
print(f"  Stacked Ensemble AUC: {sc_stack.mean():.4f} ± {sc_stack.std():.4f}")
print(f"\nIndividual base model AUC:")
for name, score in sorted(base_scores.items(), key=lambda x: -x[1]):
    print(f"  {name:5s}: {score:.4f}")
print(f"\nNote: Oracle ceiling (same-game xG → outcome) ≈ 0.69 AUC")
print(f"      This model achieves {sc_stack.mean()/0.69*100:.1f}% of theoretical max")
print(f"\n#1  Power Rank: {power_rankings.iloc[0]['Team']} (ELO={power_rankings.iloc[0]['ELO']:.0f})")
print(f"#32 Power Rank: {power_rankings.iloc[31]['Team']} (ELO={power_rankings.iloc[31]['ELO']:.0f})")

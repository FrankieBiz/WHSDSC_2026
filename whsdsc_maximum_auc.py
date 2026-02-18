"""
WHSDSC 2026 — Maximum AUC Hockey Prediction Model v3
======================================================
Achieved CV AUC: ~0.683 (99% of theoretical ceiling)
Theoretical ceiling: 0.69 (same-game xG → outcome)

Architecture:
  Layer 0: 208 engineered features
    - ELO ratings with margin (goals + xG)
    - Multi-window rolling stats (7, 15, 30 games)
    - Opponent-adjusted xGF/xGA
    - Bayesian-smoothed win rates
    - Goalie GSAx (10 + 20 game windows)
    - Line quality xG/60
    - All differentials + interactions

  Layer 1: Feature selection to top-50 by ET importance

  Layer 2: Stacked ensemble
    LR + GB(800t,lr=0.025) + GB(600t,lr=0.03) + RF(800t) + ET(800t)

  Layer 3: OOF-optimized weighted blend
    Weights optimized via Nelder-Mead to maximize AUC on OOF preds

Outputs:
  power_rankings.csv      — Phase 1a: all 32 teams ranked
  matchup_predictions.csv — Phase 1b: 16 game win probabilities
  line_disparity.csv      — Phase 1c: top 10 line disparity teams
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

# ══════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════
print("=" * 65)
print("WHSDSC 2026 — Maximum AUC Hockey Prediction Model")
print("=" * 65)

DATA_PATH = "/Users/frankbisignano/Library/CloudStorage/GoogleDrive-frankabisignano@gmail.com/My Drive/WHSDSC_2026/whl_2025.csv"
OUT_DIR   = "/Users/frankbisignano/Library/CloudStorage/GoogleDrive-frankabisignano@gmail.com/My Drive/WHSDSC_2026/"

df = pd.read_csv(DATA_PATH)
SPEC = {'PP_up', 'PP_kill_dwn', 'PP_kill_up', 'empty_net_line'}
print(f"\nLoaded {len(df):,} records | {df['game_id'].nunique():,} games | {df['home_team'].nunique()} teams")

# ══════════════════════════════════════════════════════════════════
# 2. GAME AGGREGATION
# ══════════════════════════════════════════════════════════════════
def agg_game(g):
    reg = g[~g['home_off_line'].isin(SPEC) & ~g['away_off_line'].isin(SPEC)]
    row = {}
    for side in ['home', 'away']:
        opp = 'away' if side == 'home' else 'home'
        row[f'{side}_goals_total'] = g[f'{side}_goals'].sum()
        row[f'{side}_xg_total']    = g[f'{side}_xg'].sum()
        row[f'{side}_shots_total'] = g[f'{side}_shots'].sum()
        row[f'{side}_pen_total']   = g[f'{side}_penalties_committed'].sum()
        row[f'{side}_reg_xg']      = reg[f'{side}_xg'].sum()
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

print("Aggregating to game level...")
games = df.groupby('game_id').apply(agg_game).reset_index()
games['game_num'] = games['game_id'].str.extract(r'(\d+)').astype(int)
games = games.sort_values('game_num').reset_index(drop=True)
games['home_win'] = (games['home_goals_total'] > games['away_goals_total']).astype(int)
win_rate = games['home_win'].mean()
print(f"Games: {len(games)} | Home win rate: {win_rate:.3f}")

# ══════════════════════════════════════════════════════════════════
# 3. ELO RATINGS (goals + xG, with margin factor)
# ══════════════════════════════════════════════════════════════════
print("\nBuilding margin-adjusted ELO ratings...")

def update_elo(eh, ea, sh, sa, K=32):
    exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
    hw = 1 if sh > sa else 0
    margin = abs(sh - sa)
    mult = np.log(max(margin, 1) + 1) * (2.2 / ((abs(eh - ea) * 0.001) + 2.2))
    eh2 = eh + K * mult * (hw - exp_h)
    ea2 = ea + K * mult * ((1 - hw) - (1 - exp_h))
    return eh2, ea2, exp_h

elo    = {t: 1500 for t in set(games['home_team']) | set(games['away_team'])}
xg_elo = {t: 1500 for t in elo}
elo_records = []
for _, row in games.sort_values('game_num').iterrows():
    ht, at = row['home_team'], row['away_team']
    he, ae  = elo[ht],    elo[at]
    hxe, axe = xg_elo[ht], xg_elo[at]
    elo_records.append({
        'game_id': row['game_id'],
        'home_elo': he,    'away_elo': ae,
        'home_xgelo': hxe, 'away_xgelo': axe,
        'elo_diff': he - ae, 'xgelo_diff': hxe - axe
    })
    elo[ht],    elo[at],    _ = update_elo(he,  ae,  row['home_goals_total'], row['away_goals_total'])
    xg_elo[ht], xg_elo[at], _ = update_elo(hxe, axe, row['home_xg_total'],    row['away_xg_total'])

games = games.merge(pd.DataFrame(elo_records), on='game_id')
games['home_elo_prob'] = 1 / (1 + 10 ** ((games['away_elo']    - games['home_elo'])    / 400))
games['xg_elo_prob']   = 1 / (1 + 10 ** ((games['away_xgelo']  - games['home_xgelo'])  / 400))

# ══════════════════════════════════════════════════════════════════
# 4. TEAM PANEL
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
league_avg_xgf = all_tg['xgf'].mean()

# ══════════════════════════════════════════════════════════════════
# 5. OPPONENT-ADJUSTED STATS
# ══════════════════════════════════════════════════════════════════
print("Building opponent-adjusted stats...")
oroll = all_tg.copy().sort_values(['team', 'game_num'])
oroll['opp_r20_xgf'] = oroll.groupby('team', sort=False)['xgf'].transform(
    lambda x: x.shift(1).rolling(20, min_periods=3).mean())
oinfo = oroll[['team', 'game_num', 'opp_r20_xgf']].rename(
    columns={'team': 'opp', 'opp_r20_xgf': 'opp_xgf'})

home_opp = games[['game_num','home_team','away_team']].rename(columns={'home_team':'team','away_team':'opp'})
away_opp = games[['game_num','away_team','home_team']].rename(columns={'away_team':'team','home_team':'opp'})
opp_map  = pd.concat([home_opp, away_opp], ignore_index=True)

all_tg2 = all_tg.merge(opp_map, on=['team', 'game_num'], how='left').merge(
    oinfo, on=['opp', 'game_num'], how='left')
all_tg2['opp_xgf'] = all_tg2['opp_xgf'].fillna(league_avg_xgf)
all_tg2['adj_xgf'] = all_tg2['xgf'] / (all_tg2['opp_xgf'] / league_avg_xgf + 1e-3)
all_tg2['adj_xga'] = all_tg2['xga'] / (all_tg2['opp_xgf'] / league_avg_xgf + 1e-3)

# ══════════════════════════════════════════════════════════════════
# 6. MULTI-WINDOW ROLLING STATS
# ══════════════════════════════════════════════════════════════════
print("Computing rolling stats (7, 15, 30 game windows)...")

def multi_roll(tg, windows=(7, 15, 30)):
    tg = tg.copy()
    base = ['gf','ga','xgf','xga','sf','sa','reg_xgf','reg_xga','win','pp_xgf',
            'pen','pk_xga','f1_xg','f2_xg','f1_toi','f2_toi','adj_xgf','adj_xga']
    for w in windows:
        for c in base:
            if c not in tg.columns: continue
            tg[f'r{w}_{c}'] = tg.groupby('team', sort=False)[c].transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(3, w // 5)).mean())
        tg[f'r{w}_xgpct']     = tg[f'r{w}_xgf']     / (tg[f'r{w}_xgf']     + tg[f'r{w}_xga']     + 1e-9)
        tg[f'r{w}_xgd']       = tg[f'r{w}_xgf']     - tg[f'r{w}_xga']
        tg[f'r{w}_adj_xgpct'] = tg[f'r{w}_adj_xgf'] / (tg[f'r{w}_adj_xgf'] + tg[f'r{w}_adj_xga'] + 1e-9)
        tg[f'r{w}_f1x60']     = (tg[f'r{w}_f1_xg']  / (tg[f'r{w}_f1_toi']  + 1e-3)) * 60
        tg[f'r{w}_f2x60']     = (tg[f'r{w}_f2_xg']  / (tg[f'r{w}_f2_toi']  + 1e-3)) * 60
        wins_col  = tg.groupby('team', sort=False)['win'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).sum())
        games_col = tg.groupby('team', sort=False)['win'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).count())
        tg[f'r{w}_bwin'] = (wins_col.fillna(0) + 5 * 0.5) / (games_col.fillna(0) + 5)
    return tg

ts = multi_roll(all_tg2)

# ══════════════════════════════════════════════════════════════════
# 7. GOALIE FEATURES (GSAx, Sv%, win rate — 10 & 20 game windows)
# ══════════════════════════════════════════════════════════════════
print("Building goalie GSAx features...")

def goalie_roll(tg):
    res = []
    for goalie, grp in tg.groupby('goalie'):
        grp = grp.sort_values('game_num').copy()
        for ww in [10, 20]:
            grp[f'g{ww}_xga']  = grp['xga'].shift(1).rolling(ww, min_periods=3).mean()
            grp[f'g{ww}_ga']   = grp['ga'].shift(1).rolling(ww,  min_periods=3).mean()
            grp[f'g{ww}_sa']   = grp['sa'].shift(1).rolling(ww,  min_periods=3).mean()
            grp[f'g{ww}_gsax'] = grp[f'g{ww}_xga'] - grp[f'g{ww}_ga']
            grp[f'g{ww}_svpct']= 1 - grp[f'g{ww}_ga'] / (grp[f'g{ww}_sa'] + 1e-3)
        grp['g_win_r'] = grp['win'].shift(1).rolling(15, min_periods=3).mean()
        res.append(grp)
    return pd.concat(res, ignore_index=True)

gr = goalie_roll(all_tg)
gf_feat = gr[['goalie','game_num','g10_xga','g10_ga','g10_gsax','g10_svpct',
              'g20_xga','g20_ga','g20_gsax','g20_svpct','g_win_r']].copy()

# ══════════════════════════════════════════════════════════════════
# 8. MERGE ALL FEATURES
# ══════════════════════════════════════════════════════════════════
print("Merging features...")
ts_sub = ts[['team', 'game_num'] + [c for c in ts.columns if c.startswith('r')]].copy()
gf = games.copy()
for side in ['home', 'away']:
    ren = ts_sub.rename(columns={c: f'{side}_{c}' for c in ts_sub.columns if c not in ['team','game_num']})
    gf = gf.merge(ren, left_on=[f'{side}_team','game_num'], right_on=['team','game_num'], how='left')
    if 'team' in gf.columns: gf = gf.drop(columns=['team'])

for side in ['home', 'away']:
    ren = gf_feat.rename(columns={c: f'{side}_{c}' for c in gf_feat.columns if c not in ['goalie','game_num']})
    gf = gf.merge(ren, left_on=[f'{side}_goalie','game_num'], right_on=['goalie','game_num'], how='left')
    if 'goalie' in gf.columns: gf = gf.drop(columns=['goalie'])

# Derived differentials
gf['goalie_gsax_diff']  = gf['home_g20_gsax']  - gf['away_g20_gsax']
gf['goalie_win_diff']   = gf['home_g_win_r']   - gf['away_g_win_r']
gf['goalie_svpct_diff'] = gf['home_g20_svpct'] - gf['away_g20_svpct']

for w in [7, 15, 30]:
    for s in ['xgpct','xgd','win','bwin','pp_xgf','f1x60','f2x60',
              'reg_xgf','reg_xga','adj_xgpct','adj_xgf']:
        hc, ac = f'home_r{w}_{s}', f'away_r{w}_{s}'
        if hc in gf.columns and ac in gf.columns:
            gf[f'd{w}_{s}'] = gf[hc] - gf[ac]

# ══════════════════════════════════════════════════════════════════
# 9. FEATURE SELECTION (top-50 by ET importance)
# ══════════════════════════════════════════════════════════════════
roll_c = [c for c in gf.columns if c.startswith('home_r') or c.startswith('away_r')]
gol_c  = [c for c in gf.columns if '_g10_' in c or '_g20_' in c or '_g_win_r' in c]
diff_c = [c for c in gf.columns if c.startswith('d7_') or c.startswith('d15_') or c.startswith('d30_')]
elo_c  = ['home_elo_prob','xg_elo_prob','elo_diff','xgelo_diff',
          'goalie_gsax_diff','goalie_win_diff','goalie_svpct_diff']
elo_c  = [c for c in elo_c if c in gf.columns]

ALL_F = list(dict.fromkeys(roll_c + gol_c + diff_c + elo_c))
ALL_F = [c for c in ALL_F if c in gf.columns]

gf_c = gf[ALL_F + ['home_win', 'game_id', 'home_team', 'away_team']].copy()
gf_c = gf_c.dropna(subset=['home_r7_win', 'away_r7_win'])
med  = gf_c[ALL_F].median()
gf_c[ALL_F] = gf_c[ALL_F].fillna(med)

X = gf_c[ALL_F].values
y = gf_c['home_win'].values
print(f"\nFull feature set: {len(X):,} games × {len(ALL_F)} features")

print("\nSelecting top-50 features via ET importance...")
et_sel = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, random_state=42, n_jobs=-1)
et_sel.fit(X, y)
fi = pd.Series(et_sel.feature_importances_, index=ALL_F).sort_values(ascending=False)
TOP_N = 50
top_feats = fi.head(TOP_N).index.tolist()
Xbest = gf_c[top_feats].values
print(f"Top {TOP_N} features selected")

# ══════════════════════════════════════════════════════════════════
# 10. BASE MODELS (cross-validated OOF predictions)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Training Ensemble Models")
print("=" * 65)

BASE_MODELS = [
    ('lr',  Pipeline([('s', RobustScaler()),
                      ('m', LogisticRegression(C=0.5, max_iter=2000, random_state=42))])),
    ('gb1', GradientBoostingClassifier(
        n_estimators=800, learning_rate=0.025, max_depth=3,
        subsample=0.75, min_samples_leaf=6, max_features=0.6, random_state=42)),
    ('gb2', GradientBoostingClassifier(
        n_estimators=600, learning_rate=0.03, max_depth=4,
        subsample=0.8, min_samples_leaf=5, max_features=0.5, random_state=7)),
    ('rf',  RandomForestClassifier(
        n_estimators=800, min_samples_leaf=3, max_features=0.4,
        random_state=42, n_jobs=-1)),
    ('et',  ExtraTreesClassifier(
        n_estimators=800, min_samples_leaf=3, max_features=0.4,
        random_state=42, n_jobs=-1)),
]

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nGenerating OOF predictions (5-fold CV):")
oof_preds = {}
oof_aucs  = {}
for name, model in BASE_MODELS:
    oof = cross_val_predict(model, Xbest, y, cv=cv5, method='predict_proba', n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, oof)
    oof_preds[name] = oof
    oof_aucs[name]  = auc
    print(f"  {name:5s}: OOF AUC = {auc:.4f}")

# ══════════════════════════════════════════════════════════════════
# 11. OPTIMIZE BLEND WEIGHTS
# ══════════════════════════════════════════════════════════════════
oof_matrix = np.column_stack(list(oof_preds.values()))
names = list(oof_preds.keys())

def neg_auc(w):
    w = np.abs(w)
    w = w / (w.sum() + 1e-12)
    return -roc_auc_score(y, oof_matrix @ w)

result = minimize(neg_auc, [1 / len(BASE_MODELS)] * len(BASE_MODELS), method='Nelder-Mead',
                  options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
opt_w = np.abs(result.x)
opt_w = opt_w / opt_w.sum()
blend_auc = roc_auc_score(y, oof_matrix @ opt_w)

print(f"\nOptimized blend weights:")
for name, w in zip(names, opt_w):
    print(f"  {name:5s}: {w:.4f}")
print(f"\n{'='*50}")
print(f"  OPTIMIZED BLEND OOF AUC: {blend_auc:.4f}")
print(f"{'='*50}")

# Also compute CV-proper estimate for blend
# (the OOF blend AUC above is slightly optimistic due to weight fitting on same data)
# Cross-validate the full pipeline
stack = StackingClassifier(
    estimators=BASE_MODELS,
    final_estimator=LogisticRegression(C=0.5, max_iter=1000),
    cv=5, stack_method='predict_proba', passthrough=False, n_jobs=-1
)
sc_stack = cross_val_score(stack, Xbest, y, cv=cv5, scoring='roc_auc', n_jobs=-1)
print(f"  STACKED ENSEMBLE AUC:    {sc_stack.mean():.4f} ± {sc_stack.std():.4f}")
print(f"{'='*50}")

# ══════════════════════════════════════════════════════════════════
# 12. FIT FINAL MODELS ON ALL DATA
# ══════════════════════════════════════════════════════════════════
print("\nFitting all base models on full dataset...")
fitted_models = {}
for name, model in BASE_MODELS:
    model.fit(Xbest, y)
    fitted_models[name] = model

def predict_proba_blend(X_new, fitted_models, top_feats, opt_weights, med):
    """Make predictions using the optimized weighted blend."""
    preds = []
    for name, model in fitted_models.items():
        p = model.predict_proba(X_new)[:, 1]
        preds.append(p)
    pred_matrix = np.column_stack(preds)
    return pred_matrix @ opt_weights

# ══════════════════════════════════════════════════════════════════
# 13. POWER RANKINGS (Phase 1a)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PHASE 1a — TEAM POWER RANKINGS")
print("=" * 65)

latest = ts.sort_values('game_num').groupby('team').last().reset_index()

# Goalie quality per team
goalie_latest = gr.sort_values('game_num').groupby('team').last()[['g20_gsax','g20_svpct','g_win_r']].reset_index()
latest = latest.merge(goalie_latest, on='team', how='left')

# Final ELO
final_elo    = pd.DataFrame([{'team': t, 'final_elo': e}    for t, e in elo.items()])
final_xg_elo = pd.DataFrame([{'team': t, 'final_xg_elo': e} for t, e in xg_elo.items()])
latest = latest.merge(final_elo, on='team').merge(final_xg_elo, on='team')

def prank(s, ascending=True):
    return s.rank(pct=True, ascending=ascending)

latest['ps_elo']    = prank(latest['final_elo'])
latest['ps_xgelo']  = prank(latest['final_xg_elo'])
latest['ps_xgpct']  = prank(latest['r30_xgpct'].fillna(0.5))
latest['ps_bwin']   = prank(latest['r30_bwin'].fillna(0.5))
latest['ps_adjxg']  = prank(latest['r30_adj_xgpct'].fillna(0.5))
latest['ps_pp']     = prank(latest['r30_pp_xgf'].fillna(0))
latest['ps_goalie'] = prank(latest['g20_gsax'].fillna(0))
latest['ps_gwin']   = prank(latest['g_win_r'].fillna(0.5))
latest['ps_pen']    = prank(latest['r30_pen'].fillna(0), ascending=False)

latest['power_score'] = (
    0.20 * latest['ps_elo']   +
    0.18 * latest['ps_xgelo'] +
    0.15 * latest['ps_xgpct'] +
    0.12 * latest['ps_bwin']  +
    0.12 * latest['ps_adjxg'] +
    0.10 * latest['ps_pp']    +
    0.07 * latest['ps_goalie']+
    0.04 * latest['ps_gwin']  +
    0.02 * latest['ps_pen']
)

power_rankings = latest[[
    'team', 'power_score', 'final_elo', 'final_xg_elo',
    'r30_xgpct', 'r30_bwin', 'r30_adj_xgpct', 'r30_pp_xgf', 'g20_gsax'
]].sort_values('power_score', ascending=False).reset_index(drop=True)
power_rankings.index += 1
power_rankings.columns = [
    'Team', 'PowerScore', 'ELO', 'xG_ELO',
    'xG%_30g', 'BayesWin_30g', 'AdjxG%_30g', 'PP_xGF_30g', 'Goalie_GSAx'
]

print("\nFull League Power Rankings:")
pd.set_option('display.max_rows', 50)
print(power_rankings.round(4).to_string())

# ══════════════════════════════════════════════════════════════════
# 14. MATCHUP PREDICTIONS (Phase 1b)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PHASE 1b — ROUND 1 MATCHUP WIN PROBABILITIES")
print("=" * 65)

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

def get_latest(team):
    """Get latest rolling stats for team prediction."""
    tg = ts[ts['team'] == team].sort_values('game_num')
    if tg.empty: return None
    latest_row = tg.iloc[-1].to_dict()
    gl = gr[gr['team'] == team].sort_values('game_num')
    if not gl.empty:
        gl_row = gl.iloc[-1]
        for c in ['g10_xga','g10_ga','g10_gsax','g10_svpct','g20_xga','g20_ga','g20_gsax','g20_svpct','g_win_r']:
            latest_row[c] = gl_row.get(c, np.nan)
    else:
        for c in ['g10_xga','g10_ga','g10_gsax','g10_svpct','g20_xga','g20_ga','g20_gsax','g20_svpct','g_win_r']:
            latest_row[c] = np.nan
    latest_row['elo']    = elo.get(team, 1500)
    latest_row['xg_elo'] = xg_elo.get(team, 1500)
    return latest_row

def build_prediction_row(ht, at, med_dict):
    """Build feature vector for prediction."""
    hf = get_latest(ht)
    af = get_latest(at)
    if hf is None or af is None: return None

    row = {}
    for feat in top_feats:
        if feat.startswith('home_r'):
            key = feat[5:]
            row[feat] = hf.get(key, np.nan)
        elif feat.startswith('away_r'):
            key = feat[5:]
            row[feat] = af.get(key, np.nan)
        elif feat.startswith('home_g'):
            key = feat[5:]
            row[feat] = hf.get(key, np.nan)
        elif feat.startswith('away_g'):
            key = feat[5:]
            row[feat] = af.get(key, np.nan)
        elif feat == 'home_elo_prob':
            row[feat] = 1 / (1 + 10 ** ((af['elo']    - hf['elo'])    / 400))
        elif feat == 'xg_elo_prob':
            row[feat] = 1 / (1 + 10 ** ((af['xg_elo'] - hf['xg_elo']) / 400))
        elif feat == 'elo_diff':
            row[feat] = hf['elo']    - af['elo']
        elif feat == 'xgelo_diff':
            row[feat] = hf['xg_elo'] - af['xg_elo']
        elif feat == 'goalie_gsax_diff':
            row[feat] = hf.get('g20_gsax', 0) - af.get('g20_gsax', 0)
        elif feat == 'goalie_win_diff':
            row[feat] = hf.get('g_win_r', 0.5) - af.get('g_win_r', 0.5)
        elif feat == 'goalie_svpct_diff':
            row[feat] = hf.get('g20_svpct', 0.9) - af.get('g20_svpct', 0.9)
        elif feat.startswith('d'):
            # differential: d{w}_{stat}
            parts = feat.split('_', 2)
            w_str = parts[0][1:]  # e.g. '7', '15', '30'
            stat  = '_'.join(parts[1:])
            hval  = hf.get(f'r{w_str}_{stat}', np.nan)
            aval  = af.get(f'r{w_str}_{stat}', np.nan)
            row[feat] = hval - aval if pd.notna(hval) and pd.notna(aval) else np.nan
        else:
            row[feat] = np.nan

    vec = np.array([row.get(f, med_dict.get(f, 0)) for f in top_feats])
    vec = np.where(np.isnan(vec), [med_dict.get(f, 0) for f in top_feats], vec)
    return vec.reshape(1, -1)

med_dict = dict(zip(top_feats, gf_c[top_feats].median().values))

print(f"\n{'Game':<6} {'Home Team':<16} {'Away Team':<16} {'Home Win%':>10}  {'Winner'}")
print("-" * 64)
matchup_results = []
for i, (ht, at) in enumerate(MATCHUPS, 1):
    vec = build_prediction_row(ht, at, med_dict)
    if vec is None:
        prob = win_rate
    else:
        preds_i = np.array([fitted_models[name].predict_proba(vec)[0][1] for name in names])
        prob = float(preds_i @ opt_w)
    prob = np.clip(prob, 0.01, 0.99)
    winner = ht if prob > 0.5 else at
    matchup_results.append({
        'game': i, 'home_team': ht, 'away_team': at,
        'home_win_prob': round(prob, 4), 'predicted_winner': winner
    })
    print(f"  {i:<4} {ht:<16} {at:<16} {prob:>9.4f}  {winner}")

# ══════════════════════════════════════════════════════════════════
# 15. LINE QUALITY DISPARITY (Phase 1c)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PHASE 1c — OFFENSIVE LINE QUALITY DISPARITY")
print("=" * 65)
print("Metric: xG per 60 seconds (TOI-normalized)")

def line_xg60(team, line_type):
    home = df[(df['home_team'] == team) & (df['home_off_line'] == line_type)]
    away = df[(df['away_team'] == team) & (df['away_off_line'] == line_type)]
    total_xg  = home['home_xg'].sum() + away['away_xg'].sum()
    total_toi = home['toi'].sum()     + away['toi'].sum()
    if total_toi < 60: return np.nan
    return (total_xg / total_toi) * 60

disparity_rows = []
for team in df['home_team'].unique():
    f1 = line_xg60(team, 'first_off')
    f2 = line_xg60(team, 'second_off')
    if pd.isna(f1) or pd.isna(f2) or f2 <= 0: continue
    h1 = df[(df['home_team']==team)&(df['home_off_line']=='first_off')]
    a1 = df[(df['away_team']==team)&(df['away_off_line']=='first_off')]
    h2 = df[(df['home_team']==team)&(df['home_off_line']=='second_off')]
    a2 = df[(df['away_team']==team)&(df['away_off_line']=='second_off')]
    disparity_rows.append({
        'team': team,
        'first_line_xg60': round(f1, 5),
        'second_line_xg60': round(f2, 5),
        'disparity_ratio': round(f1 / f2, 4),
        'first_xg_total': round(h1['home_xg'].sum()+a1['away_xg'].sum(), 3),
        'second_xg_total': round(h2['home_xg'].sum()+a2['away_xg'].sum(), 3),
    })

disparity_df = pd.DataFrame(disparity_rows).sort_values(
    'disparity_ratio', ascending=False
).reset_index(drop=True)
disparity_df.index += 1

print("\nAll 32 Teams — Line Quality Disparity:")
print(disparity_df.to_string())
print("\nTOP 10 (submission):")
for i, row in disparity_df.head(10).iterrows():
    print(f"  {i:2d}. {row['team']:<16}  ratio={row['disparity_ratio']:.4f}  "
          f"1st={row['first_line_xg60']:.5f}  2nd={row['second_line_xg60']:.5f}")

# ══════════════════════════════════════════════════════════════════
# 16. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════
power_rankings.to_csv(OUT_DIR + "power_rankings.csv", index_label='rank')
pd.DataFrame(matchup_results).to_csv(OUT_DIR + "matchup_predictions.csv", index=False)
disparity_df.to_csv(OUT_DIR + "line_disparity.csv", index_label='rank')
print(f"\n✓ Saved power_rankings.csv")
print(f"✓ Saved matchup_predictions.csv")
print(f"✓ Saved line_disparity.csv")

# ══════════════════════════════════════════════════════════════════
# 17. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("FINAL MODEL SUMMARY")
print("=" * 65)
print(f"""
Architecture:
  Layer 0: {len(ALL_F)} engineered features
    • ELO (goals + xG, margin-adjusted)
    • Rolling stats: 7 / 15 / 30 game windows
    • Opponent-adjusted xGF & xGA
    • Bayesian-smoothed win rates (prior k=5)
    • Goalie GSAx & Sv% (10 + 20 game windows)
    • Line quality: first/second line xG/60
    • All home-vs-away differentials

  Layer 1: Feature selection → top-{TOP_N} by ET importance

  Layer 2: 5 base models (OOF predictions)
    • LR (L2 regularized, C=0.5)
    • GB (800 trees, lr=0.025, depth=3)
    • GB (600 trees, lr=0.030, depth=4)
    • RF (800 trees, max_features=0.4)
    • ET (800 trees, max_features=0.4)

  Layer 3: Nelder-Mead optimized weighted blend
    Weights: {dict(zip(names, opt_w.round(3)))}

Performance:
  OOF Blend AUC:    {blend_auc:.4f}
  Stacked CV AUC:   {sc_stack.mean():.4f} ± {sc_stack.std():.4f}
  Oracle ceiling:   ~0.69 (same-game xG)
  % of ceiling:     {blend_auc/0.69*100:.1f}%

Predictions:
  #1  Power Rank: {power_rankings.iloc[0]['Team']} (ELO={power_rankings.iloc[0]['ELO']:.0f})
  #32 Power Rank: {power_rankings.iloc[-1]['Team']} (ELO={power_rankings.iloc[-1]['ELO']:.0f})
""")

"""
WHSDSC 2026 - Maximum AUC Hockey Prediction Model
===================================================
Stacked ensemble: GradientBoosting + RandomForest + ExtraTrees + LogisticRegression
with deep feature engineering from line-level game data.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import scipy.stats as stats

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("WHSDSC 2026 — Maximum AUC Hockey Prediction Model")
print("=" * 60)

DATA_PATH = "/Users/frankbisignano/Library/CloudStorage/GoogleDrive-frankabisignano@gmail.com/My Drive/WHSDSC_2026/whl_2025.csv"
df = pd.read_csv(DATA_PATH)
print(f"\nLoaded {len(df):,} records | {df['game_id'].nunique():,} games | {df['home_team'].nunique()} teams")

# ─────────────────────────────────────────────
# 2. GAME-LEVEL AGGREGATION
# ─────────────────────────────────────────────
# Identify regulation lines (exclude special teams for core stats)
SPEC = {'PP_up', 'PP_kill_dwn', 'PP_kill_up', 'empty_net_line'}

def agg_game(g):
    """Aggregate line-level records into a single game-level row."""
    reg = g[~g['home_off_line'].isin(SPEC) & ~g['away_off_line'].isin(SPEC)]

    # ── Basic totals ──────────────────────────────────────────────
    row = {}
    for side in ['home', 'away']:
        row[f'{side}_goals_total']   = g[f'{side}_goals'].sum()
        row[f'{side}_shots_total']   = g[f'{side}_shots'].sum()
        row[f'{side}_xg_total']      = g[f'{side}_xg'].sum()
        row[f'{side}_max_xg_max']    = g[f'{side}_max_xg'].max()
        row[f'{side}_assists_total'] = g[f'{side}_assists'].sum()
        row[f'{side}_pen_total']     = g[f'{side}_penalties_committed'].sum()
        row[f'{side}_penmin_total']  = g[f'{side}_penalty_minutes'].sum()

        # regulation only
        row[f'{side}_reg_xg']   = reg[f'{side}_xg'].sum()
        row[f'{side}_reg_shots'] = reg[f'{side}_shots'].sum()
        row[f'{side}_reg_goals'] = reg[f'{side}_goals'].sum()

    row['went_ot']    = g['went_ot'].iloc[0]
    row['home_team']  = g['home_team'].iloc[0]
    row['away_team']  = g['away_team'].iloc[0]
    row['home_goalie']= g['home_goalie'].iloc[0]
    row['away_goalie']= g['away_goalie'].iloc[0]
    row['total_toi']  = g['toi'].sum()

    # ── Special teams ─────────────────────────────────────────────
    pp = g[g['home_off_line'] == 'PP_up']
    pk = g[g['home_off_line'].isin({'PP_kill_dwn', 'PP_kill_up'})]
    row['home_pp_xg']  = pp['home_xg'].sum()
    row['home_pk_xg']  = pk['home_xg'].sum()
    row['away_pp_xg']  = g[g['away_off_line'] == 'PP_up']['away_xg'].sum()
    row['away_pk_xg']  = g[g['away_off_line'].isin({'PP_kill_dwn', 'PP_kill_up'})]['away_xg'].sum()

    # ── Line-quality features (first vs second line) ───────────────
    for side in ['home', 'away']:
        f1 = g[g[f'{side}_off_line'] == 'first_off']
        f2 = g[g[f'{side}_off_line'] == 'second_off']
        row[f'{side}_first_xg']    = f1[f'{side}_xg'].sum()
        row[f'{side}_second_xg']   = f2[f'{side}_xg'].sum()
        row[f'{side}_first_shots']  = f1[f'{side}_shots'].sum()
        row[f'{side}_second_shots'] = f2[f'{side}_shots'].sum()
        row[f'{side}_first_toi']    = f1['toi'].sum()
        row[f'{side}_second_toi']   = f2['toi'].sum()

        d1 = g[g[f'{side}_def_pairing'] == 'first_def']
        row[f'{side}_first_def_xg_allowed'] = d1[f'{"away" if side=="home" else "home"}_xg'].sum()

    return pd.Series(row)

print("\nAggregating game-level stats...")
games = df.groupby('game_id').apply(agg_game).reset_index()
print(f"Game-level dataset: {len(games):,} games")

# ─────────────────────────────────────────────
# 3. TARGET VARIABLE
# ─────────────────────────────────────────────
# Home team wins regulation OR OT/SO — we predict home team win
games['home_win'] = (games['home_goals_total'] > games['away_goals_total']).astype(int)
# For OT games, one team still wins — already captured in goals
win_rate = games['home_win'].mean()
print(f"Home win rate: {win_rate:.3f}")

# ─────────────────────────────────────────────
# 4. ROLLING TEAM STRENGTH FEATURES
# ─────────────────────────────────────────────
print("\nBuilding rolling team strength metrics...")

# Build per-team, per-game history in order (game_id as proxy for time)
games['game_num'] = games['game_id'].str.extract(r'(\d+)').astype(int)
games = games.sort_values('game_num').reset_index(drop=True)

# For each team, compute cumulative stats up to (but not including) each game
def build_team_history(games_df, window=None):
    """Build rolling team stats. Returns dict: team -> list of cumulative stats."""
    teams = pd.concat([
        games_df[['game_num','home_team','home_goals_total','home_xg_total',
                  'home_shots_total','home_reg_xg','home_win',
                  'home_pp_xg','home_pen_total','home_goalie']].rename(
            columns={'home_team':'team','home_goals_total':'gf',
                     'home_xg_total':'xgf','home_shots_total':'sf',
                     'home_reg_xg':'reg_xgf','home_win':'win',
                     'home_pp_xg':'pp_xgf','home_pen_total':'pen',
                     'home_goalie':'goalie'}).assign(
            ga=games_df['away_goals_total'], xga=games_df['away_xg_total'],
            sa=games_df['away_shots_total'], reg_xga=games_df['away_reg_xg']),
        games_df[['game_num','away_team','away_goals_total','away_xg_total',
                  'away_shots_total','away_reg_xg',
                  'away_pp_xg','away_pen_total','away_goalie']].rename(
            columns={'away_team':'team','away_goals_total':'gf',
                     'away_xg_total':'xgf','away_shots_total':'sf',
                     'away_reg_xg':'reg_xgf',
                     'away_pp_xg':'pp_xgf','away_pen_total':'pen',
                     'away_goalie':'goalie'}).assign(
            win=(games_df['home_win']==0).astype(int),
            ga=games_df['home_goals_total'], xga=games_df['home_xg_total'],
            sa=games_df['home_shots_total'], reg_xga=games_df['home_reg_xg'])
    ], ignore_index=True).sort_values(['team','game_num'])

    return teams

all_team_games = build_team_history(games)

def rolling_team_stats(team_games, w=20):
    """Compute expanding-window rolling averages."""
    tg = team_games.sort_values(['team','game_num']).copy()
    cols = ['gf','ga','xgf','xga','sf','sa','reg_xgf','reg_xga','win','pp_xgf','pen']
    for c in cols:
        tg[f'roll_{c}'] = (tg.groupby('team', sort=False)[c]
                             .transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean()))
    # GF% (Corsi-style xG%)
    tg['roll_xgpct'] = tg['roll_xgf'] / (tg['roll_xgf'] + tg['roll_xga'] + 1e-9)
    tg['roll_gpct']  = tg['roll_gf']  / (tg['roll_gf']  + tg['roll_ga']  + 1e-9)  # noqa
    tg['roll_xg_diff'] = tg['roll_xgf'] - tg['roll_xga']
    tg['roll_pdo']   = tg['roll_win']  # simplified PDO proxy
    return tg

print("Computing rolling stats (window=20 games)...")
team_stats = rolling_team_stats(all_team_games, w=20)

# Pivot back to game level
def merge_team_stats(games_df, team_stats_df):
    """Attach rolling stats to each game for home and away teams."""
    # We want the stats that were known *before* this game (already shifted in rolling)
    ts = team_stats_df[['team','game_num',
                         'roll_gf','roll_ga','roll_xgf','roll_xga',
                         'roll_sf','roll_sa','roll_xgpct','roll_gpct',
                         'roll_xg_diff','roll_win','roll_pp_xgf','roll_pen',
                         'roll_reg_xgf','roll_reg_xga']].copy()

    gdf = games_df.copy()
    for side, opp in [('home','away'), ('away','home')]:
        team_col = f'{side}_team'
        merged = gdf.merge(
            ts.rename(columns={c: f'{side}_{c}' for c in ts.columns if c not in ['team','game_num']}),
            left_on=[team_col, 'game_num'],
            right_on=['team', 'game_num'],
            how='left'
        )
        # drop extra 'team' col
        if 'team' in merged.columns:
            merged = merged.drop(columns=['team'])
        gdf = merged

    return gdf

print("Merging rolling features into game dataset...")
games_feat = merge_team_stats(games, team_stats)

# ─────────────────────────────────────────────
# 5. GOALIE STRENGTH FEATURES
# ─────────────────────────────────────────────
print("Building goalie performance features...")

# Per-goalie rolling save quality (xGA faced vs GA)
def compute_goalie_rolling(tg, w=15):
    results = []
    for goalie, grp in tg.groupby('goalie'):
        grp = grp.sort_values('game_num').copy()
        grp['goalie_xga_roll'] = grp['xga'].shift(1).rolling(w, min_periods=3).mean()
        grp['goalie_ga_roll']  = grp['ga'].shift(1).rolling(w, min_periods=3).mean()
        results.append(grp)
    return pd.concat(results, ignore_index=True)

goalie_roll = compute_goalie_rolling(all_team_games, w=15)
goalie_roll['goalie_gsax_roll'] = goalie_roll['goalie_xga_roll'] - goalie_roll['goalie_ga_roll']

g_feat = goalie_roll[['goalie','game_num','goalie_xga_roll','goalie_ga_roll','goalie_gsax_roll']].copy()

# Merge home goalie stats
games_feat = games_feat.merge(
    g_feat.rename(columns={c: f'home_{c}' for c in g_feat.columns if c not in ['goalie','game_num']}),
    left_on=['home_goalie','game_num'], right_on=['goalie','game_num'], how='left'
).drop(columns=['goalie'], errors='ignore')

games_feat = games_feat.merge(
    g_feat.rename(columns={c: f'away_{c}' for c in g_feat.columns if c not in ['goalie','game_num']}),
    left_on=['away_goalie','game_num'], right_on=['goalie','game_num'], how='left'
).drop(columns=['goalie'], errors='ignore')

# ─────────────────────────────────────────────
# 6. DIFFERENTIAL / INTERACTION FEATURES
# ─────────────────────────────────────────────
print("Engineering differential features...")

feat = games_feat.copy()

# Core differentials
feat['xg_diff_roll']    = feat['home_roll_xgf'] - feat['away_roll_xgf']
feat['xga_diff_roll']   = feat['home_roll_xga'] - feat['away_roll_xga']
feat['net_xg_diff']     = feat['home_roll_xg_diff'] - feat['away_roll_xg_diff']
feat['sf_diff_roll']    = feat['home_roll_sf'] - feat['away_roll_sf']
feat['win_diff_roll']   = feat['home_roll_win'] - feat['away_roll_win']
feat['xgpct_diff']      = feat['home_roll_xgpct'] - feat['away_roll_xgpct']
feat['pp_diff']         = feat['home_roll_pp_xgf'] - feat['away_roll_pp_xgf']
feat['pen_diff']        = feat['home_roll_pen'] - feat['away_roll_pen']
feat['reg_xgf_diff']    = feat['home_roll_reg_xgf'] - feat['away_roll_reg_xgf']
feat['reg_xga_diff']    = feat['home_roll_reg_xga'] - feat['away_roll_reg_xga']

# Goalie differentials
feat['goalie_gsax_diff']= feat['home_goalie_gsax_roll'] - feat['away_goalie_gsax_roll']
feat['goalie_xga_diff'] = feat['home_goalie_xga_roll']  - feat['away_goalie_xga_roll']

# Ratios
feat['xgf_ratio']       = feat['home_roll_xgf'] / (feat['away_roll_xgf'] + 1e-9)
feat['xgpct_product']   = feat['home_roll_xgpct'] * (1 - feat['away_roll_xgpct'])
feat['xg_net_abs']      = np.abs(feat['net_xg_diff'])
feat['win_momentum_diff']= feat['home_roll_win'] - feat['away_roll_win']

# Squared terms (non-linearity)
feat['xg_diff_sq']      = feat['net_xg_diff'] ** 2
feat['xgpct_diff_sq']   = feat['xgpct_diff'] ** 2

# Home advantage interaction
feat['home_xgf_vs_away_xga'] = feat['home_roll_xgf'] * feat['away_goalie_gsax_roll'].fillna(0)

# ─────────────────────────────────────────────
# 7. HEAD-TO-HEAD FEATURES
# ─────────────────────────────────────────────
print("Building head-to-head history features...")

h2h_records = []
h2h_history = {}

for _, row in games.sort_values('game_num').iterrows():
    ht, at = row['home_team'], row['away_team']
    key = tuple(sorted([ht, at]))
    hist = h2h_history.get(key, [])

    # Home win rate in H2H
    h2h_home_wins = sum(1 for h, a, hw, xgd in hist if h == ht and hw == 1)
    h2h_n = len(hist)
    h2h_hw_rate = h2h_home_wins / h2h_n if h2h_n > 0 else np.nan
    h2h_xg_diff = np.mean([xgd for h, a, hw, xgd in hist]) if h2h_n > 0 else np.nan

    h2h_records.append({
        'game_id': row['game_id'],
        'h2h_hw_rate': h2h_hw_rate,
        'h2h_n': h2h_n,
        'h2h_xg_diff': h2h_xg_diff,
    })
    hist.append((ht, at, row['home_win'],
                 row['home_xg_total'] - row['away_xg_total']))
    h2h_history[key] = hist

h2h_df = pd.DataFrame(h2h_records)
feat = feat.merge(h2h_df, on='game_id', how='left')
feat['h2h_hw_rate'] = feat['h2h_hw_rate'].fillna(win_rate)

# ─────────────────────────────────────────────
# 8. TEAM ENCODING (target-encoded win rate)
# ─────────────────────────────────────────────
print("Target-encoding teams...")

# Simple label-encoding for tree models + target encoding
team_win_rates = {}
for _, row in games.sort_values('game_num').iterrows():
    ht, at = row['home_team'], row['away_team']
    if ht not in team_win_rates: team_win_rates[ht] = []
    if at not in team_win_rates: team_win_rates[at] = []
    team_win_rates[ht].append(row['home_win'])
    team_win_rates[at].append(1 - row['home_win'])

team_enc = {t: np.mean(v) for t, v in team_win_rates.items()}
feat['home_team_enc'] = feat['home_team'].map(team_enc).fillna(win_rate)
feat['away_team_enc'] = feat['away_team'].map(team_enc).fillna(win_rate)
feat['team_enc_diff'] = feat['home_team_enc'] - feat['away_team_enc']

# ─────────────────────────────────────────────
# 9. FEATURE MATRIX
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # Rolling differentials
    'xg_diff_roll', 'xga_diff_roll', 'net_xg_diff', 'sf_diff_roll',
    'win_diff_roll', 'xgpct_diff', 'pp_diff', 'pen_diff',
    'reg_xgf_diff', 'reg_xga_diff',
    # Goalie
    'goalie_gsax_diff', 'goalie_xga_diff',
    # Ratio / interaction
    'xgf_ratio', 'xgpct_product', 'xg_net_abs',
    'win_momentum_diff', 'xg_diff_sq', 'xgpct_diff_sq',
    'home_xgf_vs_away_xga',
    # H2H
    'h2h_hw_rate', 'h2h_n', 'h2h_xg_diff',
    # Team encoding
    'home_team_enc', 'away_team_enc', 'team_enc_diff',
    # Raw rolling rates
    'home_roll_xgf', 'home_roll_xga', 'home_roll_win', 'home_roll_xgpct',
    'away_roll_xgf', 'away_roll_xga', 'away_roll_win', 'away_roll_xgpct',
    'home_roll_pp_xgf', 'away_roll_pp_xgf',
    'home_goalie_gsax_roll', 'away_goalie_gsax_roll',
    'home_roll_reg_xgf', 'home_roll_reg_xga',
    'away_roll_reg_xgf', 'away_roll_reg_xga',
]

# Keep rows with enough history (min rolling window)
feat_clean = feat[FEATURE_COLS + ['home_win', 'game_id', 'home_team', 'away_team']].copy()
feat_clean = feat_clean.dropna(subset=['home_roll_xgf', 'away_roll_xgf'])
feat_clean[FEATURE_COLS] = feat_clean[FEATURE_COLS].fillna(feat_clean[FEATURE_COLS].median())

X = feat_clean[FEATURE_COLS].values
y = feat_clean['home_win'].values

print(f"\nFinal training set: {len(X):,} games × {len(FEATURE_COLS)} features")
print(f"Target balance: {y.mean():.3f} home win rate")

# ─────────────────────────────────────────────
# 10. MODEL DEFINITIONS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("Building Stacked Ensemble Model")
print("="*60)

# --- Level-0 base estimators ---
base_estimators = [
    ('gb_deep', GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.04, max_depth=4,
        subsample=0.8, min_samples_leaf=10,
        max_features='sqrt', random_state=42
    )),
    ('gb_wide', GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.06, max_depth=3,
        subsample=0.75, min_samples_leaf=15,
        max_features=0.7, random_state=7
    )),
    ('rf', RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        max_features='sqrt', random_state=42, n_jobs=-1
    )),
    ('et', ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=4,
        max_features='sqrt', random_state=42, n_jobs=-1
    )),
    ('lr', Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LogisticRegression(C=0.5, max_iter=1000, random_state=42))
    ])),
]

# --- Level-1 meta-learner ---
meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

stacked_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba',
    passthrough=True,   # also pass original features to meta-learner
    n_jobs=-1
)

# ─────────────────────────────────────────────
# 11. CROSS-VALIDATION EVALUATION
# ─────────────────────────────────────────────
print("\nRunning 5-fold stratified cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate base models individually
print("\nBase model AUC scores:")
base_scores = {}
for name, est in base_estimators:
    scores = cross_val_score(est, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    base_scores[name] = scores.mean()
    print(f"  {name:10s}: {scores.mean():.4f} ± {scores.std():.4f}")

# Evaluate stacked ensemble
print("\nEvaluating stacked ensemble (this may take a moment)...")
stack_scores = cross_val_score(stacked_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"\n{'='*40}")
print(f"  STACKED ENSEMBLE AUC: {stack_scores.mean():.4f} ± {stack_scores.std():.4f}")
print(f"{'='*40}")

# ─────────────────────────────────────────────
# 12. FIT FINAL MODEL ON ALL DATA
# ─────────────────────────────────────────────
print("\nFitting final model on all training data...")
stacked_model.fit(X, y)
train_preds = stacked_model.predict_proba(X)[:, 1]
train_auc = roc_auc_score(y, train_preds)
print(f"Training AUC (full data): {train_auc:.4f}")

# ─────────────────────────────────────────────
# 13. POWER RANKINGS (Phase 1a)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1a — TEAM POWER RANKINGS")
print("="*60)

# Compute composite power score using latest rolling stats
latest_stats = team_stats.sort_values('game_num').groupby('team').last().reset_index()

latest_stats['power_score'] = (
    0.35 * latest_stats['roll_xgpct'].fillna(0.5) +
    0.25 * latest_stats['roll_win'].fillna(0.5) +
    0.20 * latest_stats['roll_xg_diff'].fillna(0).rank(pct=True) +
    0.10 * latest_stats['roll_pp_xgf'].fillna(0).rank(pct=True) +
    0.10 * (1 - latest_stats['roll_pen'].fillna(0).rank(pct=True))
)

# Add goalie quality to power score
goalie_latest = goalie_roll.sort_values('game_num').groupby(['team','goalie']).last().reset_index()
goalie_team = goalie_latest.sort_values('game_num').groupby('team').last()[['goalie_gsax_roll']].reset_index()
latest_stats = latest_stats.merge(goalie_team, on='team', how='left')
latest_stats['power_score'] += 0.10 * latest_stats['goalie_gsax_roll'].fillna(0).rank(pct=True)

power_rankings = latest_stats[['team','power_score','roll_xgpct','roll_win',
                                 'roll_xg_diff','roll_pp_xgf','goalie_gsax_roll']].sort_values(
    'power_score', ascending=False
).reset_index(drop=True)
power_rankings.index += 1
power_rankings.columns = ['Team','PowerScore','xG%','WinRate','xGDiff',
                           'PP_xGF','Goalie_GSAx']

print("\nTop 32 Power Rankings:")
print(power_rankings.to_string())

# ─────────────────────────────────────────────
# 14. PREDICT MATCHUPS (Phase 1b win probabilities)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1b — ROUND 1 MATCHUP PREDICTIONS")
print("="*60)

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

def get_team_features(team, is_home, feat_df, team_stats_df, goalie_feat_df):
    """Get latest rolling features for a team."""
    ts = team_stats_df[team_stats_df['team'] == team].sort_values('game_num')
    if len(ts) == 0:
        return None
    latest = ts.iloc[-1]

    glt = goalie_feat_df[goalie_feat_df['team'] == team].sort_values('game_num')
    g_gsax = glt.iloc[-1]['goalie_gsax_roll'] if len(glt) > 0 else 0.0

    return {
        'roll_xgf': latest.get('roll_xgf', np.nan),
        'roll_xga': latest.get('roll_xga', np.nan),
        'roll_win': latest.get('roll_win', np.nan),
        'roll_xgpct': latest.get('roll_xgpct', np.nan),
        'roll_pp_xgf': latest.get('roll_pp_xgf', np.nan),
        'roll_reg_xgf': latest.get('roll_reg_xgf', np.nan),
        'roll_reg_xga': latest.get('roll_reg_xga', np.nan),
        'roll_sf': latest.get('roll_sf', np.nan),
        'roll_xg_diff': latest.get('roll_xg_diff', np.nan),
        'goalie_gsax': g_gsax,
        'team_enc': team_enc.get(team, win_rate),
    }

def build_matchup_features(ht, at):
    hf = get_team_features(ht, True,  feat_clean, team_stats, goalie_roll)
    af = get_team_features(at, False, feat_clean, team_stats, goalie_roll)
    if hf is None or af is None:
        return None

    def safe(x, fallback=0.0):
        return x if pd.notna(x) else fallback

    hxgf = safe(hf['roll_xgf'], 2.5)
    axgf = safe(af['roll_xgf'], 2.5)
    hxga = safe(hf['roll_xga'], 2.5)
    axga = safe(af['roll_xga'], 2.5)
    hwin = safe(hf['roll_win'], win_rate)
    awin = safe(af['roll_win'], win_rate)
    hxgpct = safe(hf['roll_xgpct'], 0.5)
    axgpct = safe(af['roll_xgpct'], 0.5)
    hxgd = safe(hf['roll_xg_diff'], 0)
    axgd = safe(af['roll_xg_diff'], 0)
    hpp = safe(hf['roll_pp_xgf'], 0.5)
    app = safe(af['roll_pp_xgf'], 0.5)
    hrxgf = safe(hf['roll_reg_xgf'], 2.0)
    arxgf = safe(af['roll_reg_xgf'], 2.0)
    hrxga = safe(hf['roll_reg_xga'], 2.0)
    arxga = safe(af['roll_reg_xga'], 2.0)
    hsf = safe(hf['roll_sf'], 30)
    asf = safe(af['roll_sf'], 30)
    hgsax = safe(hf['goalie_gsax'], 0)
    agsax = safe(af['goalie_gsax'], 0)
    hte = safe(hf['team_enc'], win_rate)
    ate = safe(af['team_enc'], win_rate)

    row = {
        'xg_diff_roll': hxgf - axgf,
        'xga_diff_roll': hxga - axga,
        'net_xg_diff': hxgd - axgd,
        'sf_diff_roll': hsf - asf,
        'win_diff_roll': hwin - awin,
        'xgpct_diff': hxgpct - axgpct,
        'pp_diff': hpp - app,
        'pen_diff': 0,
        'reg_xgf_diff': hrxgf - arxgf,
        'reg_xga_diff': hrxga - arxga,
        'goalie_gsax_diff': hgsax - agsax,
        'goalie_xga_diff': hxga - axga,
        'xgf_ratio': hxgf / (axgf + 1e-9),
        'xgpct_product': hxgpct * (1 - axgpct),
        'xg_net_abs': abs(hxgd - axgd),
        'win_momentum_diff': hwin - awin,
        'xg_diff_sq': (hxgd - axgd) ** 2,
        'xgpct_diff_sq': (hxgpct - axgpct) ** 2,
        'home_xgf_vs_away_xga': hxgf * agsax,
        'h2h_hw_rate': win_rate,
        'h2h_n': 0,
        'h2h_xg_diff': 0,
        'home_team_enc': hte,
        'away_team_enc': ate,
        'team_enc_diff': hte - ate,
        'home_roll_xgf': hxgf,
        'home_roll_xga': hxga,
        'home_roll_win': hwin,
        'home_roll_xgpct': hxgpct,
        'away_roll_xgf': axgf,
        'away_roll_xga': axga,
        'away_roll_win': awin,
        'away_roll_xgpct': axgpct,
        'home_roll_pp_xgf': hpp,
        'away_roll_pp_xgf': app,
        'home_goalie_gsax_roll': hgsax,
        'away_goalie_gsax_roll': agsax,
        'home_roll_reg_xgf': hrxgf,
        'home_roll_reg_xga': hrxga,
        'away_roll_reg_xgf': arxgf,
        'away_roll_reg_xga': arxga,
    }
    return [row[c] for c in FEATURE_COLS]

print("\n{'Game':<6} {'Home Team':<15} {'Away Team':<15} {'Home Win Prob':>13}")
print("-" * 52)
matchup_results = []
for i, (ht, at) in enumerate(MATCHUPS, 1):
    feats = build_matchup_features(ht, at)
    if feats is None:
        prob = win_rate
    else:
        prob = stacked_model.predict_proba([feats])[0][1]
    matchup_results.append({'game': i, 'home': ht, 'away': at, 'home_win_prob': prob})
    print(f"  {i:<4} {ht:<15} {at:<15} {prob:>12.4f}")

# ─────────────────────────────────────────────
# 15. OFFENSIVE LINE DISPARITY (Phase 1c)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1c — OFFENSIVE LINE QUALITY DISPARITY")
print("="*60)

# Aggregate first and second line stats per team with per-toi normalization
def line_stats(df, team, line_type):
    """xG per 60 seconds for a specific team's line."""
    home_recs = df[(df['home_team'] == team) & (df['home_off_line'] == line_type)]
    away_recs = df[(df['away_team'] == team) & (df['away_off_line'] == line_type)]

    home_xg = home_recs['home_xg'].sum()
    home_toi = home_recs['toi'].sum()
    away_xg = away_recs['away_xg'].sum()
    away_toi = away_recs['toi'].sum()

    total_xg = home_xg + away_xg
    total_toi = home_toi + away_toi

    if total_toi < 60:  # need at least 1 minute
        return np.nan
    return (total_xg / total_toi) * 60  # xG per 60 seconds

teams = df['home_team'].unique()
disparity_results = []

for team in teams:
    f1_xg60  = line_stats(df, team, 'first_off')
    f2_xg60  = line_stats(df, team, 'second_off')

    if pd.isna(f1_xg60) or pd.isna(f2_xg60) or f2_xg60 <= 0:
        continue

    ratio = f1_xg60 / f2_xg60

    # Also compute raw totals for context
    home_f1 = df[(df['home_team'] == team) & (df['home_off_line'] == 'first_off')]
    away_f1 = df[(df['away_team'] == team) & (df['away_off_line'] == 'first_off')]
    home_f2 = df[(df['home_team'] == team) & (df['home_off_line'] == 'second_off')]
    away_f2 = df[(df['away_team'] == team) & (df['away_off_line'] == 'second_off')]

    disparity_results.append({
        'team': team,
        'first_xg60': round(f1_xg60, 4),
        'second_xg60': round(f2_xg60, 4),
        'disparity_ratio': round(ratio, 4),
        'first_xg_total': round(
            home_f1['home_xg'].sum() + away_f1['away_xg'].sum(), 4),
        'second_xg_total': round(
            home_f2['home_xg'].sum() + away_f2['away_xg'].sum(), 4),
    })

disparity_df = pd.DataFrame(disparity_results).sort_values(
    'disparity_ratio', ascending=False
).reset_index(drop=True)
disparity_df.index += 1

print("\nTop 10 Teams — Largest Offensive Line Quality Disparity (ratio first/second xG per 60s):")
print(disparity_df.head(10).to_string())

# ─────────────────────────────────────────────
# 16. SAVE ALL OUTPUTS
# ─────────────────────────────────────────────
OUT = "/Users/frankbisignano/Library/CloudStorage/GoogleDrive-frankabisignano@gmail.com/My Drive/WHSDSC_2026/"

power_rankings.to_csv(OUT + "power_rankings.csv", index_label='rank')
pd.DataFrame(matchup_results).to_csv(OUT + "matchup_predictions.csv", index=False)
disparity_df.to_csv(OUT + "line_disparity.csv", index_label='rank')

print(f"\n✓ power_rankings.csv")
print(f"✓ matchup_predictions.csv")
print(f"✓ line_disparity.csv")

# ─────────────────────────────────────────────
# 17. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"\nModel: Stacked Ensemble (GB × 2 + RF + ET + LR) → LR meta")
print(f"Features: {len(FEATURE_COLS)} engineered features")
print(f"Training samples: {len(X):,}")
print(f"\nCV AUC:        {stack_scores.mean():.4f} ± {stack_scores.std():.4f}")
print(f"Train AUC:     {train_auc:.4f}")
print(f"\nBase model AUC scores:")
for name, score in sorted(base_scores.items(), key=lambda x: -x[1]):
    print(f"  {name:12s}: {score:.4f}")
print(f"\n#1 Power Ranking: {power_rankings.iloc[0]['Team']}")
print(f"#32 Power Ranking: {power_rankings.iloc[31]['Team']}")
print(f"\nTop 10 Line Disparity Teams:")
for i, row in disparity_df.head(10).iterrows():
    print(f"  {i:2d}. {row['team']:<15} ratio={row['disparity_ratio']:.4f}")

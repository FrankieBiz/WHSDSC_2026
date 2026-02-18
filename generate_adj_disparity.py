"""
WHSDSC 2026 — Opponent-Adjusted Line Quality Disparity (Phase 1b)
===================================================================
Adjusts each team's offensive line xG/60 for the quality of defensive
pairings they faced. This is specifically requested by the competition:
  "Consider accounting for TOI and defensive matchups since tougher
   opponents can affect performance."

Method:
  1. Compute each (team, def_pairing) defensive strength = xGA/60 allowed
  2. For each offensive line matchup, compute adjustment factor:
       adj_factor = league_avg_def_strength / opponent_def_strength
     If the opponent defense is tougher than average, factor > 1 (boosts xG).
     If the opponent defense is weaker than average, factor < 1 (penalizes xG).
  3. Adjusted xG = raw xG * adj_factor per matchup row
  4. Disparity ratio = adjusted first-line xG/60 / adjusted second-line xG/60
"""
import pandas as pd
import numpy as np

ROOT = "/Users/frankbisignano/Library/CloudStorage/GoogleDrive-frankabisignano@gmail.com/My Drive/WHSDSC_2026/"
df = pd.read_csv(ROOT + "whl_2025.csv")
SPEC = {'PP_up', 'PP_kill_dwn', 'PP_kill_up', 'empty_net_line'}

print(f"Loaded {len(df):,} records")

# ── Step 1: Compute defensive pairing strength ───────────────
# For each (team, def_pairing), how much xG/60 do they ALLOW?
# i.e., the opponent's xG generated against this defensive pairing.
print("\nComputing defensive pairing strength (xGA/60 allowed)...")

def_strength = {}
teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))

for team in teams:
    for dp in ['first_def', 'second_def']:
        # When team is home and deploying this def pairing
        h = df[(df['home_team'] == team) & (df['home_def_pairing'] == dp)]
        # When team is away and deploying this def pairing
        a = df[(df['away_team'] == team) & (df['away_def_pairing'] == dp)]

        # xG ALLOWED = opponent's xG against this def pairing
        xga = h['away_xg'].sum() + a['home_xg'].sum()
        toi = h['toi'].sum() + a['toi'].sum()

        if toi > 60:  # minimum 1 minute
            def_strength[(team, dp)] = (xga / toi) * 60
        else:
            def_strength[(team, dp)] = np.nan

league_avg_def = np.nanmean(list(def_strength.values()))
print(f"League average defensive xGA/60: {league_avg_def:.5f}")
print(f"Defensive pairings computed: {len(def_strength)}")

# Fill NaN with league average
for k in def_strength:
    if np.isnan(def_strength[k]):
        def_strength[k] = league_avg_def

# ── Step 2: Compute adjusted xG per matchup row ─────────────
print("Computing opponent-adjusted xG per matchup...")

# Filter to only even-strength offensive lines (first_off, second_off)
es = df[df['home_off_line'].isin(['first_off', 'second_off']) |
        df['away_off_line'].isin(['first_off', 'second_off'])].copy()

# For each row, adjust xG based on opponent's defensive pairing quality
# Home team's first/second line faces away team's defensive pairing
# Away team's first/second line faces home team's defensive pairing

rows = []
for _, r in es.iterrows():
    # Home offensive line vs away defensive pairing
    if r['home_off_line'] in ['first_off', 'second_off']:
        opp_def_key = (r['away_team'], r['away_def_pairing'])
        opp_def_str = def_strength.get(opp_def_key, league_avg_def)
        # Tougher defense (higher xGA/60) → less adjustment needed (they allow goals)
        # Weaker defense (lower xGA/60) → penalize the offense (they scored vs weak D)
        # adj_factor = opp_def_str / league_avg  would boost vs tough D
        # But we want: scoring vs tough D is worth more, so:
        # adj_factor = league_avg / opp_def_str  → if opp allows LESS, factor > 1
        # Wait—higher xGA/60 = weaker defense (allows more xG).
        # Lower xGA/60 = stronger defense (allows less xG).
        # So if you score against a strong defense (low xGA/60), your xG should be boosted:
        # adj_factor = league_avg_def / opp_def_str
        # opp_def_str low (tough D) → factor > 1 → boost xG
        # opp_def_str high (weak D) → factor < 1 → penalize xG
        adj_factor = league_avg_def / max(opp_def_str, 1e-6)
        rows.append({
            'team': r['home_team'],
            'off_line': r['home_off_line'],
            'raw_xg': r['home_xg'],
            'adj_xg': r['home_xg'] * adj_factor,
            'toi': r['toi'],
            'opp_def': r['away_def_pairing'],
            'opp_team': r['away_team'],
            'adj_factor': adj_factor,
        })

    # Away offensive line vs home defensive pairing
    if r['away_off_line'] in ['first_off', 'second_off']:
        opp_def_key = (r['home_team'], r['home_def_pairing'])
        opp_def_str = def_strength.get(opp_def_key, league_avg_def)
        adj_factor = league_avg_def / max(opp_def_str, 1e-6)
        rows.append({
            'team': r['away_team'],
            'off_line': r['away_off_line'],
            'raw_xg': r['away_xg'],
            'adj_xg': r['away_xg'] * adj_factor,
            'toi': r['toi'],
            'opp_def': r['home_def_pairing'],
            'opp_team': r['home_team'],
            'adj_factor': adj_factor,
        })

adj_df = pd.DataFrame(rows)
print(f"Matchup rows: {len(adj_df):,}")

# ── Step 3: Compute adjusted xG/60 per (team, line) ─────────
print("\nComputing adjusted & raw xG/60 per team per line...")

line_stats = adj_df.groupby(['team', 'off_line']).agg(
    raw_xg=('raw_xg', 'sum'),
    adj_xg=('adj_xg', 'sum'),
    toi=('toi', 'sum'),
    matchups=('raw_xg', 'count'),
).reset_index()

line_stats['raw_xg60'] = (line_stats['raw_xg'] / line_stats['toi']) * 60
line_stats['adj_xg60'] = (line_stats['adj_xg'] / line_stats['toi']) * 60

# ── Step 4: Compute disparity ratio ─────────────────────────
print("\nBuilding disparity table...")

disp_rows = []
for team in teams:
    f1 = line_stats[(line_stats['team'] == team) & (line_stats['off_line'] == 'first_off')]
    f2 = line_stats[(line_stats['team'] == team) & (line_stats['off_line'] == 'second_off')]

    if f1.empty or f2.empty:
        continue

    f1 = f1.iloc[0]
    f2 = f2.iloc[0]

    if f1['toi'] < 60 or f2['toi'] < 60:
        continue

    # Raw disparity
    raw_ratio = f1['raw_xg60'] / f2['raw_xg60'] if f2['raw_xg60'] > 0 else np.nan
    # Adjusted disparity
    adj_ratio = f1['adj_xg60'] / f2['adj_xg60'] if f2['adj_xg60'] > 0 else np.nan

    disp_rows.append({
        'team': team,
        'first_line_raw_xg60': round(f1['raw_xg60'], 5),
        'second_line_raw_xg60': round(f2['raw_xg60'], 5),
        'raw_disparity_ratio': round(raw_ratio, 4) if pd.notna(raw_ratio) else np.nan,
        'first_line_adj_xg60': round(f1['adj_xg60'], 5),
        'second_line_adj_xg60': round(f2['adj_xg60'], 5),
        'adj_disparity_ratio': round(adj_ratio, 4) if pd.notna(adj_ratio) else np.nan,
        'first_xg_total': round(f1['raw_xg'], 3),
        'second_xg_total': round(f2['raw_xg'], 3),
        'first_adj_xg_total': round(f1['adj_xg'], 3),
        'second_adj_xg_total': round(f2['adj_xg'], 3),
        'first_toi': round(f1['toi'], 1),
        'second_toi': round(f2['toi'], 1),
    })

disp_df = pd.DataFrame(disp_rows)

# Sort by ADJUSTED disparity ratio (the primary metric)
disp_df = disp_df.sort_values('adj_disparity_ratio', ascending=False).reset_index(drop=True)
disp_df.index += 1

print("\nAll 32 Teams — Opponent-Adjusted Line Quality Disparity:")
print(disp_df[['team', 'first_line_adj_xg60', 'second_line_adj_xg60',
               'adj_disparity_ratio', 'raw_disparity_ratio']].to_string())

print("\n" + "=" * 65)
print("TOP 10 — Highest Offensive Line Quality Disparity (Adjusted)")
print("=" * 65)
for i, row in disp_df.head(10).iterrows():
    print(f"  {i:2d}. {row['team']:<16}  adj_ratio={row['adj_disparity_ratio']:.4f}  "
          f"raw_ratio={row['raw_disparity_ratio']:.4f}  "
          f"1st_adj={row['first_line_adj_xg60']:.5f}  2nd_adj={row['second_line_adj_xg60']:.5f}")

# ── Save ─────────────────────────────────────────────────────
out = ROOT + "line_disparity_adjusted.csv"
disp_df.to_csv(out, index_label='rank')
print(f"\nSaved: {out}")

# Also overwrite the main line_disparity.csv with adjusted version for the visualization
disp_out = disp_df[['team', 'first_line_adj_xg60', 'second_line_adj_xg60',
                     'adj_disparity_ratio', 'first_xg_total', 'second_xg_total']].copy()
disp_out.columns = ['team', 'first_line_xg60', 'second_line_xg60',
                     'disparity_ratio', 'first_xg_total', 'second_xg_total']
out2 = ROOT + "line_disparity.csv"
disp_out.to_csv(out2, index_label='rank')
print(f"Updated: {out2}")

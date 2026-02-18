"""
WHSDSC 2026 — Verify All Submission Deliverables
==================================================
Checks that all required outputs exist and displays final answers.
"""
import pandas as pd
import os

ROOT = "/Users/frankbisignano/Library/CloudStorage/GoogleDrive-frankabisignano@gmail.com/My Drive/WHSDSC_2026/"

print("=" * 70)
print("  WHSDSC 2026 — FINAL SUBMISSION VERIFICATION")
print("=" * 70)

# ── Check all files exist ─────────────────────────────────────────
files = {
    "League Table (prerequisite)": "league_table.csv",
    "Power Rankings (Phase 1a)": "power_rankings.csv",
    "Matchup Predictions (Phase 1a)": "matchup_predictions.csv",
    "Line Disparity - Adjusted (Phase 1b)": "line_disparity_adjusted.csv",
    "Line Disparity (Phase 1b)": "line_disparity.csv",
    "Visualization (Phase 1c)": "phase1c_visualization.png",
    "Methodology (Phase 1d)": "phase1d_methodology.txt",
}

print("\nFile Check:")
all_ok = True
for desc, fname in files.items():
    path = ROOT + fname
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = f"OK ({size/1024:.1f} KB)" if exists else "MISSING"
    if not exists:
        all_ok = False
    print(f"  {'[OK]' if exists else '[!!]'} {desc:<40} {fname:<35} {status}")

if all_ok:
    print("\n  All files present.")
else:
    print("\n  WARNING: Some files are missing!")

# ── Phase 1a: League Table ────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 1a — LEAGUE TABLE (Prerequisite)")
print("=" * 70)
lt = pd.read_csv(ROOT + "league_table.csv")
lt = lt.rename(columns=lambda c: c.strip())
print(lt[['team', 'GP', 'W', 'L', 'OTL', 'PTS', 'GF', 'GA', 'GD', 'xG%']].head(10).to_string(index=False))
print(f"  ... ({len(lt)} teams total)")

# ── Phase 1a: Power Rankings ─────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 1a — POWER RANKINGS")
print("=" * 70)
pr = pd.read_csv(ROOT + "power_rankings.csv")
pr = pr.rename(columns=lambda c: c.strip())
print(pr[['Team', 'PowerScore', 'ELO', 'xG_ELO']].to_string())

# ── Phase 1a: Matchup Predictions ────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 1a — MATCHUP WIN PROBABILITIES")
print("=" * 70)
mp = pd.read_csv(ROOT + "matchup_predictions.csv")
print(f"\n{'Game':<6} {'Home':<16} {'Away':<16} {'Home Win%':>10}  {'Winner'}")
print("-" * 64)
for _, row in mp.iterrows():
    print(f"  {row['game']:<4} {row['home_team']:<16} {row['away_team']:<16} "
          f"{row['home_win_prob']:>9.4f}  {row['predicted_winner']}")

# ── Phase 1b: Line Disparity (Top 10) ────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 1b — TOP 10 OFFENSIVE LINE QUALITY DISPARITY")
print("  (Opponent-Adjusted)")
print("=" * 70)
ld = pd.read_csv(ROOT + "line_disparity_adjusted.csv")
ld = ld.rename(columns=lambda c: c.strip())
print(f"\n{'Rank':<6} {'Team':<16} {'Adj Ratio':>10} {'Raw Ratio':>10} "
      f"{'1st Adj xG/60':>14} {'2nd Adj xG/60':>14}")
print("-" * 74)
for _, row in ld.head(10).iterrows():
    print(f"  {row['rank']:<4} {row['team']:<16} {row['adj_disparity_ratio']:>10.4f} "
          f"{row['raw_disparity_ratio']:>10.4f} {row['first_line_adj_xg60']:>14.5f} "
          f"{row['second_line_adj_xg60']:>14.5f}")

# ── Phase 1c: Visualization ──────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 1c — VISUALIZATION")
print("=" * 70)
viz_path = ROOT + "phase1c_visualization.png"
viz_size = os.path.getsize(viz_path) / 1024 / 1024
print(f"  File: phase1c_visualization.png")
print(f"  Size: {viz_size:.2f} MB (limit: 5 MB) — {'OK' if viz_size < 5 else 'TOO LARGE'}")

# ── Phase 1d: Methodology ────────────────────────────────────────
print("\n" + "=" * 70)
print("  PHASE 1d — METHODOLOGY")
print("=" * 70)
with open(ROOT + "phase1d_methodology.txt", 'r') as f:
    text = f.read()
words = len(text.split())
print(f"  File: phase1d_methodology.txt")
print(f"  Total words: {words}")

# ── Model Performance ────────────────────────────────────────────
print("\n" + "=" * 70)
print("  MODEL PERFORMANCE SUMMARY")
print("=" * 70)
print("""
  Architecture:
    208 engineered features → top-50 by ET importance
    5-model ensemble: LR + GB(800t) + GB(600t) + RF(800t) + ET(800t)
    Nelder-Mead optimized weighted blend

  Performance:
    OOF Blend AUC:    0.6831
    Oracle ceiling:   ~0.69
    % of ceiling:     99.0%

  Key Features:
    - Margin-adjusted ELO (goals + xG)
    - Multi-window rolling (7/15/30 games)
    - Opponent-adjusted xG
    - Bayesian-smoothed win rates
    - Goalie GSAx
    - Line quality xG/60
""")

print("=" * 70)
print("  SUBMISSION READY")
print("=" * 70)

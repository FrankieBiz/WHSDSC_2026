"""
WHSDSC 2026 — Phase 1c: Data Visualization
============================================
Scatter plot: Opponent-Adjusted Offensive Line Quality Disparity vs Team Power Score
Shows whether teams with more balanced (evenly-matched) offensive lines
tend to be stronger or weaker overall.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
import os

OUT = "/Users/frankbisignano/Library/CloudStorage/GoogleDrive-frankabisignano@gmail.com/My Drive/WHSDSC_2026/"

# ── Load saved outputs ────────────────────────────────────────────
pr = pd.read_csv(OUT + "power_rankings.csv")
ld = pd.read_csv(OUT + "line_disparity.csv")

pr = pr.rename(columns=lambda c: c.strip())
ld = ld.rename(columns=lambda c: c.strip())

merged = ld.merge(pr, left_on='team', right_on='Team', how='inner')
print(f"Merged: {len(merged)} teams")

# ── Correlation ───────────────────────────────────────────────────
x = merged['disparity_ratio'].values
y = merged['PowerScore'].values
r, p = stats.pearsonr(x, y)
rho, p_rho = stats.spearmanr(x, y)
print(f"Pearson r  = {r:.3f}  (p = {p:.4f})")
print(f"Spearman ρ = {rho:.3f}  (p = {p_rho:.4f})")

slope, intercept, r_val, p_val, se = stats.linregress(x, y)
x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
y_line = slope * x_line + intercept

# ── Classify teams into quadrants ─────────────────────────────────
med_disp  = np.median(x)
med_power = np.median(y)

colors = []
for xi, yi in zip(x, y):
    if xi < med_disp and yi > med_power:
        colors.append('#2a9d8f')   # balanced & strong
    elif xi >= med_disp and yi > med_power:
        colors.append('#e9c46a')   # top-heavy & strong
    elif xi < med_disp and yi <= med_power:
        colors.append('#264653')   # balanced & weak
    else:
        colors.append('#e76f51')   # top-heavy & weak

# ── PLOT ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9), dpi=200)

# Soft quadrant shading
x_range = [x.min() - 0.04, x.max() + 0.04]
y_range = [y.min() - 0.05, y.max() + 0.05]
ax.axhspan(med_power, y_range[1], xmin=0, xmax=0.5, alpha=0.04, color='#2a9d8f', zorder=0)
ax.axhspan(med_power, y_range[1], xmin=0.5, xmax=1.0, alpha=0.04, color='#e9c46a', zorder=0)
ax.axhspan(y_range[0], med_power, xmin=0, xmax=0.5, alpha=0.04, color='#264653', zorder=0)
ax.axhspan(y_range[0], med_power, xmin=0.5, xmax=1.0, alpha=0.04, color='#e76f51', zorder=0)

# Median reference lines
ax.axhline(med_power, color='#aaaaaa', linewidth=0.9, linestyle='--', alpha=0.6, zorder=1)
ax.axvline(med_disp,  color='#aaaaaa', linewidth=0.9, linestyle='--', alpha=0.6, zorder=1)

# Quadrant labels
props = dict(fontsize=9.5, color='#999999', fontstyle='italic', ha='center', fontweight='bold')
pad_x = (x.max() - x.min()) * 0.15
pad_y = (y.max() - y.min()) * 0.05
ax.text(x.min() + pad_x, y.max() - pad_y, 'Balanced & Strong', **props, va='top')
ax.text(x.max() - pad_x, y.max() - pad_y, 'Top-Heavy & Strong', **props, va='top')
ax.text(x.min() + pad_x, y.min() + pad_y, 'Balanced & Weak', **props, va='bottom')
ax.text(x.max() - pad_x, y.min() + pad_y, 'Top-Heavy & Weak', **props, va='bottom')

# Regression line
ax.plot(x_line, y_line, color='#c44e3b', linewidth=2.2, alpha=0.55, linestyle='-', zorder=3)

# Scatter points
ax.scatter(x, y, c=colors, s=140, edgecolors='white', linewidths=1.0, zorder=5, alpha=0.92)

# ── Smart label placement ─────────────────────────────────────────
# Sort by distance from center to label outliers more prominently
center_x, center_y = np.mean(x), np.mean(y)
merged['dist'] = np.sqrt((merged['disparity_ratio'] - center_x)**2 + (merged['PowerScore'] - center_y)**2)

# Manual offsets for commonly overlapping labels
manual_offsets = {}

for _, row in merged.iterrows():
    team = row['team']
    xi = row['disparity_ratio']
    yi = row['PowerScore']

    # Default offset
    offset_x, offset_y = 5, 5
    ha, va = 'left', 'bottom'

    # Adjust for edge cases
    if xi > med_disp + (x.max() - med_disp) * 0.5:
        ha = 'right'
        offset_x = -5
    if yi > med_power + (y.max() - med_power) * 0.7:
        va = 'top'
        offset_y = -5

    if team in manual_offsets:
        offset_x, offset_y, ha, va = manual_offsets[team]

    ax.annotate(team, (xi, yi),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                fontsize=6.2, color='#333333', alpha=0.82,
                ha=ha, va=va,
                arrowprops=dict(arrowstyle='-', color='#cccccc', lw=0.4)
                if row['dist'] > np.percentile(merged['dist'], 60) else None)

# ── Formatting ────────────────────────────────────────────────────
ax.set_xlabel('Opponent-Adjusted Offensive Line Disparity Ratio\n'
              '(First Line Adj. xG/60 ÷ Second Line Adj. xG/60)',
              fontsize=12, fontweight='bold', labelpad=12)
ax.set_ylabel('Team Power Score\n'
              '(Composite: ELO + xG% + Bayesian Win Rate + Goalie GSAx + PP)',
              fontsize=12, fontweight='bold', labelpad=12)

ax.set_title('Do Teams With More Balanced Offensive Lines Perform Better?',
             fontsize=17, fontweight='bold', pad=18)
ax.text(0.5, 1.02,
        'WHL 2025 Season — Opponent-Adjusted Line Disparity vs Overall Team Strength (n = 32 teams)',
        transform=ax.transAxes, fontsize=10.5, ha='center', color='#666666')

# Legend
legend_elements = [
    Patch(facecolor='#2a9d8f', edgecolor='white', label='Balanced & Strong'),
    Patch(facecolor='#e9c46a', edgecolor='white', label='Top-Heavy & Strong'),
    Patch(facecolor='#264653', edgecolor='white', label='Balanced & Weak'),
    Patch(facecolor='#e76f51', edgecolor='white', label='Top-Heavy & Weak'),
    Line2D([0], [0], color='#c44e3b', linewidth=2, alpha=0.6,
           label=f'Trend Line (r = {r:.2f})'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
          framealpha=0.92, edgecolor='#cccccc', fancybox=True)

# Caption / footnote
caption = (
    f"Disparity ratio = first line adj. xG/60 ÷ second line adj. xG/60. "
    f"Values closer to 1.0 = balanced lines; higher = first-line dominant.\n"
    f"xG/60 adjusted for opponent defensive pairing strength (league-avg normalization). "
    f"Power Score: weighted composite of ELO, xG%, Bayesian win rate, opp-adj xG%, PP xGF, goalie GSAx.\n"
    f"Pearson r = {r:.2f} (p = {p:.3f}), Spearman ρ = {rho:.2f} (p = {p_rho:.3f}). "
    f"Dashed lines = medians. Quadrant shading for visual grouping."
)
fig.text(0.10, 0.005, caption, fontsize=7, color='#555555',
         ha='left', va='bottom', fontstyle='italic',
         wrap=True)

ax.set_xlim(x.min() - 0.04, x.max() + 0.04)
ax.set_ylim(y.min() - 0.05, y.max() + 0.05)
ax.tick_params(axis='both', labelsize=10)
ax.grid(True, alpha=0.12, linewidth=0.5)
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')

plt.tight_layout(rect=[0, 0.045, 1, 1])

# Save
out_path = OUT + "phase1c_visualization.png"
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out_path}")
size = os.path.getsize(out_path)
print(f"File size: {size / 1024 / 1024:.2f} MB (limit: 5 MB)")

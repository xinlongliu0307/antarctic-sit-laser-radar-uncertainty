#!/usr/bin/env python3
"""
R3_Step10b_dhi_figures_v10.py — Figure dhi redrawn for Alex's comment 8 (27 Sep 2026)
=====================================================================================
Replaces only Figure C2_02 of R3_Step10_snow_swap_figures_v9.py (the scatter and the
heatmap are unchanged). The old three-panel figure is split in two:

  R3_Fig_C2_02_dhi_distributions_v10.{pdf,png}   -> \label{fig:dhi}
      (a) CCI v4.0, 2022-2023   (b) Cryo-TEMPO, 2022-2025
      violin + box per sub-sector; median diamond only; no mean/median/n text boxes,
      no red/blue sign boxes, no suptitle, no parameter string (all go in the caption);
      y-label in words; n under each tick label; boxed legend
  R3_Fig_C2_04_dhi_delta_sweep_v10.{pdf,png}     -> \label{fig:dhi_delta}  (old panel c)
      (a) CCI v4.0   (b) Cryo-TEMPO; one panel per product so the curves are not
      stacked; sub-sectors differ by colour AND marker AND line style, so the Eastern
      Weddell and Ross curves stay distinguishable where they coincide; all-sector
      median in black; empirical optimum as a shaded band; boxed legend

Input : R3_SIT_snow_swap_all_winters_v7.csv, R3_snow_swap_delta_sweep_v7.csv (Step 09)
        (if Step 18 shows case B, rerun Step 09 first; this script then reads the new files)
Run   : module load python3/3.9.2; source ~/cryo2ice_env/bin/activate
        python R3_Step10b_dhi_figures_v10.py        (< 1 min, login node)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import DELTA_BASELINE

L2P = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation'
IN_CSV = f'{L2P}/R3_SIT_snow_swap_all_winters_v7.csv'
SWEEP_CSV = f'{L2P}/R3_snow_swap_delta_sweep_v7.csv'
OUT_DIR = f'{L2P}/figures_v10'
os.makedirs(OUT_DIR, exist_ok=True)

SECTORS = ['Western_Weddell', 'Eastern_Weddell', 'Ross']
LAB = {'Western_Weddell': 'Western Weddell', 'Eastern_Weddell': 'Eastern Weddell', 'Ross': 'Ross', 'All': 'All sub-sectors'}
COL = {'Western_Weddell': '#0072B2', 'Eastern_Weddell': '#009E73', 'Ross': '#D55E00', 'All': 'black'}
MRK = {'Western_Weddell': 'o', 'Eastern_Weddell': 's', 'Ross': '^', 'All': 'D'}
LS = {'Western_Weddell': '-', 'Eastern_Weddell': '--', 'Ross': ':', 'All': '-'}
MFC = {'Western_Weddell': '#0072B2', 'Eastern_Weddell': 'white', 'Ross': '#D55E00', 'All': 'black'}
PRODUCTS = [('CCI', 'Delta_h_i_CCI', 'CCI v4.0, 2022–2023'),
            ('CT', 'Delta_h_i_CT', 'Cryo-TEMPO, 2022–2025')]

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.linewidth': 0.9,
})

df = pd.read_csv(IN_CSV, low_memory=False)
df['Delta_h_i_CCI'] = df['h_i_Imp_CCI'] - df['h_i_Base_CCI']
df['Delta_h_i_CT'] = df['h_i_Imp_CT'] - df['h_i_Base_CT']

# ── Figure dhi: distributions ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
for ax, (tag, col, title), letter in zip(axes, PRODUCTS, 'ab'):
    data, ticks = [], []
    for s in SECTORS:
        v = df.loc[(df['sector'] == s) & df[col].notna(), col].values
        data.append(v[(v > -5) & (v < 5)])
        ticks.append(f"{LAB[s]}\n(n = {len(v):,})")
    pos = np.arange(1, len(SECTORS) + 1)
    vp = ax.violinplot(data, positions=pos, showextrema=False, widths=0.8)
    for body, s in zip(vp['bodies'], SECTORS):
        body.set_facecolor(COL[s]); body.set_edgecolor(COL[s]); body.set_alpha(0.25)
    bp = ax.boxplot(data, positions=pos, widths=0.22, patch_artist=True, showfliers=False,
                    medianprops=dict(color='none'), whiskerprops=dict(color='0.3', lw=1.0),
                    capprops=dict(color='0.3', lw=1.0))
    for patch, s in zip(bp['boxes'], SECTORS):
        patch.set_facecolor(COL[s]); patch.set_alpha(0.7); patch.set_edgecolor('0.2')
    ax.plot(pos, [np.median(v) for v in data], 'D', color='black', ms=7, zorder=5)
    ax.axhline(0, color='0.4', ls=':', lw=1.0, zorder=0)
    ax.set_xticks(pos)
    ax.set_xticklabels(ticks)
    ax.set_title(f'({letter}) {title}', loc='left', fontweight='bold')
    ax.set_ylim(-2, 4)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(axis='y', alpha=0.3)
axes[0].set_ylabel('Change in sea-ice thickness (m)')
axes[1].legend(handles=[Line2D([0], [0], marker='D', color='black', ls='None', ms=7, label='Median'),
                        Patch(facecolor='0.6', edgecolor='0.2', label='Interquartile range'),
                        Line2D([0], [0], color='0.3', lw=1.0, label='1.5 × IQR')],
               loc='upper right', frameon=True, fancybox=False, edgecolor='black', framealpha=1)
fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(f'{OUT_DIR}/R3_Fig_C2_02_dhi_distributions_v10.{ext}')
plt.close(fig)
print('Saved R3_Fig_C2_02_dhi_distributions_v10')
for tag, col, _ in PRODUCTS:
    for s in SECTORS:
        v = df.loc[(df['sector'] == s) & df[col].notna(), col]
        print(f'  {tag:3s} {s:16s} n={len(v):6,d} median {v.median():+.3f} m')
    print(f'  {tag:3s} {"All":16s} n={df[col].notna().sum():6,d} median {df[col].median():+.3f} m')

# ── Figure dhi_delta: penetration-fraction sweep ──────────────────────────
sw = pd.read_csv(SWEEP_CSV)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
for ax, (tag, _, title), letter in zip(axes, PRODUCTS, 'ab'):
    ax.axvspan(0.55, 0.58, color='0.85', zorder=0)
    ax.axvline(DELTA_BASELINE, color='0.3', ls='--', lw=1.2, zorder=1)
    ax.axhline(0, color='0.4', ls=':', lw=1.0, zorder=1)
    for s in ['All'] + SECTORS:
        g = sw[(sw['product'] == tag) & (sw['sector'] == s)].sort_values('delta')
        if len(g) == 0:
            continue
        ax.plot(g['delta'], g['median_dh_i'], ls=LS[s], color=COL[s], lw=2.6 if s == 'All' else 1.8,
                marker=MRK[s], ms=6, mfc=MFC[s], mec=COL[s], mew=1.3, zorder=3 if s == 'All' else 2)
    ax.set_title(f'({letter}) {title}', loc='left', fontweight='bold')
    ax.set_xlabel('Penetration fraction used for CRYO2ICE snow depth')
    ax.set_xlim(0.38, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Median change in sea-ice thickness (m)')
handles = [Line2D([0], [0], ls=LS[s], color=COL[s], lw=2.6 if s == 'All' else 1.8, marker=MRK[s], ms=6,
                  mfc=MFC[s], mec=COL[s], mew=1.3, label=LAB[s]) for s in ['All'] + SECTORS]
handles += [Line2D([0], [0], color='0.3', ls='--', lw=1.2, label=f'Adopted value ({DELTA_BASELINE:.2f})'),
            Patch(facecolor='0.85', label='Empirical optimum (0.55–0.58)')]
axes[1].legend(handles=handles, loc='lower right', frameon=True, fancybox=False, edgecolor='black',
               framealpha=1, fontsize=11)
fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(f'{OUT_DIR}/R3_Fig_C2_04_dhi_delta_sweep_v10.{ext}')
plt.close(fig)
print('Saved R3_Fig_C2_04_dhi_delta_sweep_v10')
print(sw[np.isclose(sw['delta'], 0.40) | np.isclose(sw['delta'], 0.55) | np.isclose(sw['delta'], 0.70)
         | np.isclose(sw['delta'], 1.00)][['delta', 'product', 'sector', 'median_hs_c2i', 'median_dh_i']]
      .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('Output:', OUT_DIR)

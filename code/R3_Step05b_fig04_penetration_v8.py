#!/usr/bin/env python3
"""
R3_Step05b_fig04_penetration_v8.py — Figure 4 (penetration sensitivity) redrawn for Alex's comment 24
====================================================================================================
Alex: "legend is unconventional, and the Ross line is covered by the observed FYI line".
The FYI curve (87% of matchups, mostly Eastern Weddell and Ross) is almost identical to the
Ross curve, so they cannot share one panel. This version:
  (a) sectors and sub-sectors      (b) observed multi-year and first-year ice
  * legend inside each panel, framed, in plotting order (no external legend)
  * no per-point value labels, no title, no parameter string (all in the caption)
  * snow-density envelope at alpha 0.18 so it is visible but lines stay on top
  * adopted delta as a dashed line, empirical optimum 0.55-0.58 as a grey band
Input : {M3_OUTPUT}/R3_penetration_sensitivity_d070_v7.csv (Step 04 v7; unchanged)
Output: {M3_FIGURES}/R3_Fig04_penetration.{png,pdf}  (same name, so \\includegraphics is unchanged)
Run   : python R3_Step05b_fig04_penetration_v8.py   (seconds, login node)
"""
import os, sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'legend.fontsize': 11, 'font.family': 'sans-serif',
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
LAB = {'Weddell': 'Weddell Sea', 'Weddell_W': 'Western Weddell', 'Weddell_E': 'Eastern Weddell',
       'Ross': 'Ross Sea', 'MYI_obs': 'Observed multi-year ice', 'FYI_obs': 'Observed first-year ice'}
COL = dict(REGION_COLORS)
COL.update({'MYI_obs': '#CC79A7', 'FYI_obs': '#332288'})
MRK = {'Weddell': 'o', 'Weddell_W': 's', 'Weddell_E': '^', 'Ross': 'D', 'MYI_obs': 's', 'FYI_obs': 'o'}
PANELS = [('(a) Sectors and sub-sectors', ['Weddell_W', 'Weddell', 'Ross', 'Weddell_E']),
          ('(b) Observed sea-ice type', ['MYI_obs', 'FYI_obs'])]

pen = pd.read_csv(f'{M3_OUTPUT}/R3_penetration_sensitivity_d070_v7.csv')
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
for ax, (title, strata) in zip(axes, PANELS):
    ax.axvspan(0.55, 0.58, color='0.85', zorder=0)
    ax.axvline(DELTA_BASELINE, color='0.3', ls='--', lw=1.2, zorder=1)
    handles = []
    for r in strata:
        s = pen[pen.sector == r].sort_values('delta')
        if len(s) == 0:
            print('  no rows for', r)
            continue
        ax.fill_between(s.delta, s.mean_hi_rho_lo, s.mean_hi_rho_hi, color=COL[r], alpha=0.18, lw=0, zorder=2)
        h, = ax.plot(s.delta, s.mean_hi, '-', marker=MRK[r], ms=5, color=COL[r], lw=2.0, zorder=3, label=LAB[r])
        handles.append(h)
    handles += [Line2D([0], [0], color='0.3', ls='--', lw=1.2, label=f'Adopted value ({DELTA_BASELINE:.2f})'),
                Patch(facecolor='0.85', label='Empirical optimum (0.55–0.58)'),
                Patch(facecolor='0.6', alpha=0.35, label=f'$\\rho_s$ = {RHO_S_CENTRAL:.0f} ± {RHO_S_SIGMA:.0f} kg m$^{{-3}}$')]
    ax.legend(handles=handles, loc='upper left', ncol=2, fontsize=10, frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_xlabel('Ku-band penetration fraction')
    ax.set_xlim(0.38, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylim(0.8, 4.3)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('Mean sea-ice thickness (m)')
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'{M3_FIGURES}/R3_Fig04_penetration.{ext}')
print('saved', f'{M3_FIGURES}/R3_Fig04_penetration.png/pdf')

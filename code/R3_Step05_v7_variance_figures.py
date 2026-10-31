#!/usr/bin/env python3
"""
R3_Step05_v7_variance_figures.py — Figures 2, 3 and 3b from the v7 (mixture prior) outputs
==========================================================================================
Standalone replacement for the Figure 2 / 3 / 3b sections of R3_Step05_v6_figures.py,
which read v4 outputs and carry 'truncnorm' labels from earlier prior choices.
Figures 1, 4 and 5 are still produced by R3_Step05_v6_figures.py.

Inputs
  {M3_OUTPUT}/R3_ExpB_d070_v7.json            Step 02 v7 (shares by REGIONS_V6, interactions)
  {M3_OUTPUT}/R3_density_sweep_d070_v7.csv    Step 03 v7
  {M3_OUTPUT}/R3_delta_sweep_v7.csv           Step 03b v7
Outputs (same file names as before, so the LaTeX \\includegraphics lines are unchanged)
  {M3_FIGURES}/R3_Fig02_ExpB_truncnorm.{png,pdf}
  {M3_FIGURES}/R3_Fig03_sweep_rho_i.{png,pdf}
  {M3_FIGURES}/R3_Fig03b_sweep_delta.{png,pdf}

Colours: Okabe-Ito for parameters; config REGION_COLORS for sectors, with
MYI_obs / FYI_obs overridden to #CC79A7 / #999933 so no two strata share a colour.
Run: source ~/cryo2ice_env/bin/activate && python R3_Step05_v7_variance_figures.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

plt.rcParams.update({
    'font.size': 10, 'font.family': 'sans-serif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

PARAMS = ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta']
P_COL = {'fb_is2': '#56B4E9', 'fb_cs2': '#E69F00', 'rho_s': '#F0E442',
         'rho_i': '#009E73', 'delta': '#0072B2', 'interaction': '#7F7F7F'}
P_LAB = {'fb_is2': '$h_f^{\\mathrm{IS2}}$ (ICESat-2 freeboard)',
         'fb_cs2': '$h_f^{\\mathrm{CS2}}$ (CryoSat-2 freeboard)',
         'rho_s': '$\\rho_s$ (snow density)', 'rho_i': '$\\rho_i$ (sea-ice density)',
         'delta': '$\\delta$ (penetration)', 'interaction': 'interaction residual'}
STRATA = ['all'] + list(REGIONS_V6)
S_LAB = {'all': 'All', 'Weddell': 'Weddell Sea', 'Weddell_W': 'Western Weddell',
         'Weddell_E': 'Eastern Weddell', 'Ross': 'Ross Sea',
         'MYI_obs': 'Observed MYI', 'FYI_obs': 'Observed FYI'}
# Strata colours. The observed ice-type strata must not share colours with the
# sectors: W. Weddell is #882255 and E. Weddell is #44AA99 in the config, so
# MYI_obs / FYI_obs are set here explicitly (and in R3_config_v4 from v7.1).
S_COL = dict(REGION_COLORS, all=C_ALL)
S_COL.update({'MYI_obs': '#FF00FF', 'FYI_obs': '#000080'})
_clash = {s: c for s, c in S_COL.items() if list(S_COL.values()).count(c) > 1}
assert not _clash, f"duplicate stratum colours: {_clash}"
PANEL_STRATA = ['Weddell', 'Weddell_W', 'Weddell_E', 'Ross', 'MYI_obs', 'FYI_obs']

# ── Figure 2: variance decomposition, v7 ──────────────────────────────────────
print("Figure 2: variance decomposition (Step 02 v7)")
with open(f'{M3_OUTPUT}/R3_ExpB_d070_v7.json') as f:
    B = json.load(f)

cats = PARAMS + ['interaction']
x = np.arange(len(cats))
width = 0.8 / len(STRATA)
fig, ax = plt.subplots(figsize=(12.5, 6))
for i, s in enumerate(STRATA):
    vals = [B[p][s] for p in PARAMS]
    vals.append(100.0 - sum(vals))                        # residual, per stratum
    bars = ax.bar(x + (i - (len(STRATA) - 1) / 2) * width, vals, width,
                  label=S_LAB[s], color=S_COL.get(s, '#999'), alpha=0.85, edgecolor='white')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f'{v:.1f}',
                ha='center', va='bottom', fontsize=5.5, color=S_COL.get(s, '#333'), fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['ICESat-2\ntotal freeboard\n$h_f^{\\mathrm{IS2}}$', 'CryoSat-2\nradar freeboard\n$h_f^{\\mathrm{CS2}}$',
                    'Snow\ndensity\n$\\rho_s$', 'Sea-ice\ndensity\n$\\rho_i$', 'Ku-band\npenetration\n$\\delta$',
                    'Interaction\nresidual'], fontsize=9)
ax.set_ylabel('Fraction of simultaneous-perturbation variance (%)')
ax.set_ylim(0, 60)
ax.legend(fontsize=8, ncol=4, loc='upper left', frameon=False)
ax.grid(axis='y', alpha=0.2)
ax.set_title('Variance decomposition with the per-matchup mixture $\\rho_i$ prior '
             f'(recovery {B["recovery_pct"]:.1f}%, interaction {100 - B["recovery_pct"]:.1f}%)', fontsize=10)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'{M3_FIGURES}/R3_Fig02_ExpB_truncnorm.{ext}')
plt.close(fig)
print("  saved R3_Fig02_ExpB_truncnorm.png/pdf")


def sweep_panels(df, xcol, xlabel, title, fname, mark_x=None):
    """2x3 panels over PANEL_STRATA; five parameter lines + dashed interaction line."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, PANEL_STRATA):
        sub = df[df.region == s].sort_values(xcol)
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        for p in PARAMS:
            ax.plot(sub[xcol], sub[f'frac_{p}'], '-o', color=P_COL[p], label=P_LAB[p], ms=4, lw=1.4)
            for xv, yv in zip(sub[xcol], sub[f'frac_{p}']):
                if yv > 3:
                    ax.text(xv, yv + 1.0, f'{yv:.1f}', ha='center', va='bottom', fontsize=5, color=P_COL[p], clip_on=True)
        ax.plot(sub[xcol], sub['frac_interaction'], '--', color=P_COL['interaction'],
                label=P_LAB['interaction'], lw=1.2)
        if mark_x is not None:
            ax.axvline(mark_x, color='k', lw=0.6, ls=':', alpha=0.7)
        ax.set_title(S_LAB[s], fontsize=10)
        ax.set_ylim(-2, 85)
        ax.grid(True, alpha=0.25)
    for ax in axes[1, :]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel('Fraction of simultaneous variance (%)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    for ext in ('png', 'pdf'):
        fig.savefig(f'{M3_FIGURES}/{fname}.{ext}')
    plt.close(fig)
    print(f"  saved {fname}.png/pdf")


# ── Figure 3: sigma_rho_i sweep ─────────────────────────────────────────────
p = f'{M3_OUTPUT}/R3_density_sweep_d070_v7.csv'
if os.path.exists(p):
    print("Figure 3: sigma_rho_i sweep (Step 03 v7)")
    sweep_panels(pd.read_csv(p), 'rho_sigma',
                 'First year $\\sigma_{\\rho_i}$ (kg m$^{-3}$); multi-year spread scaled by 23.0/35.7',
                 'Sensitivity of the variance budget to sea-ice-density uncertainty, mixture prior '
                 f'($\\delta \\sim N({DELTA_BASELINE}, {DELTA_SIGMA})$, $\\rho_s = {RHO_S_CENTRAL:.0f} \\pm {RHO_S_SIGMA:.0f}$ kg m$^{{-3}}$)',
                 'R3_Fig03_sweep_rho_i', mark_x=RHO_I_FYI_SIGMA)
else:
    print(f"  {p} not found; skipping Figure 3")

# ── Figure 3b: sigma_delta sweep ────────────────────────────────────────────
p = f'{M3_OUTPUT}/R3_delta_sweep_v7.csv'
if os.path.exists(p):
    print("Figure 3b: sigma_delta sweep (Step 03b v7)")
    sweep_panels(pd.read_csv(p), 'delta_sigma',
                 'Penetration-fraction uncertainty $\\sigma_{\\delta}$',
                 'Sensitivity of the variance budget to penetration-fraction uncertainty, mixture prior '
                 f'($\\rho_i$: FYI ${RHO_I_FYI_MU} \\pm {RHO_I_FYI_SIGMA}$, MYI ${RHO_I_MYI_MU} \\pm {RHO_I_MYI_SIGMA}$ kg m$^{{-3}}$)',
                 'R3_Fig03b_sweep_delta', mark_x=DELTA_SIGMA)
else:
    print(f"  {p} not found; skipping Figure 3b")

print("Done.")

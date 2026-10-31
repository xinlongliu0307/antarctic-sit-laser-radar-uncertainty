#!/usr/bin/env python3
"""
R3_Step05_v7_figures.py — Figures 1, 4 and 5 from the v7 (mixture prior) outputs
=================================================================================
Replaces the Figure 1 / 4 / 5 sections of R3_Step05_v6_figures.py.
Figures 2, 3 and 3b are produced by R3_Step05_v7_variance_figures.py.

Changes from v6
  * reads R3_ExpA_d070_v7.npz (mixture prior) and R3_penetration_sensitivity_d070_v7.csv
  * Figure 1: 2x3 panels (four sectors + observed MYI / FYI); the deterministic
    central-value line uses the per-matchup mixture expected density, so it is
    consistent with the ensemble in every panel (the v6 'regional best-guess'
    workaround is no longer needed)
  * Figure 4: curves from Step 04 v7 (mixture central density), six strata
  * Figure 5: box plots of per-matchup MC means for six strata
  * no 'truncnorm' / 'eff ~ 880' labels; titles state the mixture prior
Output file names are unchanged so the LaTeX \\includegraphics lines still apply.

Run: source ~/cryo2ice_env/bin/activate && python R3_Step05_v7_figures.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif', 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'figure.dpi': 200,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

STRATA = ['Weddell', 'Weddell_W', 'Weddell_E', 'Ross', 'MYI_obs', 'FYI_obs']
S_LAB = {'Weddell': 'Weddell Sea', 'Weddell_W': 'Western Weddell', 'Weddell_E': 'Eastern Weddell',
         'Ross': 'Ross Sea', 'MYI_obs': 'Observed MYI (all sectors)', 'FYI_obs': 'Observed FYI (all sectors)'}
S_COL = dict(REGION_COLORS)
S_COL.update({'MYI_obs': '#FF00FF', 'FYI_obs': '#000080'})   # distinct from the sector colours

# ── Data ────────────────────────────────────────────────────────────────
df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_icetype.pkl')
if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df['lon'].values)
sectors, subsectors, icetype = df['sector'].values, df['subsector'].values, df['icetype'].values
masks = {r: get_region_mask_v6(sectors, subsectors, icetype, r) for r in STRATA}

A = np.load(f'{M3_OUTPUT}/R3_ExpA_d070_v7.npz', allow_pickle=True)
hi_mean, hist_bins = A['hi_mean'], A['hist_bins']
assert len(hi_mean) == len(df), "ExpA v7 and ice-type pickle have different lengths"
mc_hists = {r: A[f'mc_hist_{r}'] for r in STRATA if f'mc_hist_{r}' in A}

rho_i_c = mixture_rho_i_central(df['myi_frac'].values)
hi_central = compute_ice_thickness(df['is2_fb_wm'].values, df['hs_d070'].values,
                                   RHO_S_CENTRAL, rho_i_c, RHO_W_CENTRAL)

# ── Figure 1: all MC draws vs deterministic central-value retrieval ─────
print("Figure 1: MC draws vs central-value retrieval (2x3)")
bc = (hist_bins[:-1] + hist_bins[1:]) / 2
bw = hist_bins[1] - hist_bins[0]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for ax, r in zip(axes.flat, STRATA):
    m = masks[r]
    col = S_COL[r]
    if r in mc_hists:
        c = mc_hists[r].astype(float)
        ax.bar(bc, c / (c.sum() * bw), width=bw, color=col, alpha=0.4, edgecolor='none',
               label=f'All MC draws (N={c.sum():,.0f})')
        mc_m = np.average(bc, weights=c)
        ax.axvline(mc_m, color=col, ls='--', lw=1.2, label=f'MC mean = {mc_m:.2f} m')
    hc = hi_central[m]
    hc = hc[np.isfinite(hc)]
    ax.hist(hc, bins=hist_bins, color='black', density=True, histtype='step', lw=1.6,
            label=f'Central value (n={len(hc):,}; $\\bar\\rho_i$={rho_i_c[m].mean():.0f})')
    ax.axvline(np.mean(hc), color='black', ls=':', lw=1.0, label=f'Central mean = {np.mean(hc):.2f} m')
    ax.axvline(0, color='grey', ls=':', lw=0.6, alpha=0.5)
    ax.set_xlim(-2, 8)
    ax.set_xlabel('Sea-ice thickness $h_i$ (m)')
    ax.set_ylabel('Probability density')
    ax.set_title(f'{S_LAB[r]} ({m.sum():,} matchups)', fontsize=10)
    ax.legend(fontsize=6.5, loc='upper right')
fig.suptitle(f'Monte Carlo sea-ice-thickness distributions with the per-matchup mixture $\\rho_i$ prior '
             f'({N_MC:,} draws per matchup; $\\delta \\sim N({DELTA_BASELINE},{DELTA_SIGMA})$, '
             f'$\\rho_s = {RHO_S_CENTRAL:.0f} \\pm {RHO_S_SIGMA:.0f}$ kg m$^{{-3}}$, '
             f'$\\rho_w = {RHO_W_CENTRAL:.0f}$ kg m$^{{-3}}$ fixed)', fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ('png', 'pdf'):
    fig.savefig(f'{M3_FIGURES}/R3_Fig01_ExpA_regions.{ext}')
plt.close(fig)
print("  saved R3_Fig01_ExpA_regions.png/pdf")

# ── Figure 4: mean h_i versus delta ────────────────────────────────────
print("Figure 4: penetration sensitivity (Step 04 v7)")
pen = pd.read_csv(f'{M3_OUTPUT}/R3_penetration_sensitivity_d070_v7.csv')
fig, ax = plt.subplots(figsize=(10, 6.5))
handles = []
for r in STRATA:
    s = pen[pen.sector == r].sort_values('delta')
    if len(s) == 0:
        continue
    ls = '--' if r.endswith('_obs') else '-'
    h, = ax.plot(s.delta, s.mean_hi, ls, marker='o', color=S_COL[r], lw=2.0, ms=4,
                 label=f'{S_LAB[r]} ({int(s.n.iloc[0]):,})')
    ax.fill_between(s.delta, s.mean_hi_rho_lo, s.mean_hi_rho_hi, color=S_COL[r], alpha=0.08)
    if not r.endswith('_obs'):
        for x, y in zip(s.delta, s.mean_hi):
            ax.text(x, y + 0.02, f'{y:.2f}', ha='center', va='bottom', fontsize=5.5, color=S_COL[r], fontweight='bold')
    handles.append((s.mean_hi.iloc[-1], h))
ax.axvline(1.0, color='grey', ls=':', lw=0.8)
refs = [ax.axvline(0.70, color='black', ls='--', lw=1.2, label='$\\delta=0.70$ (adopted)'),
        ax.axvline(0.55, color='#CC79A7', ls='--', lw=1.2, label='$\\delta^*_{\\mathrm{med}} \\approx 0.55$'),
        ax.axvline(0.58, color='#E69F00', ls='-.', lw=1.2, label='$\\delta^*_{\\mathrm{mean}} \\approx 0.58$')]
handles.sort(key=lambda t: t[0], reverse=True)
ax.legend(handles=[h for _, h in handles] + refs, fontsize=8.5, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True)
ax.set_xlabel('Ku-band penetration fraction $\\delta$', fontsize=11)
ax.set_ylabel('Mean sea-ice thickness $h_i$ (m)', fontsize=11)
ax.set_title('Sensitivity of mean sea-ice thickness to the assumed penetration fraction\n'
             f'($\\rho_i$ = mixture expected value per matchup; shading = $\\rho_s \\pm {RHO_S_SIGMA:.0f}$ kg m$^{{-3}}$; '
             f'$\\rho_w = {RHO_W_CENTRAL:.0f}$ kg m$^{{-3}}$ fixed)', fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'{M3_FIGURES}/R3_Fig04_penetration.{ext}')
plt.close(fig)
print("  saved R3_Fig04_penetration.png/pdf")

# ── Figure 5: box plots of per-matchup MC means ────────────────────────
print("Figure 5: box plots (six strata)")
fig, ax = plt.subplots(figsize=(10, 5.5))
data, labels, cols = [], [], []
for r in STRATA:
    v = hi_mean[masks[r]]
    data.append(v[(v > -2) & (v < 8)])
    labels.append(S_LAB[r].replace(' (all sectors)', '\n(all sectors)'))
    cols.append(S_COL[r])
bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showfliers=False, whis=1.5,
                medianprops=dict(color='black', lw=1.5))
for p, c in zip(bp['boxes'], cols):
    p.set_facecolor(c)
    p.set_alpha(0.6)
mean_h = None
for i, r in enumerate(STRATA):
    h = ax.scatter(i + 1, hi_mean[masks[r]].mean(), color='black', marker='D', s=40, zorder=5)
    mean_h = mean_h or h
ax.legend(handles=[mean_h, Line2D([0], [0], color='black', lw=1.5), Line2D([0], [0], color='black', lw=0.8)],
          labels=['Mean', 'Median', 'Whiskers: 1.5 x IQR'], fontsize=9, loc='upper right')
ax.set_ylabel('Sea-ice thickness $h_i$ (m)', fontsize=12)
ax.set_xlabel('Sector / sub-sector / observed ice type', fontsize=11)
ax.set_title('Per-matchup Monte Carlo mean sea-ice thickness, mixture $\\rho_i$ prior', fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', labelsize=8.5)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'{M3_FIGURES}/R3_Fig05_subsectors.{ext}')
plt.close(fig)
print("  saved R3_Fig05_subsectors.png/pdf")
print("Done.")

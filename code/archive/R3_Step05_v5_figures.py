#!/usr/bin/env python3
"""
R3_Step05_v5_figures.py — Publication-quality figures (finalised consolidation)
================================================================================
This finalised version consolidates Alex's 14 April 2026 meeting instructions
with the 27 April 2026 meeting decisions and the dual sign-off from Alex and
Stuart on the 940 kg/m³ upper cap (28 April 2026).

Meeting-driven improvements (retained from earlier v5):
  - Figure 3: 2x2 layout with single shared legend at bottom
  - Figure 3b: NEW delta sensitivity sweep (reads R3_delta_sweep_v1.csv)
  - Figure 5: 1x2 layout (panel c table removed)
  - Shape robustness figure: inset showing the two rho_i distribution PDFs

Dual sign-off integration (28 April 2026):
  - All figure titles and captions reference rho_i Gaussian with mu=916.7,
    sigma=35.7, 940 kg/m^3 upper cap (Alexandrov 2010 + Timco & Frederking
    1996)
  - Shape robustness uses two shapes only: Gaussian and Skew-Normal
  - Suptitle no longer references the obsolete RHO_I_LOWER constant
  - Inset distribution PDFs use the capped Gaussian and skew-normal
    formulations matching what Step 02b actually samples

Run: source ~/cryo2ice_env/bin/activate && python R3_Step05_v5_figures.py
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, norm as norm_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

plt.rcParams.update({
    'font.size': 10, 'font.family': 'sans-serif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

RHO_I_C = RHO_I_CONFIGURED_MU   # 916.7
RHO_I_S = RHO_I_SIGMA_DEFAULT   # 35.7

# Effective mean under the capped Gaussian (verified by Step 02 output).
# Configured 916.7, realised approximately 911.2 kg/m^3 because the 940 cap
# clips approximately 26 percent of the upper tail.
RHO_I_EFF_MEAN = 911

COL = {
    'Weddell':   '#0072B2', 'Weddell_W': '#882255',
    'Weddell_E': '#44AA99', 'Ross':      '#D55E00', 'All': '#333333',
}

t0 = time.time()
print("=" * 70)
print("R3 Step 05 v5: Generate All Figures")
print("  *** Dual sign-off (Alex + Stuart, 28 April 2026):")
print("      Alexandrov 2010 FYI + 940 kg/m^3 upper cap ***")
print("=" * 70)

# Load data
df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_verified.pkl')
if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df['lon'].values)
sectors = df['sector'].values
subsectors = df['subsector'].values

for v in ['v4', 'v3', 'v2']:
    p = f'{M3_OUTPUT}/R3_ExpA_d070_{v}.npz'
    if os.path.exists(p):
        expA = np.load(p, allow_pickle=True)
        print(f"  ExpA: {p}")
        break

hi_mean = expA['hi_mean']
hi_std = expA['hi_std']
ci_width = expA['ci_width']
hist_bins = expA['hist_bins']

mc_hists = {}
for r in REGIONS:
    key = f'mc_hist_{r}'
    if key in expA:
        mc_hists[r] = expA[key]
        print(f"  MC hist {r}: {mc_hists[r].sum():,} draws")

for v in ['v4', 'v3_all_dists']:
    p = f'{M3_OUTPUT}/R3_ExpB_d070_{v}.json'
    if os.path.exists(p):
        with open(p) as f:
            results_B_raw = json.load(f)
        print(f"  ExpB: {p}")
        break

results_B = (results_B_raw.get('truncnorm') or
             results_B_raw.get('gaussian') or results_B_raw)

for v in ['v4_coupled', 'v3_coupled']:
    p = f'{M3_OUTPUT}/R3_density_sweep_d070_{v}.csv'
    if os.path.exists(p):
        df_sweep = pd.read_csv(p)
        print(f"  Sweep (rho_i): {p}")
        break

delta_sweep_path = f'{M3_OUTPUT}/R3_delta_sweep_v1.csv'
if os.path.exists(delta_sweep_path):
    df_delta_sweep = pd.read_csv(delta_sweep_path)
    print(f"  Sweep (delta): {delta_sweep_path}")
else:
    df_delta_sweep = None
    print(f"  WARNING: delta sweep CSV not found at {delta_sweep_path}")

for v in ['v3', 'v2']:
    p = f'{M3_OUTPUT}/R3_penetration_sensitivity_d070_{v}.csv'
    if os.path.exists(p):
        df_pen = pd.read_csv(p)
        print(f"  Penetration: {p}")
        break

region_col = 'region' if 'region' in df_sweep.columns else 'sector'

is2_fb_raw = df['is2_fb_wm'].values
hs_obs = df['hs_d070'].values
hi_bestguess = compute_ice_thickness(is2_fb_raw, hs_obs,
                                      RHO_S_CENTRAL, RHO_I_C, RHO_W_CENTRAL)

mc_draw_totals = {}
for r in REGIONS:
    mask = get_region_mask(sectors, subsectors, r)
    mc_draw_totals[r] = int(mask.sum()) * N_MC


# ===================================================================
# FIGURE 1: ALL MC DRAWS histogram vs best-guess histogram
# ===================================================================
print("\n  Figure 1: ALL MC draws vs best-guess...")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

bin_centres = (hist_bins[:-1] + hist_bins[1:]) / 2
bin_width = hist_bins[1] - hist_bins[0]

for ax, region in zip(axes, REGIONS):
    mask = get_region_mask(sectors, subsectors, region)
    color = COL[region]
    n_segments = mask.sum()
    n_mc_total = mc_draw_totals[region]

    if region in mc_hists:
        counts = mc_hists[region].astype(float)
        total_draws = counts.sum()
        density = counts / (total_draws * bin_width)
        ax.bar(bin_centres, density, width=bin_width, color=color, alpha=0.4,
               edgecolor='none',
               label=f'All MC draws (N={total_draws:,.0f})')

    hi_bg = hi_bestguess[mask]
    hi_bg_valid = hi_bg[np.isfinite(hi_bg)]
    ax.hist(hi_bg_valid, bins=hist_bins, color='black', alpha=0.9,
            edgecolor='black', density=True, histtype='step', linewidth=1.8,
            label=f'Best guess (n={len(hi_bg_valid):,} segments)')

    if region in mc_hists:
        mc_mean = np.average(bin_centres, weights=mc_hists[region])
        ax.axvline(mc_mean, color=color, ls='--', lw=1.2,
                   label=f'MC mean={mc_mean:.2f} m')
    bg_mean = np.nanmean(hi_bg_valid)
    ax.axvline(bg_mean, color='black', ls=':', lw=1.0,
               label=f'Best guess mean={bg_mean:.2f} m')
    ax.axvline(0, color='grey', ls=':', lw=0.6, alpha=0.5)

    ax.set_xlabel('Sea-ice thickness $h_i$ (m)')
    ax.set_ylabel('Probability density')
    ax.set_title(f'{REGION_LABELS[region]} ({n_segments:,} segments, '
                 f'{n_mc_total:,.0f} MC draws)')
    ax.set_xlim(-2, 8)
    ax.legend(fontsize=6.5, loc='upper right')

fig.suptitle(f'All MC Draws vs Best-Guess Sea-Ice Thickness\n'
             f'(MC: 5 params perturbed, {N_MC:,} draws/segment; '
             f'$\\delta \\sim N$({DELTA_BASELINE},{DELTA_SIGMA}), '
             f'$\\rho_i$ Gaussian $\\mu$={RHO_I_C}, $\\sigma$={RHO_I_S} kg m$^{{-3}}$ '
             f'with cap at {int(RHO_I_UPPER_CAP)}, '
             f'$\\rho_s$={RHO_S_CENTRAL:.0f} kg m$^{{-3}}$, '
             f'$\\rho_w$={RHO_W_CENTRAL:.0f} kg m$^{{-3}}$ fixed)',
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f'{M3_FIGURES}/R3_Fig01_ExpA_regions.png')
fig.savefig(f'{M3_FIGURES}/R3_Fig01_ExpA_regions.pdf')
print(f"    Saved: R3_Fig01_ExpA_regions.png/pdf")
plt.close()


# ===================================================================
# FIGURE 2: Variance decomposition bar chart
# ===================================================================
print("  Figure 2: Variance decomposition...")

fig, ax = plt.subplots(figsize=(12, 6))

params_plot = [p for p in ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta'] if p in results_B]
x = np.arange(len(params_plot))
width = 0.14

fullname_map = {
    'fb_is2': 'ICESat-2\ntotal freeboard\n$f_b^{IS2}$',
    'fb_cs2': 'CryoSat-2\nradar freeboard\n$f_b^{CS2}$',
    'rho_s':  'Snow\ndensity\n$\\rho_s$',
    'rho_i':  'Sea-ice\ndensity\n$\\rho_i$',
    'delta':  'Ku-band\npenetration\n$\\delta$',
}

region_order = ['all'] + list(REGIONS)
region_labels_bar = {
    'all': 'All', 'Weddell': 'Weddell Sea', 'Weddell_W': 'Western Weddell',
    'Weddell_E': 'Eastern Weddell', 'Ross': 'Ross Sea',
}

for i, region in enumerate(region_order):
    vals = [results_B.get(p, {}).get(region, 0) for p in params_plot]
    if all(v == 0 for v in vals):
        continue
    color = COL.get(region, COL.get('All', '#999'))
    label = region_labels_bar.get(region, region)
    bars = ax.bar(x + (i - 2) * width, vals, width, label=label, color=color,
                  alpha=0.75, edgecolor='white')
    for bar, val in zip(bars, vals):
        if val > 0.2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=6,
                    color=color, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([fullname_map.get(p, p) for p in params_plot], fontsize=9)
ax.set_ylabel('Variance contribution (%)', fontsize=11)
ax.set_title(f'Variance Decomposition (Alexandrov FYI Gaussian, 940 cap)\n'
             f'$\\rho_i$ $\\mu$={RHO_I_C}, $\\sigma$={RHO_I_S} kg m$^{{-3}}$, '
             f'cap at {int(RHO_I_UPPER_CAP)}; '
             f'$\\rho_s$={RHO_S_CENTRAL:.0f}$\\pm${int(RHO_S_SIGMA)} kg m$^{{-3}}$ unclipped, '
             f'$\\rho_w$={RHO_W_CENTRAL:.0f} kg m$^{{-3}}$ fixed, '
             f'$\\delta \\sim N$({DELTA_BASELINE},{DELTA_SIGMA})',
             fontsize=10)
ax.legend(fontsize=8, ncol=3, loc='upper left')
ax.grid(axis='y', alpha=0.2)
ax.set_ylim(0, 75)
fig.tight_layout()
fig.savefig(f'{M3_FIGURES}/R3_Fig02_ExpB_truncnorm.png')
fig.savefig(f'{M3_FIGURES}/R3_Fig02_ExpB_truncnorm.pdf')
print(f"    Saved: R3_Fig02_ExpB_truncnorm.png/pdf")
plt.close()


# ===================================================================
# FIGURE 3: rho_i density sweep — 2x2 layout
# ===================================================================
print(f"  Figure 3: rho_i density sweep (2x2 layout)...")

param_cols = [c.replace('frac_', '') for c in df_sweep.columns
              if c.startswith('frac_') and c != 'frac_fb_combined']

colors_sweep = {
    'fb_is2': '#56B4E9', 'fb_cs2': '#E69F00',
    'rho_s': '#F0E442', 'rho_i': '#009E73', 'delta': '#0072B2',
}
labels_sweep_full = {
    'fb_is2': '$f_b^{IS2}$ (IS2 freeboard)',
    'fb_cs2': '$f_b^{CS2}$ (CS2 freeboard)',
    'rho_s':  '$\\rho_s$ (snow density)',
    'rho_i':  '$\\rho_i$ (sea-ice density)',
    'delta':  '$\\delta$ (penetration)',
}

regions_in_sweep = df_sweep[region_col].unique()
plot_regions_sweep = [r for r in REGIONS if r in regions_in_sweep]

# Find the configured central value in the sweep CSV (handle 916.7 as 917)
sweep_centrals = sorted(df_sweep['rho_central'].unique())
target_mu = RHO_I_CONFIGURED_MU
sweep_mu = min(sweep_centrals, key=lambda c: abs(c - target_mu))

# Determine which shape label to use for filtering
shape_filter = 'gaussian' if 'gaussian' in df_sweep['dist_shape'].unique() else 'truncnorm'

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True, sharex=True)
axes_flat = axes.flatten()

for ax, region in zip(axes_flat, plot_regions_sweep):
    sub = df_sweep[(df_sweep['dist_shape'] == shape_filter) &
                   (df_sweep['rho_central'] == sweep_mu) &
                   (df_sweep[region_col] == region)].sort_values('rho_sigma')
    for param in param_cols:
        col_name = f'frac_{param}'
        if col_name in sub.columns and sub[col_name].abs().max() > 0.1:
            color_p = colors_sweep.get(param, '#999')
            vals = sub[col_name].values
            sigmas = sub['rho_sigma'].values
            ax.plot(sigmas, vals, '-o', color=color_p,
                    label=labels_sweep_full.get(param, param),
                    markersize=5, lw=1.5)
            for sx, vy in zip(sigmas, vals):
                ax.text(sx, vy + 1.2, f'{vy:.1f}', ha='center', va='bottom',
                        fontsize=5.5, color=color_p, fontweight='bold')
    ax.set_title(f'{REGION_LABELS.get(region, region)}', fontsize=11)
    ax.set_ylim(-2, 85)
    ax.grid(True, alpha=0.3)

for ax in axes[1, :]:
    ax.set_xlabel('Sea-ice density uncertainty $\\sigma_{\\rho_i}$ (kg m$^{-3}$)', fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel('Fractional variance contribution (%)', fontsize=10)

handles, labels_leg = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc='lower center', ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(f'Sensitivity of SIT variance to $\\sigma_{{\\rho_i}}$\n'
             f'($\\rho_i$ Gaussian $\\mu$={int(round(sweep_mu))} kg m$^{{-3}}$, '
             f'940 kg m$^{{-3}}$ upper cap; '
             f'$\\delta \\sim N$({DELTA_BASELINE},{DELTA_SIGMA}), '
             f'$\\rho_s$={RHO_S_CENTRAL}$\\pm${int(RHO_S_SIGMA)} kg m$^{{-3}}$, '
             f'$\\rho_w$={RHO_W_CENTRAL:.0f} kg m$^{{-3}}$ fixed)',
             fontsize=10, y=1.02)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(f'{M3_FIGURES}/R3_Fig03_sweep_rho_i.png')
fig.savefig(f'{M3_FIGURES}/R3_Fig03_sweep_rho_i.pdf')
print(f"    Saved: R3_Fig03_sweep_rho_i.png/pdf")
plt.close()


# ===================================================================
# FIGURE 3b: delta sensitivity sweep — 2x2 layout
# ===================================================================
if df_delta_sweep is not None:
    print(f"  Figure 3b: delta sensitivity sweep (2x2 layout)...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True, sharex=True)
    axes_flat = axes.flatten()
    delta_region_col = 'region'
    delta_regions = [r for r in REGIONS if r in df_delta_sweep[delta_region_col].unique()]
    for ax, region in zip(axes_flat, delta_regions):
        sub = df_delta_sweep[df_delta_sweep[delta_region_col] == region].sort_values('delta_sigma')
        for param in param_cols:
            col_name = f'frac_{param}'
            if col_name in sub.columns and sub[col_name].abs().max() > 0.1:
                color_p = colors_sweep.get(param, '#999')
                vals = sub[col_name].values
                sigmas = sub['delta_sigma'].values
                ax.plot(sigmas, vals, '-o', color=color_p,
                        label=labels_sweep_full.get(param, param),
                        markersize=5, lw=1.5)
                for sx, vy in zip(sigmas, vals):
                    ax.text(sx, vy + 1.2, f'{vy:.1f}', ha='center', va='bottom',
                            fontsize=5.5, color=color_p, fontweight='bold')
        ax.set_title(f'{REGION_LABELS.get(region, region)}', fontsize=11)
        ax.set_ylim(-2, 85)
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel('Penetration factor uncertainty $\\sigma_{\\delta}$', fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel('Fractional variance contribution (%)', fontsize=10)
    handles, labels_leg = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'Sensitivity of SIT variance to $\\sigma_{{\\delta}}$\n'
                 f'($\\rho_i$ Gaussian $\\mu$={RHO_I_C}, $\\sigma$={RHO_I_S} kg m$^{{-3}}$, '
                 f'940 kg m$^{{-3}}$ cap; '
                 f'$\\rho_s$={RHO_S_CENTRAL}$\\pm${int(RHO_S_SIGMA)} kg m$^{{-3}}$, '
                 f'$\\rho_w$={RHO_W_CENTRAL:.0f} kg m$^{{-3}}$ fixed)',
                 fontsize=10, y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(f'{M3_FIGURES}/R3_Fig03b_sweep_delta.png')
    fig.savefig(f'{M3_FIGURES}/R3_Fig03b_sweep_delta.pdf')
    print(f"    Saved: R3_Fig03b_sweep_delta.png/pdf")
    plt.close()
else:
    print("  SKIPPING Figure 3b: delta sweep data not available")


# ===================================================================
# FIGURE 4: Penetration sensitivity — 4 regions
# ===================================================================
print("  Figure 4: Penetration sensitivity...")

fig, ax = plt.subplots(figsize=(10, 6.5))

delta_f_arr = df['delta_f'].values
rho_s_c = RHO_S_CENTRAL
rho_w_c = RHO_W_CENTRAL
delta_range = np.arange(0.40, 1.01, 0.05)

fig4_regions = ['Weddell', 'Weddell_W', 'Weddell_E', 'Ross']
fig4_styles = {
    'Weddell':   {'ls': '-', 'lw': 2.5, 'ms': 5, 'af': 0.10},
    'Weddell_W': {'ls': '-', 'lw': 2.0, 'ms': 4, 'af': 0.08},
    'Weddell_E': {'ls': '-', 'lw': 2.0, 'ms': 4, 'af': 0.08},
    'Ross':      {'ls': '-', 'lw': 2.5, 'ms': 5, 'af': 0.10},
}

for region in fig4_regions:
    mask = get_region_mask(sectors, subsectors, region)
    if mask.sum() < 10:
        continue
    color = COL[region]
    style = fig4_styles[region]
    n_seg = mask.sum()
    mh, mh_lo, mh_hi = [], [], []
    for dv in delta_range:
        hs_d = delta_f_arr[mask] / (dv * ETA_S)
        mh.append(np.mean(compute_ice_thickness(is2_fb_raw[mask], hs_d, rho_s_c, RHO_I_C, rho_w_c)))
        mh_lo.append(np.mean(compute_ice_thickness(is2_fb_raw[mask], hs_d, rho_s_c - RHO_S_SIGMA, RHO_I_C, rho_w_c)))
        mh_hi.append(np.mean(compute_ice_thickness(is2_fb_raw[mask], hs_d, rho_s_c + RHO_S_SIGMA, RHO_I_C, rho_w_c)))
    ax.plot(delta_range, mh, f'{style["ls"]}o', color=color, lw=style['lw'],
            markersize=style['ms'],
            label=f'{REGION_LABELS[region]} ({n_seg:,} seg.)')
    ax.fill_between(delta_range, mh_lo, mh_hi, color=color, alpha=style['af'])
    for dx, hv in zip(delta_range, mh):
        ax.text(dx, hv + 0.02, f'{hv:.2f}', ha='center', va='bottom',
                fontsize=5.5, color=color, fontweight='bold')

ax.axvline(1.0, color='grey', ls=':', lw=0.8)
ax.axvline(0.70, color='black', ls='--', lw=1.2, label='$\\delta=0.70$ (baseline)')
ax.axvline(0.55, color='#CC79A7', ls='--', lw=1.2, label='$\\delta^*_{med}\\approx0.55$')
ax.axvline(0.58, color='#E69F00', ls='-.', lw=1.2, label='$\\delta^*_{mean}\\approx0.58$')

ax.set_xlabel('Ku-band penetration fraction $\\delta$', fontsize=11)
ax.set_ylabel('Sea-ice thickness $h_i$ (m)', fontsize=11)
ax.set_title(f'Sensitivity of Sea-Ice Thickness to Ku-band Penetration\n'
             f'($\\rho_i = {RHO_I_C}$ kg m$^{{-3}}$, '
             f'$\\rho_w$={RHO_W_CENTRAL:.0f} kg m$^{{-3}}$ fixed, '
             f'shading=$\\rho_s\\pm${RHO_S_SIGMA:.0f} kg m$^{{-3}}$)')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{M3_FIGURES}/R3_Fig04_penetration.png')
fig.savefig(f'{M3_FIGURES}/R3_Fig04_penetration.pdf')
print(f"    Saved: R3_Fig04_penetration.png/pdf")
plt.close()


# ===================================================================
# FIGURE 5: Sub-sector — 1x2 layout
# ===================================================================
print("  Figure 5: Sub-sector summary (1x2 layout, table removed)...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
plot_regions_fig5 = ['Weddell', 'Weddell_W', 'Weddell_E', 'Ross']

ax = axes[0]
for region in plot_regions_fig5:
    mask = get_region_mask(sectors, subsectors, region)
    color = COL[region]
    alpha_hist = 0.15 if region == 'Weddell' else 0.3
    n_seg = mask.sum()
    n_mc_total = mc_draw_totals[region]
    if region in mc_hists:
        counts = mc_hists[region].astype(float)
        total_draws = counts.sum()
        density_vals = counts / (total_draws * bin_width)
        ax.bar(bin_centres, density_vals, width=bin_width, color=color, alpha=alpha_hist,
               label=f'{REGION_LABELS[region]} ({total_draws:,.0f} MC draws)',
               edgecolor='none')
    else:
        hi_mc_valid = hi_mean[mask][np.isfinite(hi_mean[mask])]
        ax.hist(hi_mc_valid, bins=60, range=(-1, 6), color=color, alpha=alpha_hist,
                edgecolor='none', density=True,
                label=f'{REGION_LABELS[region]} ({n_mc_total:,.0f} MC draws)')
    hi_bg = hi_bestguess[mask]
    hi_bg_valid = hi_bg[np.isfinite(hi_bg)]
    lw_step = 1.2 if region == 'Weddell' else 1.8
    ls_step = '--' if region == 'Weddell' else '-'
    ax.hist(hi_bg_valid, bins=hist_bins, color=color, alpha=0.9,
            edgecolor=color, density=True, histtype='step',
            linewidth=lw_step, linestyle=ls_step)
ax.set_xlabel('Sea-ice thickness $h_i$ (m)')
ax.set_ylabel('Probability density')
ax.set_title('(a) All MC draws by sector\n(shaded = MC draws, outline = best guess)')
ax.legend(fontsize=6, loc='upper right')
ax.set_xlim(-2, 8)

ax = axes[1]
data_boxes, labels_boxes, colors_boxes = [], [], []
for region in plot_regions_fig5:
    mask = get_region_mask(sectors, subsectors, region)
    hi_r = hi_mean[mask]
    hi_clipped = hi_r[(hi_r > -2) & (hi_r < 8)]
    data_boxes.append(hi_clipped)
    labels_boxes.append(REGION_LABELS[region])
    colors_boxes.append(COL[region])

bp = ax.boxplot(data_boxes, labels=labels_boxes, patch_artist=True, widths=0.6,
                showfliers=False, medianprops=dict(color='black', lw=1.5))
for patch, color in zip(bp['boxes'], colors_boxes):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

from matplotlib.lines import Line2D
mean_handle = None
for i, region in enumerate(plot_regions_fig5):
    mask = get_region_mask(sectors, subsectors, region)
    h = ax.scatter(i + 1, hi_mean[mask].mean(), color='black', marker='D',
                   s=30, zorder=5, label='Mean' if i == 0 else None)
    if i == 0:
        mean_handle = h
median_handle = Line2D([0], [0], color='black', lw=1.5, label='Median')
ax.legend(handles=[mean_handle, median_handle], fontsize=8, loc='upper right')
ax.set_ylabel('Sea-ice thickness $h_i$ (m)')
ax.set_title('(b) Sea-ice thickness by sector\n(per-segment MC means)')
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=15)

fig.suptitle('Sector and Sub-Sector Analysis\n'
             f'($\\rho_i$ Gaussian $\\mu$={RHO_I_C}, $\\sigma$={RHO_I_S} kg m$^{{-3}}$, '
             f'940 cap; $\\rho_s$ unclipped, '
             f'$\\delta \\sim N$({DELTA_BASELINE},{DELTA_SIGMA}), '
             f'$\\rho_w$={RHO_W_CENTRAL:.0f} kg m$^{{-3}}$ fixed)',
             fontsize=10, y=1.04)
fig.tight_layout()
fig.savefig(f'{M3_FIGURES}/R3_Fig05_subsectors.png')
fig.savefig(f'{M3_FIGURES}/R3_Fig05_subsectors.pdf')
print(f"    Saved: R3_Fig05_subsectors.png/pdf")
plt.close()


# ===================================================================
# SHAPE ROBUSTNESS FIGURE — two shapes, with distribution-PDF inset
# *** Suptitle FIXED: removed obsolete RHO_I_LOWER reference ***
# ===================================================================
print("  Shape robustness figure (with inset distributions)...")
shape_json_paths = [
    f'{M3_OUTPUT}/R3_shape_robustness_v5.json',
    f'{M3_OUTPUT}/R3_shape_robustness_v4.json',
]
shape_json_path = None
for p in shape_json_paths:
    if os.path.exists(p):
        shape_json_path = p
        break

if shape_json_path is not None:
    with open(shape_json_path) as f:
        all_shape_results = json.load(f)
    print(f"    Loaded: {shape_json_path}")

    # Determine which shape keys are present in the JSON
    available_shapes = list(all_shape_results.keys())
    SHAPES = [s for s in ['gaussian', 'skewnorm'] if s in available_shapes]

    SHAPE_LABELS = {
        'gaussian':  'Gaussian',
        'skewnorm':  'Skew-Normal ($\\alpha$=$-$2)',
    }
    shape_colors = {
        'gaussian':  '#0072B2',
        'skewnorm':  '#882255',
    }
    params_rob = ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta']
    param_labels_rob = {
        'fb_is2': '$f_b^{IS2}$', 'fb_cs2': '$f_b^{CS2}$',
        'rho_s': '$\\rho_s$', 'rho_i': '$\\rho_i$', 'delta': '$\\delta$',
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a): All regions grouped bar chart
    ax = axes[0]
    x = np.arange(len(params_rob))
    width_r = 0.35
    for i, shape in enumerate(SHAPES):
        if shape not in all_shape_results or 'all' not in all_shape_results[shape]:
            continue
        vals = [all_shape_results[shape]['all'].get(p, 0) for p in params_rob]
        bars = ax.bar(x + (i - 0.5) * width_r, vals, width_r,
                      label=SHAPE_LABELS[shape], color=shape_colors[shape],
                      alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 1.0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                        color=shape_colors[shape], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([param_labels_rob[p] for p in params_rob], fontsize=12)
    ax.set_ylabel('Variance contribution (%)', fontsize=11)
    ax.set_title('(a) All regions', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylim(0, 90)

    # Panel (b): rho_i contribution by region
    ax = axes[1]
    regions_plot_rob = REGIONS
    x_r = np.arange(len(regions_plot_rob))
    for i, shape in enumerate(SHAPES):
        if shape not in all_shape_results:
            continue
        vals = []
        for region in regions_plot_rob:
            if region in all_shape_results[shape]:
                vals.append(all_shape_results[shape][region].get('rho_i', 0))
            else:
                vals.append(0)
        bars = ax.bar(x_r + (i - 0.5) * width_r, vals, width_r,
                      label=SHAPE_LABELS[shape], color=shape_colors[shape],
                      alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 1.0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                        color=shape_colors[shape], fontweight='bold')
    ax.set_xticks(x_r)
    ax.set_xticklabels([REGION_LABELS[r] for r in regions_plot_rob], fontsize=10)
    ax.set_ylabel('$\\rho_i$ variance contribution (%)', fontsize=11)
    ax.set_title('(b) $\\rho_i$ contribution by region', fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylim(0, 90)

    # *** FIXED suptitle: no RHO_I_LOWER reference ***
    fig.suptitle('Robustness of Variance Decomposition to $\\rho_i$ Distribution Shape\n'
                 f'(Alexandrov 2010 FYI: $\\mu$={RHO_I_FYI_MU} kg m$^{{-3}}$, '
                 f'$\\sigma$={RHO_I_FYI_SIGMA} kg m$^{{-3}}$, '
                 f'{int(RHO_I_UPPER_CAP)} kg m$^{{-3}}$ upper cap; '
                 f'$\\delta \\sim N$({DELTA_BASELINE},{DELTA_SIGMA}), '
                 f'$\\rho_s$ unclipped)',
                 fontsize=10, y=1.00)
    fig.tight_layout()

    # Inset: two distribution PDFs matching Step 02b
    ax_inset = axes[0].inset_axes([0.55, 0.50, 0.42, 0.45])

    rho_range = np.linspace(800, 950, 500)
    mu = RHO_I_FYI_MU
    sigma = RHO_I_FYI_SIGMA
    dx = rho_range[1] - rho_range[0]

    # Gaussian PDF (uncapped, shown for shape; cap at 940 marked separately)
    pdf_g = norm_dist.pdf(rho_range, loc=mu, scale=sigma)
    ax_inset.plot(rho_range, pdf_g, color=shape_colors['gaussian'], lw=2,
                  label='Gaussian')

    # Skew-normal PDF with matched mean and sigma
    alpha_sn = -2.0
    delta_skew = alpha_sn / np.sqrt(1 + alpha_sn**2)
    loc_adj = mu - sigma * np.sqrt(2/np.pi) * delta_skew
    scale_adj = sigma / np.sqrt(1 - 2 * delta_skew**2 / np.pi)
    pdf_sn = skewnorm.pdf(rho_range, alpha_sn, loc=loc_adj, scale=scale_adj)
    ax_inset.plot(rho_range, pdf_sn, color=shape_colors['skewnorm'], lw=2,
                  ls='--', label='Skew-Normal')

    # Reference lines: pure-ice density and 940 cap
    ax_inset.axvline(917, color='red', ls=':', lw=0.8, alpha=0.5)
    ax_inset.axvline(RHO_I_UPPER_CAP, color='black', ls='-', lw=0.9, alpha=0.7)

    ymax_inset = ax_inset.get_ylim()[1] if ax_inset.get_ylim()[1] > 0 else 0.02
    ax_inset.text(917, ymax_inset * 0.92,
                  '917', color='red', fontsize=7, alpha=0.7,
                  rotation=0, ha='right')
    ax_inset.text(RHO_I_UPPER_CAP, ymax_inset * 0.92,
                  '940 cap', color='black', fontsize=7,
                  rotation=0, ha='left')

    ax_inset.set_xlim(800, 950)
    ax_inset.set_xlabel('$\\rho_i$ (kg m$^{-3}$)', fontsize=8)
    ax_inset.set_ylabel('PDF', fontsize=8)
    ax_inset.set_title('$\\rho_i$ distributions tested', fontsize=8.5)
    ax_inset.tick_params(labelsize=7)
    ax_inset.legend(fontsize=7, loc='upper left')
    ax_inset.patch.set_alpha(0.9)

    fig.savefig(f'{M3_FIGURES}/R3_Fig_shape_robustness.png')
    fig.savefig(f'{M3_FIGURES}/R3_Fig_shape_robustness.pdf')
    print(f"    Saved: R3_Fig_shape_robustness.png/pdf")
    plt.close()
else:
    print("  SKIPPING shape robustness figure: JSON not found")


print("\n" + "=" * 70)
print("ALL FIGURES:")
for f in sorted(os.listdir(M3_FIGURES)):
    if not f.startswith('.'):
        fpath = f'{M3_FIGURES}/{f}'
        print(f"  {f} ({os.path.getsize(fpath)/1024:.0f} KB)")

print(f"\nStep 05 v5 complete in {time.time()-t0:.1f} s")

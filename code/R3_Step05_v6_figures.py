#!/usr/bin/env python3
"""
R3_Step05_v6_figures.py — Publication-quality figures (Alex Comments 16 Jun 2026)
================================================================================
*** v6 CHANGES (16 June 2026, Alex Figure Comments 3, 4, 5): ***
  Comment 3 (Fig01 Western Weddell Sea best-guess shifted-right concern):
    The best-guess line in each panel now uses the REGIONAL central rho_i
    value (882 kg/m^3 MYI for the western Weddell Sea sub-sector; 916.7
    kg/m^3 FYI for the eastern Weddell Sea, the Ross Sea, and the Weddell
    Sea aggregate), instead of the single global RHO_I_C value used in v5.
    Under v5, the western Weddell Sea best-guess used FYI density (916.7
    kg/m^3) while its Monte Carlo ensemble integrated over the MYI
    parameterisation (mu=882, sigma=23 kg/m^3), which produced the visual
    rightward shift Alex flagged. The v6 fix makes the best-guess and the
    MC ensemble use the same regional central density value, removing the
    artefactual shift.

  Comment 4 (Fig05 panel (a) redundant with Fig01):
    Panel (a) of Figure 5 has been removed because its content was a
    sub-sector-level repetition of the same MC vs best-guess histograms
    already shown in Figure 1. The Figure 5 layout is now a SINGLE panel
    showing only the box plots of per-segment Monte Carlo means, which is
    the new and non-redundant content. The figure file name remains
    R3_Fig05_subsectors.{pdf,png} but the panel structure is simplified
    from 1x2 to 1x1.

  Comment 5 (Fig05 whisker definition):
    The Figure 5 caption (set in LaTeX) and the script header note now
    explicitly state that the whiskers extend from the box to the most
    extreme data point within 1.5 x IQR from the box, with individual
    outliers shown as small markers beyond the whiskers (this is the
    standard matplotlib boxplot convention).

v5 changes preserved:
  Comment 2b: Fig05 — remove panel (c) table; figure becomes 1×2
  Comment 3a: Fig03 — change from 1×4 to 2×2 layout; improve legend
  Comment 3b: NEW Fig03b — δ sensitivity sweep
  Comment 4:  Shape robustness figure — add inset showing 4 distribution PDFs

Run on Gadi:
  source ~/cryo2ice_env/bin/activate
  python R3_Step05_v6_figures.py
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, skewnorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

plt.rcParams.update({
    'font.size': 10, 'font.family': 'sans-serif',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

RHO_I_C = RHO_I_CONFIGURED_MU
RHO_I_S = RHO_I_SIGMA_DEFAULT

# v6 Comment 3: regional central rho_i values for the best-guess line in
# Figure 1. The western Weddell Sea is treated as multi-year ice (MYI);
# all other regions are treated as first-year ice (FYI). The Weddell Sea
# aggregate uses FYI here because the aggregate combines E and W
# sub-sectors at the segment level and the FYI value is the dominant
# population (~74%) by sample count.
RHO_I_FYI = 916.7
RHO_I_MYI = 882.0
REGION_RHO_I = {
    'Weddell':   RHO_I_FYI,
    'Weddell_W': RHO_I_MYI,
    'Weddell_E': RHO_I_FYI,
    'Ross':      RHO_I_FYI,
}

COL = {
    'Weddell':   '#0072B2', 'Weddell_W': '#882255',
    'Weddell_E': '#44AA99', 'Ross':      '#D55E00', 'All': '#333333',
}

t0 = time.time()
print("=" * 70)
print("R3 Step 05 v5: Generate All Figures (Alex meeting revisions)")
print("=" * 70)

# ── Load data ──
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

if 'truncnorm' in results_B_raw:
    results_B = results_B_raw['truncnorm']
else:
    results_B = results_B_raw

for v in ['v4_coupled', 'v3_coupled']:
    p = f'{M3_OUTPUT}/R3_density_sweep_d070_{v}.csv'
    if os.path.exists(p):
        df_sweep = pd.read_csv(p)
        print(f"  Sweep (ρ_i): {p}")
        break

# *** NEW: Load δ sweep data ***
delta_sweep_path = f'{M3_OUTPUT}/R3_delta_sweep_v1.csv'
if os.path.exists(delta_sweep_path):
    df_delta_sweep = pd.read_csv(delta_sweep_path)
    print(f"  Sweep (δ): {delta_sweep_path}")
else:
    df_delta_sweep = None
    print(f"  WARNING: δ sweep CSV not found at {delta_sweep_path}")
    print(f"           Run R3_Step03b_delta_sweep.py first!")

for v in ['v3', 'v2']:
    p = f'{M3_OUTPUT}/R3_penetration_sensitivity_d070_{v}.csv'
    if os.path.exists(p):
        df_pen = pd.read_csv(p)
        print(f"  Penetration: {p}")
        break

region_col = 'region' if 'region' in df_sweep.columns else 'sector'

is2_fb_raw = df['is2_fb_wm'].values
hs_obs = df['hs_d070'].values

# v5 (kept for non-figure-1 sections that still expect a global best-guess)
hi_bestguess = compute_ice_thickness(is2_fb_raw, hs_obs,
                                      RHO_S_CENTRAL, RHO_I_C, RHO_W_CENTRAL)

# v6 Comment 3: regional best-guess arrays, used in Figure 1 only.
# The Western Weddell Sea segments are computed at rho_i = 882 (MYI), and
# all other regions at rho_i = 916.7 (FYI), to match the regional rho_i
# treatment used in the Monte Carlo ensemble for the same region.
hi_bestguess_regional = {}
for r in REGIONS:
    rho_i_r = REGION_RHO_I[r]
    hi_bestguess_regional[r] = compute_ice_thickness(
        is2_fb_raw, hs_obs, RHO_S_CENTRAL, rho_i_r, RHO_W_CENTRAL)

mc_draw_totals = {}
for r in REGIONS:
    mask = get_region_mask(sectors, subsectors, r)
    mc_draw_totals[r] = int(mask.sum()) * N_MC


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: ALL MC DRAWS histogram vs best-guess histogram
# (unchanged from v4)
# ═══════════════════════════════════════════════════════════════════
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

    # v6 Comment 3: use the REGIONAL best-guess array so the best-guess
    # uses the same central rho_i as the MC ensemble for this region
    # (882 kg/m^3 MYI for western Weddell Sea; 916.7 kg/m^3 FYI otherwise).
    rho_i_used = REGION_RHO_I[region]
    hi_bg = hi_bestguess_regional[region][mask]
    hi_bg_valid = hi_bg[np.isfinite(hi_bg)]
    ax.hist(hi_bg_valid, bins=hist_bins, color='black', alpha=0.9,
            edgecolor='black', density=True, histtype='step', linewidth=1.8,
            label=f'Best guess (n={len(hi_bg_valid):,} segs, '
                  f'$\\rho_i$={rho_i_used:.0f})')

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
             f'$\\rho_i$ eff$\\approx$880 kg m$^{{-3}}$, '
             f'$\\rho_s$={RHO_S_CENTRAL:.0f} kg m$^{{-3}}$, '
             f'$\\rho_w$={RHO_W_CENTRAL:.0f} kg m$^{{-3}}$ fixed)',
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f'{M3_FIGURES}/R3_Fig01_ExpA_regions.png')
fig.savefig(f'{M3_FIGURES}/R3_Fig01_ExpA_regions.pdf')
print(f"    Saved: R3_Fig01_ExpA_regions.png/pdf")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Variance decomposition bar chart (unchanged from v4)
# ═══════════════════════════════════════════════════════════════════
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
ax.set_title(f'Variance Decomposition (truncnorm $\\rho_i$, Gaussian others)\n'
             f'$\\rho_i$ eff$\\approx$880$\\pm${RHO_I_S} kg m$^{{-3}}$, '
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


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: ρ_i density sweep — *** NOW 2×2 LAYOUT (Comment 3a) ***
# ═══════════════════════════════════════════════════════════════════
print(f"  Figure 3: ρ_i density sweep (2×2 layout)...")

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

sweep_mu = RHO_I_CONFIGURED_MU

# *** 2×2 layout ***
fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True, sharex=True)
axes_flat = axes.flatten()

for ax, region in zip(axes_flat, plot_regions_sweep):
    sub = df_sweep[(df_sweep['dist_shape'] == 'truncnorm') &
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
    ax.set_ylim(-2, 80)
    ax.grid(True, alpha=0.3)

# x/y labels on outer axes only
for ax in axes[1, :]:
    ax.set_xlabel('Sea-ice density uncertainty $\\sigma_{\\rho_i}$ (kg m$^{-3}$)', fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel('Fractional variance contribution (%)', fontsize=10)

# *** Single legend at bottom ***
handles, labels_leg = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc='lower center', ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(f'Sensitivity of SIT variance to $\\sigma_{{\\rho_i}}$\n'
             f'($\\delta \\sim N$({DELTA_BASELINE},{DELTA_SIGMA}), '
             f'$\\rho_s$={RHO_S_CENTRAL}$\\pm${int(RHO_S_SIGMA)} kg m$^{{-3}}$, '
             f'$\\rho_w$={RHO_W_CENTRAL:.0f} kg m$^{{-3}}$ fixed)',
             fontsize=11, y=1.02)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(f'{M3_FIGURES}/R3_Fig03_sweep_rho_i.png')
fig.savefig(f'{M3_FIGURES}/R3_Fig03_sweep_rho_i.pdf')
print(f"    Saved: R3_Fig03_sweep_rho_i.png/pdf")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3b: δ sensitivity sweep — 2×2 LAYOUT (Comment 3b)
# ═══════════════════════════════════════════════════════════════════
if df_delta_sweep is not None:
    print(f"  Figure 3b: δ sensitivity sweep (2×2 layout)...")

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
        ax.set_ylim(-2, 80)
        ax.grid(True, alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel('Penetration factor uncertainty $\\sigma_{\\delta}$', fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel('Fractional variance contribution (%)', fontsize=10)

    handles, labels_leg = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Sensitivity of SIT variance to $\\sigma_{{\\delta}}$\n'
                 f'($\\rho_i$ truncnorm $\\mu$={RHO_I_C}, $\\sigma$={RHO_I_S}, '
                 f'$\\rho_s$={RHO_S_CENTRAL}$\\pm${int(RHO_S_SIGMA)} kg m$^{{-3}}$, '
                 f'$\\rho_w$={RHO_W_CENTRAL:.0f} kg m$^{{-3}}$ fixed)',
                 fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(f'{M3_FIGURES}/R3_Fig03b_sweep_delta.png')
    fig.savefig(f'{M3_FIGURES}/R3_Fig03b_sweep_delta.pdf')
    print(f"    Saved: R3_Fig03b_sweep_delta.png/pdf")
    plt.close()
else:
    print("  SKIPPING Figure 3b: δ sweep data not available")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Penetration sensitivity (unchanged from v4)
# ═══════════════════════════════════════════════════════════════════
print("  Figure 4: Penetration sensitivity...")

fig, ax = plt.subplots(figsize=(10, 6.5))

delta_f_arr = df['delta_f'].values
rho_s_c = RHO_S_CENTRAL
rho_w_c = RHO_W_CENTRAL
delta_range = np.arange(0.40, 1.01, 0.05)

fig4_regions = ['Weddell', 'Weddell_W', 'Weddell_E', 'Ross']
fig4_styles = {
    'Weddell':   {'ls': '-',  'lw': 2.5, 'ms': 5, 'af': 0.10},
    'Weddell_W': {'ls': '-',  'lw': 2.0, 'ms': 4, 'af': 0.08},
    'Weddell_E': {'ls': '-',  'lw': 2.0, 'ms': 4, 'af': 0.08},
    'Ross':      {'ls': '-',  'lw': 2.5, 'ms': 5, 'af': 0.10},
}

for region in fig4_regions:
    mask = get_region_mask(sectors, subsectors, region)
    if mask.sum() < 10:
        continue
    color = COL[region]
    style = fig4_styles[region]
    n_seg = mask.sum()
    n_mc_total = n_seg * N_MC
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
             f'($\\rho_i$ eff$\\approx$880 kg m$^{{-3}}$, '
             f'$\\rho_w$={RHO_W_CENTRAL:.0f} kg m$^{{-3}}$ fixed, '
             f'shading=$\\rho_s\\pm${RHO_S_SIGMA:.0f} kg m$^{{-3}}$)')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{M3_FIGURES}/R3_Fig04_penetration.png')
fig.savefig(f'{M3_FIGURES}/R3_Fig04_penetration.pdf')
print(f"    Saved: R3_Fig04_penetration.png/pdf")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Sub-sector — *** v6 1×1 LAYOUT (Comment 4: panel (a) removed) ***
#
# Comment 4: Panel (a) was a sub-sector-level repetition of the same MC vs
# best-guess histograms already shown in Figure 1, so it has been deleted.
# Figure 5 is now a single panel showing the box plots of per-segment
# Monte Carlo means, which is the genuinely new content.
#
# Comment 5: The matplotlib boxplot whiskers extend from the box to the
# most extreme data point within 1.5 x IQR from the box (the standard
# convention; whis=1.5 is the default). Outlier points beyond the
# whiskers are suppressed in this figure (showfliers=False) because
# their inclusion would compress the visible scale of the boxes. The
# caption (set in LaTeX) explicitly states the whisker definition.
# ═══════════════════════════════════════════════════════════════════
print("  Figure 5: Sub-sector box plots (1×1, panel (a) removed; v6)...")

fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

plot_regions_fig5 = ['Weddell', 'Weddell_W', 'Weddell_E', 'Ross']

# Box plots of per-segment Monte Carlo means with mean diamonds and
# median lines. Whiskers are matplotlib default (1.5 x IQR).
data_boxes, labels_boxes, colors_boxes = [], [], []
for region in plot_regions_fig5:
    mask = get_region_mask(sectors, subsectors, region)
    hi_r = hi_mean[mask]
    hi_clipped = hi_r[(hi_r > -2) & (hi_r < 8)]
    data_boxes.append(hi_clipped)
    labels_boxes.append(REGION_LABELS[region])
    colors_boxes.append(COL[region])

bp = ax.boxplot(data_boxes, labels=labels_boxes, patch_artist=True, widths=0.6,
                showfliers=False, whis=1.5,
                medianprops=dict(color='black', lw=1.5))
for patch, color in zip(bp['boxes'], colors_boxes):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

from matplotlib.lines import Line2D
mean_handle = None
for i, region in enumerate(plot_regions_fig5):
    mask = get_region_mask(sectors, subsectors, region)
    h = ax.scatter(i + 1, hi_mean[mask].mean(), color='black', marker='D',
                   s=40, zorder=5, label='Mean' if i == 0 else None)
    if i == 0:
        mean_handle = h

median_handle = Line2D([0], [0], color='black', lw=1.5, label='Median')
whisker_handle = Line2D([0], [0], color='black', lw=0.8,
                         label='Whiskers: 1.5 × IQR')
ax.legend(handles=[mean_handle, median_handle, whisker_handle],
          fontsize=10, loc='upper right')

ax.set_ylabel('Sea-ice thickness $h_i$ (m)', fontsize=12)
ax.set_xlabel('Sector / sub-sector', fontsize=12)
ax.set_title('Per-segment Monte Carlo mean sea-ice thickness by sector and sub-sector',
             fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='both', labelsize=10)

fig.tight_layout()
fig.savefig(f'{M3_FIGURES}/R3_Fig05_subsectors.png')
fig.savefig(f'{M3_FIGURES}/R3_Fig05_subsectors.pdf')
print(f"    Saved: R3_Fig05_subsectors.png/pdf (v6: 1x1 box plot only)")
plt.close()


# ═══════════════════════════════════════════════════════════════════
# SHAPE ROBUSTNESS FIGURE — *** ADD INSET (Comment 4) ***
# ═══════════════════════════════════════════════════════════════════
print("  Shape robustness figure (with inset distributions)...")

shape_json_path = f'{M3_OUTPUT}/R3_shape_robustness_v4.json'
if os.path.exists(shape_json_path):
    with open(shape_json_path) as f:
        all_shape_results = json.load(f)

    SHAPES = ['truncnorm', 'gaussian_clipped', 'skewnorm', 'uniform']
    SHAPE_LABELS = {
        'truncnorm': 'Truncated Normal',
        'gaussian_clipped': 'Gaussian (clipped)',
        'skewnorm': 'Skew-Normal ($\\alpha$=$-$2)',
        'uniform': 'Uniform [750, 917]',
    }
    shape_colors = {
        'truncnorm': '#0072B2', 'gaussian_clipped': '#D55E00',
        'skewnorm': '#882255', 'uniform': '#44AA99',
    }
    params_rob = ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta']
    param_labels_rob = {
        'fb_is2': '$f_b^{IS2}$', 'fb_cs2': '$f_b^{CS2}$',
        'rho_s': '$\\rho_s$', 'rho_i': '$\\rho_i$', 'delta': '$\\delta$',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): All regions grouped bar chart
    ax = axes[0]
    x = np.arange(len(params_rob))
    width_r = 0.18

    for i, shape in enumerate(SHAPES):
        if shape not in all_shape_results or 'all' not in all_shape_results[shape]:
            continue
        vals = [all_shape_results[shape]['all'].get(p, 0) for p in params_rob]
        bars = ax.bar(x + (i - 1.5) * width_r, vals, width_r,
                      label=SHAPE_LABELS[shape], color=shape_colors[shape],
                      alpha=0.8, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=6,
                        color=shape_colors[shape], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([param_labels_rob[p] for p in params_rob], fontsize=11)
    ax.set_ylabel('Variance contribution (%)', fontsize=11)
    ax.set_title('(a) All regions', fontsize=11)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylim(0, 80)

    # Panel (b): δ contribution by region
    ax = axes[1]
    regions_plot_rob = REGIONS
    x_r = np.arange(len(regions_plot_rob))

    for i, shape in enumerate(SHAPES):
        if shape not in all_shape_results:
            continue
        vals = []
        for region in regions_plot_rob:
            if region in all_shape_results[shape]:
                vals.append(all_shape_results[shape][region].get('delta', 0))
            else:
                vals.append(0)
        bars = ax.bar(x_r + (i - 1.5) * width_r, vals, width_r,
                      label=SHAPE_LABELS[shape], color=shape_colors[shape],
                      alpha=0.8, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=6.5,
                        color=shape_colors[shape], fontweight='bold')

    ax.set_xticks(x_r)
    ax.set_xticklabels([REGION_LABELS[r] for r in regions_plot_rob], fontsize=9)
    ax.set_ylabel('$\\delta$ variance contribution (%)', fontsize=11)
    ax.set_title('(b) Penetration $\\delta$ contribution by region', fontsize=11)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylim(0, 80)

    fig.suptitle('Robustness of Variance Decomposition to $\\rho_i$ Distribution Shape\n'
                 f'($\\mu$={RHO_I_CONFIGURED_MU}, $\\sigma$={RHO_I_SIGMA_DEFAULT}, '
                 f'bound $\\leq${RHO_I_UPPER:.0f} kg m$^{{-3}}$, '
                 f'$\\delta \\sim N$({DELTA_BASELINE},{DELTA_SIGMA}), '
                 f'$\\rho_s$ unclipped)',
                 fontsize=10, y=0.96)
    fig.tight_layout()

    # *** INSET: 4 distribution shape PDFs (Comment 4) ***
    # MUST be created AFTER tight_layout() so that the parent axes
    # positions are finalized and the inset coordinates are respected.
    # Position: [x, y, width, height] in axes fraction (0–1 range)
    #   x: 0=left edge, 1=right edge of panel (a)
    #   y: 0=bottom edge, 1=top edge of panel (a)
    #   To move left:  decrease x (e.g., 0.35)
    #   To move down:  decrease y (e.g., 0.30)
    #   To make larger: increase width/height
    ax_inset = axes[0].inset_axes([0.42, 0.35, 0.45, 0.42])

    from scipy.stats import norm as norm_dist
    rho_range = np.linspace(740, 930, 500)
    mu = RHO_I_CONFIGURED_MU
    sigma = RHO_I_SIGMA_DEFAULT
    dx = rho_range[1] - rho_range[0]

    # Truncated normal PDF
    a_tn = (RHO_I_LOWER - mu) / sigma
    b_tn = (RHO_I_UPPER - mu) / sigma
    pdf_tn = truncnorm.pdf(rho_range, a_tn, b_tn, loc=mu, scale=sigma)
    ax_inset.plot(rho_range, pdf_tn, color=shape_colors['truncnorm'], lw=1.5,
                  label='Trunc. Norm.')

    # Gaussian clipped
    pdf_gc = norm_dist.pdf(rho_range, loc=mu, scale=sigma)
    pdf_gc[(rho_range < RHO_I_LOWER) | (rho_range > RHO_I_UPPER)] = 0
    pdf_gc = pdf_gc / (pdf_gc.sum() * dx)
    ax_inset.plot(rho_range, pdf_gc, color=shape_colors['gaussian_clipped'], lw=1.5,
                  ls='--', label='Gauss. (clip)')

    # Skew-normal (α=-2)
    alpha_sn = -2.0
    loc_adj = mu + sigma * 0.636 * (-alpha_sn / abs(alpha_sn))
    pdf_sn = skewnorm.pdf(rho_range, alpha_sn, loc=loc_adj, scale=sigma)
    pdf_sn[(rho_range < RHO_I_LOWER) | (rho_range > RHO_I_UPPER)] = 0
    pdf_sn = pdf_sn / (pdf_sn.sum() * dx)
    ax_inset.plot(rho_range, pdf_sn, color=shape_colors['skewnorm'], lw=1.5,
                  ls='-.', label='Skew-Norm.')

    # Uniform
    pdf_uni = np.where((rho_range >= RHO_I_LOWER) & (rho_range <= RHO_I_UPPER),
                       1.0 / (RHO_I_UPPER - RHO_I_LOWER), 0)
    ax_inset.plot(rho_range, pdf_uni, color=shape_colors['uniform'], lw=1.5,
                  ls=':', label='Uniform')

    ax_inset.axvline(RHO_I_UPPER, color='red', ls=':', lw=0.8, alpha=0.5)
    ax_inset.set_xlim(740, 930)
    ax_inset.set_xlabel('$\\rho_i$ (kg m$^{-3}$)', fontsize=7)
    ax_inset.set_ylabel('PDF', fontsize=7)
    ax_inset.set_title('$\\rho_i$ distributions tested', fontsize=7.5)
    ax_inset.tick_params(labelsize=6)
    ax_inset.legend(fontsize=5.5, loc='upper left')
    ax_inset.patch.set_alpha(0.85)

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

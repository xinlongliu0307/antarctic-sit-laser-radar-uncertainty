#!/usr/bin/env python3
"""
R3_Step02b_shape_robustness.py — Distribution Shape Robustness Check
=========================================================================
*** Updated for dual sign-off (Alex + Stuart, 28 April 2026) ***

Tests whether the Experiment B variance decomposition is sensitive to the
assumed distribution shape of sea-ice density rho_i.

Two shapes tested at the Alexandrov FYI parameterisation:
  1. Gaussian:    pure Gaussian, mu = 916.7, sigma = 35.7 kg/m^3,
                  with 940 kg/m^3 soft upper cap, no lower bound
  2. Skew-Normal: scipy skewnorm with alpha = -2 (negative skew toward
                  thinner ice), location and scale adjusted so that the
                  realised mean and standard deviation match the Gaussian,
                  with 940 kg/m^3 soft upper cap

The 940 cap is the Timco and Frederking (1996) review upper bound for
in-situ first-year ice below the waterline, just below the 945 kg/m^3
Cox and Weeks (1983) air-free theoretical maximum, and well below
seawater density at 1024 kg/m^3.

All other parameters identical to the primary analysis:
  IS2/CS2 freeboard: Gaussian sigma = 0.030 m
  rho_s:             Gaussian mu = 330, sigma = 70 (no clipping)
  delta:             Gaussian mu = 0.70, sigma = 0.20, clipped [0.30, 1.00]
  rho_w:             FIXED at 1024

Run: source ~/cryo2ice_env/bin/activate && python R3_Step02b_shape_robustness.py
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

t0 = time.time()


# ===================================================================
# DISTRIBUTION SAMPLERS — TWO shapes, both with 940 kg/m^3 cap
# ===================================================================

def sample_rho_i(shape, n, rng):
    """
    Draw n samples of sea-ice density from one of two distributions.
    Both shapes apply the 940 kg/m^3 soft upper cap (no lower bound).

    Both shapes use the Alexandrov FYI parameterisation:
        mu = 916.7 kg/m^3, sigma = 35.7 kg/m^3.
    """
    mu = RHO_I_FYI_MU
    sigma = RHO_I_FYI_SIGMA

    if shape == 'gaussian':
        # Pure Gaussian — primary distribution
        samples = rng.normal(mu, sigma, size=n)

    elif shape == 'skewnorm':
        # Skew-normal with alpha = -2 (negative skew)
        # Adjust location and scale so realised mean and sigma match Gaussian
        alpha = -2.0
        delta_skew = alpha / np.sqrt(1 + alpha**2)
        loc_adj = mu - sigma * np.sqrt(2/np.pi) * delta_skew
        scale_adj = sigma / np.sqrt(1 - 2 * delta_skew**2 / np.pi)
        samples = skewnorm.rvs(alpha, loc=loc_adj, scale=scale_adj, size=n,
                               random_state=rng.integers(2**31))

    else:
        raise ValueError(f"Unknown shape: {shape}. "
                         f"Only 'gaussian' and 'skewnorm' supported.")

    # Apply 940 kg/m^3 soft upper cap
    samples = np.minimum(samples, RHO_I_UPPER_CAP)
    return samples


# ===================================================================
# MAIN
# ===================================================================

SHAPES = ['gaussian', 'skewnorm']
SHAPE_LABELS = {
    'gaussian':  'Gaussian',
    'skewnorm':  'Skew-Normal (alpha = -2)',
}

print("=" * 70)
print("R3 Step 02b: Distribution Shape Robustness Check")
print("  *** Dual sign-off (Alex + Stuart, 28 April 2026) ***")
print("  *** Alexandrov 2010 FYI with 940 kg/m^3 cap ***")
print(f"  Shapes: {SHAPES}")
print(f"  rho_i:  mu = {RHO_I_FYI_MU}, sigma = {RHO_I_FYI_SIGMA} kg/m^3")
print(f"  Upper cap: {RHO_I_UPPER_CAP} kg/m^3, no lower bound")
print(f"  rho_s:  Gaussian, no clipping")
print(f"  delta:  Gaussian(mu = {DELTA_BASELINE}, sigma = {DELTA_SIGMA})")
print(f"  rho_w:  FIXED at {RHO_W_CENTRAL}")
print("=" * 70)

# Verify distribution properties
print("\n-- Distribution Properties (500,000 draws each) --")
rng_test = np.random.default_rng(0)
print(f"  {'Shape':28s}  {'Mean':>7s}  {'Std':>6s}  {'Med':>7s}  "
      f"{'Min':>7s}  {'Max':>7s}  {'%@cap':>6s}")
print("  " + "-" * 78)
for shape in SHAPES:
    s = sample_rho_i(shape, 500000, rng_test)
    pct_at_cap = 100 * np.mean(s >= RHO_I_UPPER_CAP - 1e-6)
    print(f"  {SHAPE_LABELS[shape]:28s}  "
          f"{s.mean():7.1f}  {s.std():6.1f}  {np.median(s):7.1f}  "
          f"{s.min():7.1f}  {s.max():7.1f}  {pct_at_cap:5.1f}%")

# Load data
df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_verified.pkl')
if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df['lon'].values)

n_total = len(df)
is2_fb = df['is2_fb_wm'].values
cs2_fb = df['cs2_rfb'].values
sectors = df['sector'].values
subsectors = df['subsector'].values

print(f"\nLoaded {n_total:,} matchups")

# Subsample for variance decomposition
rng = np.random.default_rng(42)
N_SUB = 5000
idx = rng.choice(n_total, N_SUB, replace=False)
idx.sort()

is2_c = is2_fb[idx]
cs2_c = cs2_fb[idx]
sectors_sub = sectors[idx]
subsectors_sub = subsectors[idx]

params = ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta']
N_MC_ROB = 5000

all_results = {}

for shape in SHAPES:
    t_shape = time.time()
    print(f"\n-- {SHAPE_LABELS[shape]} --")

    var_contributions = {p: np.zeros(N_SUB) for p in params}

    for ip, param in enumerate(params):
        rng_p = np.random.default_rng(500 + ip + SHAPES.index(shape) * 100)
        hi_samples = np.zeros((N_MC_ROB, N_SUB))

        for trial in range(N_MC_ROB):
            is2_t = is2_c.copy()
            cs2_t = cs2_c.copy()
            rho_s_t = RHO_S_CENTRAL
            rho_i_t = RHO_I_FYI_MU
            delta_t = DELTA_BASELINE

            if param == 'fb_is2':
                is2_t += rng_p.normal(0, FB_SIGMA_RAW, N_SUB)
            elif param == 'fb_cs2':
                cs2_t += rng_p.normal(0, FB_SIGMA_RAW, N_SUB)
            elif param == 'rho_s':
                rho_s_t = rng_p.normal(RHO_S_CENTRAL, RHO_S_SIGMA)
            elif param == 'rho_i':
                rho_i_t = sample_rho_i(shape, 1, rng_p)[0]
            elif param == 'delta':
                delta_t = sample_delta(1, rng_p)[0]

            eta_t = (1 + 0.51 * (rho_s_t if isinstance(rho_s_t, (int, float))
                     else rho_s_t) / 1000.0)**1.5
            hs_t = (is2_t - cs2_t) / (delta_t * eta_t)
            hi_samples[trial] = compute_ice_thickness(is2_t, hs_t,
                                                      rho_s_t, rho_i_t,
                                                      RHO_W_CENTRAL)

        var_contributions[param] = np.var(hi_samples, axis=0)

    total_var = sum(var_contributions.values())

    shape_results = {}
    for r in ['all'] + list(REGIONS):
        if r == 'all':
            m = np.ones(N_SUB, dtype=bool)
        else:
            m = get_region_mask(sectors_sub, subsectors_sub, r)
        if m.sum() < 10:
            continue
        tv = np.mean(total_var[m])
        row = {}
        for p in params:
            row[p] = float(np.mean(var_contributions[p][m]) / tv * 100) if tv > 0 else 0
        shape_results[r] = row

    all_results[shape] = shape_results
    dt = time.time() - t_shape
    print(f"  Completed in {dt:.1f}s")


# ===================================================================
# COMPARISON TABLE
# ===================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS COMPARISON: Variance Decomposition by rho_i Shape")
print(f"  (Alexandrov FYI: mu = {RHO_I_FYI_MU}, sigma = {RHO_I_FYI_SIGMA},")
print(f"   940 kg/m^3 upper cap)")
print("=" * 70)

print(f"\n  All regions combined:")
print(f"  {'Shape':28s}  {'fb_IS2':>6s}  {'fb_CS2':>6s}  "
      f"{'rho_s':>6s}  {'rho_i':>6s}  {'delta':>6s}")
print("  " + "-" * 68)
for shape in SHAPES:
    r = all_results[shape]['all']
    print(f"  {SHAPE_LABELS[shape]:28s}  "
          f"{r['fb_is2']:5.1f}%  {r['fb_cs2']:5.1f}%  "
          f"{r['rho_s']:5.1f}%  {r['rho_i']:5.1f}%  {r['delta']:5.1f}%")

for region in REGIONS:
    print(f"\n  {REGION_LABELS[region]}:")
    print(f"  {'Shape':28s}  {'fb_IS2':>6s}  {'fb_CS2':>6s}  "
          f"{'rho_s':>6s}  {'rho_i':>6s}  {'delta':>6s}")
    print("  " + "-" * 68)
    for shape in SHAPES:
        if region in all_results[shape]:
            r = all_results[shape][region]
            print(f"  {SHAPE_LABELS[shape]:28s}  "
                  f"{r['fb_is2']:5.1f}%  {r['fb_cs2']:5.1f}%  "
                  f"{r['rho_s']:5.1f}%  {r['rho_i']:5.1f}%  {r['delta']:5.1f}%")

print(f"\n  Dominant parameter by shape:")
for shape in SHAPES:
    r = all_results[shape]['all']
    dominant = max(r, key=r.get)
    print(f"  {SHAPE_LABELS[shape]:28s}  ->  {dominant} ({r[dominant]:.1f}%)")

json_path = f'{M3_OUTPUT}/R3_shape_robustness_v5.json'
with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Saved: {json_path}")


# ===================================================================
# FIGURE: Two-shape comparison
# ===================================================================
print("\n  Generating robustness figure (2-shape comparison)...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

shape_colors = {
    'gaussian':  '#0072B2',
    'skewnorm':  '#882255',
}

params_rob = ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta']
param_labels_rob = {
    'fb_is2': '$f_b^{IS2}$', 'fb_cs2': '$f_b^{CS2}$',
    'rho_s': '$\\rho_s$', 'rho_i': '$\\rho_i$', 'delta': '$\\delta$',
}

# Panel (a): All regions grouped bar chart
ax = axes[0]
x = np.arange(len(params_rob))
width = 0.35
for i, shape in enumerate(SHAPES):
    if 'all' not in all_results[shape]:
        continue
    vals = [all_results[shape]['all'].get(p, 0) for p in params_rob]
    bars = ax.bar(x + (i - 0.5) * width, vals, width,
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
regions_plot = REGIONS
x_r = np.arange(len(regions_plot))
width_r = 0.35

for i, shape in enumerate(SHAPES):
    vals = []
    for region in regions_plot:
        if region in all_results[shape]:
            vals.append(all_results[shape][region].get('rho_i', 0))
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
ax.set_xticklabels([REGION_LABELS[r] for r in regions_plot], fontsize=10)
ax.set_ylabel('$\\rho_i$ variance contribution (%)', fontsize=11)
ax.set_title('(b) $\\rho_i$ contribution by region', fontsize=11)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.2)
ax.set_ylim(0, 90)

fig.suptitle('Robustness of Variance Decomposition to $\\rho_i$ Distribution Shape\n'
             f'(Alexandrov 2010 FYI: $\\mu$ = {RHO_I_FYI_MU} kg m$^{{-3}}$, '
             f'$\\sigma$ = {RHO_I_FYI_SIGMA} kg m$^{{-3}}$, '
             f'940 kg m$^{{-3}}$ upper cap; '
             f'$\\delta \\sim N$({DELTA_BASELINE},{DELTA_SIGMA}))',
             fontsize=10, y=1.00)
fig.tight_layout()
fig.savefig(f'{M3_FIGURES}/R3_Fig_shape_robustness.png')
fig.savefig(f'{M3_FIGURES}/R3_Fig_shape_robustness.pdf')
print(f"    Saved: R3_Fig_shape_robustness.png/pdf")
plt.close()

print(f"\nStep 02b complete in {time.time()-t0:.1f} s")

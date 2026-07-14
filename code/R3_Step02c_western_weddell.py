#!/usr/bin/env python3
"""
R3_Step02c_western_weddell.py — Western Weddell sub-sector sensitivity
=========================================================================
*** Alex's 27 April 2026 meeting instruction ***
*** Dual-confirmed parameterisation: Alex + Stuart, 28 April 2026 ***

Tests four parameterisations of sea-ice density specifically for the
western Weddell sub-sector, where multi-year ice and first-year ice
coexist in unknown proportions. The empirical motivation is the
discrepancy between the FYI-derived western Weddell mean ice thickness
of 3.390 metres from the primary Step 02 analysis and the IceBridge
benchmark of approximately 2.4 to 2.6 metres reported by Kwok and
Kacimi (2018).

Four parameterisations tested (all with 940 kg/m^3 upper cap):
  1. Pure FYI:    mu = 916.7, sigma = 35.7  (Alexandrov 2010 FYI)
  2. 1/3 MYI:     mu = 905.1, sigma = 31.5  (FYI/MYI weighted average)
  3. 2/3 MYI:     mu = 893.6, sigma = 27.2  (FYI/MYI weighted average)
  4. Pure MYI:    mu = 882.0, sigma = 23.0  (Alexandrov 2010 MYI)

The weighted averages are computed as:
  mu_w   = (1 - f_MYI) * mu_FYI   + f_MYI * mu_MYI
  sigma_w (linear interpolation as first-order approximation)
where f_MYI is the multi-year ice fraction in the western Weddell.

For each parameterisation, the script reports:
  - Experiment A: regional mean h_i, std, 90% CI under the western
    Weddell sub-sector mask
  - Experiment B: variance decomposition for the five parameters
  - Comparison against IceBridge benchmark of 2.4-2.6 m

All other parameters identical to the primary analysis:
  IS2/CS2 freeboard: Gaussian sigma = 0.030 m
  rho_s:             Gaussian mu = 330, sigma = 70 (no clipping)
  delta:             Gaussian mu = 0.70, sigma = 0.20, clipped [0.30, 1.00]
  rho_w:             FIXED at 1024 kg/m^3

Run: source ~/cryo2ice_env/bin/activate && python R3_Step02c_western_weddell.py
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

t0 = time.time()


# ===================================================================
# Western Weddell parameterisation cases
# ===================================================================

# Compute weighted averages between FYI and MYI Alexandrov parameters
def weighted_param(f_myi, mu_fyi=RHO_I_FYI_MU, sigma_fyi=RHO_I_FYI_SIGMA,
                   mu_myi=RHO_I_MYI_MU, sigma_myi=RHO_I_MYI_SIGMA):
    """Linear weighted average for mean; linear interpolation for sigma."""
    mu = (1 - f_myi) * mu_fyi + f_myi * mu_myi
    sigma = (1 - f_myi) * sigma_fyi + f_myi * sigma_myi
    return mu, sigma

CASES = []
for f_myi, label in [(0.0,  'Pure FYI'),
                     (1/3,  '1/3 MYI'),
                     (2/3,  '2/3 MYI'),
                     (1.0,  'Pure MYI')]:
    mu, sigma = weighted_param(f_myi)
    CASES.append({
        'label': label,
        'f_myi': f_myi,
        'mu': mu,
        'sigma': sigma,
    })


def sample_rho_i_capped(mu, sigma, n, rng):
    """Sample rho_i from a Gaussian with the 940 kg/m^3 upper cap."""
    samples = rng.normal(mu, sigma, size=n)
    samples = np.minimum(samples, RHO_I_UPPER_CAP)
    return samples


# ===================================================================
# MAIN
# ===================================================================

print("=" * 72)
print("R3 Step 02c: Western Weddell Sub-sector Sensitivity")
print("  *** Alex's 27 April 2026 meeting instruction ***")
print("  *** Four parameterisations with 940 kg/m^3 cap ***")
print("=" * 72)
print("\nParameterisation cases:")
print(f"  {'Case':14s}  {'f_MYI':>6s}  {'mu':>7s}  {'sigma':>6s}  {'denom':>6s}  {'amp':>5s}")
print("  " + "-" * 60)
for c in CASES:
    denom = RHO_W_CENTRAL - c['mu']
    amp = RHO_W_CENTRAL / denom
    print(f"  {c['label']:14s}  {c['f_myi']:6.3f}  {c['mu']:7.1f}  "
          f"{c['sigma']:6.1f}  {denom:6.1f}  {amp:5.2f}")

# Verify distribution properties for each case
print("\nDistribution properties (500,000 draws each):")
print(f"  {'Case':14s}  {'Mean':>7s}  {'Std':>6s}  {'Min':>7s}  {'Max':>7s}  {'%@cap':>6s}")
print("  " + "-" * 56)
rng_test = np.random.default_rng(0)
for c in CASES:
    s = sample_rho_i_capped(c['mu'], c['sigma'], 500000, rng_test)
    pct_at_cap = 100 * np.mean(s >= RHO_I_UPPER_CAP - 1e-6)
    print(f"  {c['label']:14s}  {s.mean():7.2f}  {s.std():6.2f}  "
          f"{s.min():7.1f}  {s.max():7.1f}  {pct_at_cap:5.2f}%")

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

# Western Weddell sub-sector mask
ww_mask = get_region_mask(sectors, subsectors, 'Weddell_W')
n_ww = ww_mask.sum()
print(f"Western Weddell sub-sector: {n_ww:,} matchups")

is2_ww = is2_fb[ww_mask]
cs2_ww = cs2_fb[ww_mask]


# ===================================================================
# EXPERIMENT A: Full simultaneous perturbation per case
# ===================================================================

print("\n" + "=" * 72)
print("EXPERIMENT A — Western Weddell ice thickness per parameterisation")
print(f"  N_MC = {N_MC:,} per segment, n = {n_ww:,} segments")
print("=" * 72)

results_A = {}

for c in CASES:
    t_case = time.time()
    print(f"\n-- {c['label']} (mu = {c['mu']:.1f}, sigma = {c['sigma']:.1f}) --")

    rng = np.random.default_rng(42)
    hi_mean_arr = np.zeros(n_ww)
    hi_std_arr = np.zeros(n_ww)
    hi_p05_arr = np.zeros(n_ww)
    hi_p95_arr = np.zeros(n_ww)

    batch = 5000
    n_batches = (n_ww + batch - 1) // batch

    for b in range(n_batches):
        i0, i1 = b * batch, min((b + 1) * batch, n_ww)
        nb = i1 - i0

        is2_mc = rng.normal(is2_ww[i0:i1], FB_SIGMA_RAW, size=(N_MC, nb))
        cs2_mc = rng.normal(cs2_ww[i0:i1], FB_SIGMA_RAW, size=(N_MC, nb))
        delta_mc = sample_delta(N_MC, rng).reshape(-1, 1)
        rho_s_mc = rng.normal(RHO_S_CENTRAL, RHO_S_SIGMA, size=(N_MC, 1))
        rho_i_mc = sample_rho_i_capped(c['mu'], c['sigma'], N_MC, rng).reshape(-1, 1)
        rho_w_mc = RHO_W_CENTRAL

        eta_s_mc = (1 + 0.51 * rho_s_mc / 1000.0)**1.5
        hs_mc = (is2_mc - cs2_mc) / (delta_mc * eta_s_mc)
        hi_mc = compute_ice_thickness(is2_mc, hs_mc, rho_s_mc, rho_i_mc, rho_w_mc)

        hi_mean_arr[i0:i1] = np.mean(hi_mc, axis=0)
        hi_std_arr[i0:i1] = np.std(hi_mc, axis=0)
        hi_p05_arr[i0:i1] = np.percentile(hi_mc, 5, axis=0)
        hi_p95_arr[i0:i1] = np.percentile(hi_mc, 95, axis=0)

    ci_width = hi_p95_arr - hi_p05_arr

    results_A[c['label']] = {
        'mu': c['mu'],
        'sigma': c['sigma'],
        'f_myi': c['f_myi'],
        'h_i_mean': float(hi_mean_arr.mean()),
        'h_i_std': float(hi_std_arr.mean()),
        'h_i_p05': float(hi_p05_arr.mean()),
        'h_i_p95': float(hi_p95_arr.mean()),
        'ci_width': float(ci_width.mean()),
    }

    dt = time.time() - t_case
    print(f"  h_i = {hi_mean_arr.mean():.3f} +/- {hi_std_arr.mean():.3f} m, "
          f"90%CI = [{hi_p05_arr.mean():.3f}, {hi_p95_arr.mean():.3f}], "
          f"width = {ci_width.mean():.3f} m  ({dt:.1f}s)")


# ===================================================================
# EXPERIMENT B: Variance decomposition per case
# ===================================================================

print("\n" + "=" * 72)
print("EXPERIMENT B — Western Weddell variance decomposition per case")
print("=" * 72)

results_B = {}
N_SUB = min(3000, n_ww)
rng_idx = np.random.default_rng(43)
idx_sub = rng_idx.choice(n_ww, N_SUB, replace=False)
idx_sub.sort()

is2_sub = is2_ww[idx_sub]
cs2_sub = cs2_ww[idx_sub]

params = ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta']

for c in CASES:
    t_case = time.time()
    print(f"\n-- {c['label']} --")

    var_contributions = {p: np.zeros(N_SUB) for p in params}
    N_MC_VAR = 5000

    for ip, param in enumerate(params):
        rng_p = np.random.default_rng(700 + ip + CASES.index(c) * 50)
        hi_samples = np.zeros((N_MC_VAR, N_SUB))

        for trial in range(N_MC_VAR):
            is2_t = is2_sub.copy()
            cs2_t = cs2_sub.copy()
            rho_s_t = RHO_S_CENTRAL
            rho_i_t = c['mu']
            delta_t = DELTA_BASELINE

            if param == 'fb_is2':
                is2_t += rng_p.normal(0, FB_SIGMA_RAW, N_SUB)
            elif param == 'fb_cs2':
                cs2_t += rng_p.normal(0, FB_SIGMA_RAW, N_SUB)
            elif param == 'rho_s':
                rho_s_t = rng_p.normal(RHO_S_CENTRAL, RHO_S_SIGMA)
            elif param == 'rho_i':
                rho_i_t = sample_rho_i_capped(c['mu'], c['sigma'], 1, rng_p)[0]
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
    tv = np.mean(total_var)

    case_results = {}
    for p in params:
        case_results[p] = float(np.mean(var_contributions[p]) / tv * 100) if tv > 0 else 0
    results_B[c['label']] = case_results

    dt = time.time() - t_case
    print(f"  fb_IS2={case_results['fb_is2']:5.1f}%  "
          f"fb_CS2={case_results['fb_cs2']:5.1f}%  "
          f"rho_s={case_results['rho_s']:5.1f}%  "
          f"rho_i={case_results['rho_i']:5.1f}%  "
          f"delta={case_results['delta']:5.1f}%  "
          f"({dt:.1f}s)")


# ===================================================================
# COMPARISON AGAINST ICEBRIDGE BENCHMARK
# ===================================================================

ICEBRIDGE_LOW = 2.4
ICEBRIDGE_HIGH = 2.6
ICEBRIDGE_MID = (ICEBRIDGE_LOW + ICEBRIDGE_HIGH) / 2.0

print("\n" + "=" * 72)
print("COMPARISON AGAINST ICEBRIDGE BENCHMARK")
print("  Kwok and Kacimi (2018): western Weddell mean h_i = 2.4-2.6 m")
print("=" * 72)

print(f"\n  {'Case':14s}  {'mu':>7s}  {'sigma':>6s}  {'h_i':>6s}  "
      f"{'Delta':>7s}  {'Match?':>8s}")
print("  " + "-" * 60)

best_case = None
best_diff = float('inf')

for c in CASES:
    h_i = results_A[c['label']]['h_i_mean']
    delta_h = h_i - ICEBRIDGE_MID
    abs_diff = abs(delta_h)

    if ICEBRIDGE_LOW <= h_i <= ICEBRIDGE_HIGH:
        match_flag = '*** YES'
    elif abs_diff < 0.3:
        match_flag = 'close'
    else:
        match_flag = 'no'

    if abs_diff < best_diff:
        best_diff = abs_diff
        best_case = c['label']

    print(f"  {c['label']:14s}  {c['mu']:7.1f}  {c['sigma']:6.1f}  "
          f"{h_i:6.3f}  {delta_h:+7.3f}  {match_flag:>8s}")

print(f"\n  Best match to IceBridge midpoint ({ICEBRIDGE_MID:.2f} m): {best_case}")


# ===================================================================
# Variance decomposition summary table
# ===================================================================

print("\n" + "=" * 72)
print("VARIANCE DECOMPOSITION SUMMARY")
print("=" * 72)
print(f"\n  {'Case':14s}  {'fb_IS2':>6s}  {'fb_CS2':>6s}  {'rho_s':>6s}  "
      f"{'rho_i':>6s}  {'delta':>6s}")
print("  " + "-" * 55)
for c in CASES:
    r = results_B[c['label']]
    print(f"  {c['label']:14s}  {r['fb_is2']:5.1f}%  {r['fb_cs2']:5.1f}%  "
          f"{r['rho_s']:5.1f}%  {r['rho_i']:5.1f}%  {r['delta']:5.1f}%")


# ===================================================================
# Save combined results
# ===================================================================

output = {
    'description': ('Western Weddell sub-sector sensitivity to FYI vs MYI '
                    'parameterisation, with 940 kg/m^3 cap'),
    'icebridge_benchmark': {'low': ICEBRIDGE_LOW, 'high': ICEBRIDGE_HIGH,
                             'mid': ICEBRIDGE_MID,
                             'reference': 'Kwok and Kacimi (2018)'},
    'best_match_case': best_case,
    'cap_kg_per_m3': RHO_I_UPPER_CAP,
    'experiment_A': results_A,
    'experiment_B': results_B,
}

json_path = f'{M3_OUTPUT}/R3_western_weddell_sensitivity_v1.json'
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {json_path}")


# ===================================================================
# FIGURE: Western Weddell sensitivity comparison
# ===================================================================

print("\n  Generating western Weddell sensitivity figure...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

case_labels = [c['label'] for c in CASES]
case_colors = ['#0072B2', '#56B4E9', '#D6A8B5', '#882255']  # FYI to MYI gradient

# Panel (a): h_i with 90% CI per case, with IceBridge benchmark band
ax = axes[0]
x = np.arange(len(CASES))
h_i_means = [results_A[c['label']]['h_i_mean'] for c in CASES]
h_i_p05 = [results_A[c['label']]['h_i_p05'] for c in CASES]
h_i_p95 = [results_A[c['label']]['h_i_p95'] for c in CASES]

# IceBridge benchmark shaded band
ax.axhspan(ICEBRIDGE_LOW, ICEBRIDGE_HIGH, color='#FFE599', alpha=0.6,
           label=f'IceBridge benchmark ({ICEBRIDGE_LOW}-{ICEBRIDGE_HIGH} m)',
           zorder=1)
ax.axhline(ICEBRIDGE_MID, color='#B45F06', ls='--', lw=1.0, alpha=0.7,
           zorder=2)

# Plot per-case mean and 90% CI
for i, (xi, m, p05, p95, col) in enumerate(zip(x, h_i_means, h_i_p05, h_i_p95,
                                                 case_colors)):
    ax.errorbar(xi, m, yerr=[[m - p05], [p95 - m]],
                fmt='o', color=col, ms=10, lw=2, capsize=8, capthick=2,
                zorder=4)
    ax.text(xi, m + 0.15, f'{m:.2f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=col, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(case_labels, fontsize=10)
ax.set_ylabel('Mean ice thickness $h_i$ (m)', fontsize=11)
ax.set_title('(a) Western Weddell mean h$_i$ vs parameterisation\n'
             '(error bars = mean 90% CI from per-segment MC)', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(h_i_p95) + 1.0)

# Panel (b): variance decomposition per case
ax = axes[1]
params_plot = ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta']
param_labels_plot = {
    'fb_is2': '$f_b^{IS2}$', 'fb_cs2': '$f_b^{CS2}$',
    'rho_s': '$\\rho_s$', 'rho_i': '$\\rho_i$', 'delta': '$\\delta$',
}
param_colors = {
    'fb_is2': '#56B4E9', 'fb_cs2': '#E69F00',
    'rho_s': '#F0E442', 'rho_i': '#009E73', 'delta': '#0072B2',
}

x_b = np.arange(len(CASES))
width = 0.16

for ip, p in enumerate(params_plot):
    vals = [results_B[c['label']][p] for c in CASES]
    bars = ax.bar(x_b + (ip - 2) * width, vals, width,
                  label=param_labels_plot[p], color=param_colors[p],
                  alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        if val > 1.0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=7,
                    color=param_colors[p], fontweight='bold')

ax.set_xticks(x_b)
ax.set_xticklabels(case_labels, fontsize=10)
ax.set_ylabel('Variance contribution (%)', fontsize=11)
ax.set_title('(b) Variance decomposition vs parameterisation', fontsize=11)
ax.legend(fontsize=9, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.10))
ax.grid(axis='y', alpha=0.2)
ax.set_ylim(0, 75)

fig.suptitle('Western Weddell Sub-sector Sensitivity to $\\rho_i$ Parameterisation\n'
             f'(four cases between Alexandrov FYI and MYI, all with 940 kg m$^{{-3}}$ cap; '
             f'$\\delta \\sim N$({DELTA_BASELINE}, {DELTA_SIGMA}))',
             fontsize=10, y=0.92)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(f'{M3_FIGURES}/R3_Fig_western_weddell_sensitivity.png')
fig.savefig(f'{M3_FIGURES}/R3_Fig_western_weddell_sensitivity.pdf')
print(f"    Saved: R3_Fig_western_weddell_sensitivity.png/pdf")
plt.close()

print(f"\nStep 02c complete in {time.time()-t0:.1f} s")

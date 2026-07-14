#!/usr/bin/env python3
"""
R3_Step02_v4_coupled.py — Experiments A + B with regional rho_i
===================================================================
Change from previous version (Alex 29 April 2026 refinement):
  - Per-segment regional rho_i parameterisation:
      Western Weddell: Alexandrov MYI (882, 23 kg/m^3)
      All other regions: Alexandrov FYI (916.7, 35.7 kg/m^3)
  - 940 kg/m^3 soft upper cap retained for all parameterisations
  - Weddell Sea aggregate now reflects the differentiated treatment
    automatically as the sample-weighted union of E. and W. Weddell
  - Print headers and saved fields updated to record regional treatment

Run: source ~/cryo2ice_env/bin/activate && python R3_Step02_v4_coupled.py
"""

import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

t0 = time.time()

print("=" * 72)
print("R3 Step 02 v4: Coupled delta-h_s with regional rho_i")
print("  *** Alex's 29 April 2026 refinement: per-segment Alexandrov ***")
print(f"  rho_i FYI (E. Weddell, Ross): mu={RHO_I_FYI_MU}, sigma={RHO_I_FYI_SIGMA} kg/m^3")
print(f"  rho_i MYI (W. Weddell only):  mu={RHO_I_MYI_MU}, sigma={RHO_I_MYI_SIGMA} kg/m^3")
print(f"  Soft upper cap at {RHO_I_UPPER_CAP} kg/m^3 (no lower bound)")
print(f"  Weddell aggregate = sample-weighted union of E. + W. Weddell")
print(f"  rho_s: Gaussian mu={RHO_S_CENTRAL}, sigma={RHO_S_SIGMA} (no clipping)")
print(f"  delta: Gaussian(mu={DELTA_BASELINE}, sigma={DELTA_SIGMA}), clipped {DELTA_CLIP}")
print(f"  rho_w = {RHO_W_CENTRAL} (FIXED), N_MC = {N_MC:,}")
print("=" * 72)

df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_verified.pkl')
n_total = len(df)

if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df['lon'].values)

is2_fb = df['is2_fb_wm'].values
cs2_fb = df['cs2_rfb'].values
sectors = df['sector'].values
subsectors = df['subsector'].values

assert np.all(is2_fb > 0), "ERROR: negative IS2 freeboards"
assert np.all(cs2_fb > 0), "ERROR: negative CS2 freeboards"

print(f"Loaded {n_total:,} matchups (positive-fb filtered)")
for r in REGIONS:
    m = get_region_mask(sectors, subsectors, r)
    print(f"  {r:12s}: {m.sum():,}")

# Print sub-sector composition of the Weddell aggregate
n_ww = (subsectors == 'Weddell_W').sum()
n_we = (subsectors == 'Weddell_E').sum()
n_wed_total = (sectors == 'Weddell').sum()
print(f"\nWeddell aggregate composition:")
print(f"  Eastern Weddell (FYI): {n_we:,} segments ({100*n_we/n_wed_total:.1f}%)")
print(f"  Western Weddell (MYI): {n_ww:,} segments ({100*n_ww/n_wed_total:.1f}%)")

# Pre-compute per-segment Alexandrov parameters for the entire dataset
mu_seg, sigma_seg = get_regional_rho_i_params(subsectors)
print(f"\nPer-segment rho_i parameter assignment:")
print(f"  Segments at FYI ({RHO_I_FYI_MU}, {RHO_I_FYI_SIGMA}): "
      f"{(mu_seg == RHO_I_FYI_MU).sum():,}")
print(f"  Segments at MYI ({RHO_I_MYI_MU}, {RHO_I_MYI_SIGMA}): "
      f"{(mu_seg == RHO_I_MYI_MU).sum():,}")

region_masks = {}
for r in REGIONS:
    region_masks[r] = get_region_mask(sectors, subsectors, r)

# ===================================================================
# EXPERIMENT A: Full simultaneous perturbation + histogram accumulation
# ===================================================================
print("\n" + "=" * 72)
print("EXPERIMENT A — Full simultaneous perturbation (5 params, coupled)")
print(f"  *** Per-segment regional rho_i (Alex 29 April 2026) ***")
print(f"  Accumulating full MC draw histograms for Figure 1")
print("=" * 72)

rng = np.random.default_rng(42)
hi_mean = np.zeros(n_total)
hi_std = np.zeros(n_total)
hi_p05 = np.zeros(n_total)
hi_p95 = np.zeros(n_total)

n_bins = len(HIST_BINS) - 1
mc_hist_counts = {r: np.zeros(n_bins, dtype=np.int64) for r in REGIONS}

batch = 10000
n_batches = (n_total + batch - 1) // batch

for b in range(n_batches):
    i0, i1 = b * batch, min((b + 1) * batch, n_total)
    nb = i1 - i0

    is2_mc = rng.normal(is2_fb[i0:i1], FB_SIGMA_RAW, size=(N_MC, nb))
    cs2_mc = rng.normal(cs2_fb[i0:i1], FB_SIGMA_RAW, size=(N_MC, nb))
    delta_mc = sample_delta(N_MC, rng).reshape(-1, 1)
    rho_s_mc = rng.normal(RHO_S_CENTRAL, RHO_S_SIGMA, size=(N_MC, 1))

    # Per-segment regional rho_i sampling
    mu_batch = mu_seg[i0:i1]
    sigma_batch = sigma_seg[i0:i1]
    rho_i_mc = rng.normal(loc=mu_batch[np.newaxis, :],
                          scale=sigma_batch[np.newaxis, :],
                          size=(N_MC, nb))
    rho_i_mc = np.minimum(rho_i_mc, RHO_I_UPPER_CAP)

    rho_w_mc = RHO_W_CENTRAL

    eta_s_mc = (1 + 0.51 * rho_s_mc / 1000.0)**1.5
    hs_mc = (is2_mc - cs2_mc) / (delta_mc * eta_s_mc)
    hi_mc = compute_ice_thickness(is2_mc, hs_mc, rho_s_mc, rho_i_mc, rho_w_mc)

    hi_mean[i0:i1] = np.mean(hi_mc, axis=0)
    hi_std[i0:i1] = np.std(hi_mc, axis=0)
    hi_p05[i0:i1] = np.percentile(hi_mc, 5, axis=0)
    hi_p95[i0:i1] = np.percentile(hi_mc, 95, axis=0)

    for r in REGIONS:
        batch_mask = region_masks[r][i0:i1]
        if batch_mask.sum() == 0:
            continue
        draws = hi_mc[:, batch_mask].ravel()
        counts, _ = np.histogram(draws, bins=HIST_BINS)
        mc_hist_counts[r] += counts

    if (b + 1) % 5 == 0 or b == n_batches - 1:
        print(f"  Exp A: {i1:,}/{n_total:,} ({100*i1/n_total:.0f}%, {time.time()-t0:.0f}s)")

ci_width = hi_p95 - hi_p05

print(f"\n--- Experiment A Results (regional Alexandrov, 940 cap) ---")
for r in REGIONS:
    m = get_region_mask(sectors, subsectors, r)
    print(f"  {r:12s} (n={m.sum():,}): h_i={hi_mean[m].mean():.3f} +/- {hi_std[m].mean():.3f} m, "
          f"90%CI=[{hi_p05[m].mean():.3f},{hi_p95[m].mean():.3f}], "
          f"width={ci_width[m].mean():.3f} m")

print(f"\n--- MC Draw Histogram Totals ---")
for r in REGIONS:
    total_draws = mc_hist_counts[r].sum()
    expected = region_masks[r].sum() * N_MC
    print(f"  {r:12s}: {total_draws:,} draws accumulated (expected: {expected:,})")

save_dict = dict(
    hi_mean=hi_mean, hi_std=hi_std, hi_p05=hi_p05, hi_p95=hi_p95,
    ci_width=ci_width, sectors=sectors, subsectors=subsectors,
    rho_i_mu_seg=mu_seg, rho_i_sigma_seg=sigma_seg,
    rho_i_fyi_mu=RHO_I_FYI_MU, rho_i_fyi_sigma=RHO_I_FYI_SIGMA,
    rho_i_myi_mu=RHO_I_MYI_MU, rho_i_myi_sigma=RHO_I_MYI_SIGMA,
    rho_i_upper_cap=RHO_I_UPPER_CAP,
    delta_sigma=DELTA_SIGMA,
    hist_bins=HIST_BINS,
    parameterisation_note='Regional Alexandrov: FYI for E.Weddell+Ross, MYI for W.Weddell',
)
for r in REGIONS:
    save_dict[f'mc_hist_{r}'] = mc_hist_counts[r]

np.savez_compressed(f'{M3_OUTPUT}/R3_ExpA_d070_v4.npz', **save_dict)
print(f"  Saved: {M3_OUTPUT}/R3_ExpA_d070_v4.npz")


# ===================================================================
# EXPERIMENT B: Variance decomposition
# ===================================================================
print("\n" + "=" * 72)
print("EXPERIMENT B — Variance decomposition (5 params)")
print(f"  *** Per-segment regional rho_i (Alex 29 April 2026) ***")
print("=" * 72)

n_sub = min(5000, n_total)
idx_sub = rng.choice(n_total, n_sub, replace=False)
idx_sub.sort()

is2_c = is2_fb[idx_sub]
cs2_c = cs2_fb[idx_sub]
sectors_sub = sectors[idx_sub]
subsectors_sub = subsectors[idx_sub]

# Per-segment regional parameters for the variance subsample
mu_sub, sigma_sub = get_regional_rho_i_params(subsectors_sub)

params = ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta']

var_contributions = {p: np.zeros(n_sub) for p in params}

for ip, param in enumerate(params):
    rng_p = np.random.default_rng(300 + ip)
    hi_samples = np.zeros((N_MC, n_sub))

    for trial in range(N_MC):
        is2_t = is2_c.copy()
        cs2_t = cs2_c.copy()
        rho_s_t = RHO_S_CENTRAL
        # Default rho_i for "no perturbation" case: per-segment central value
        rho_i_t = mu_sub.copy()
        delta_t = DELTA_BASELINE

        if param == 'fb_is2':
            is2_t += rng_p.normal(0, FB_SIGMA_RAW, n_sub)
        elif param == 'fb_cs2':
            cs2_t += rng_p.normal(0, FB_SIGMA_RAW, n_sub)
        elif param == 'rho_s':
            rho_s_t = rng_p.normal(RHO_S_CENTRAL, RHO_S_SIGMA)
        elif param == 'rho_i':
            # Per-segment regional sampling
            rho_i_draws = rng_p.normal(loc=mu_sub, scale=sigma_sub)
            rho_i_t = np.minimum(rho_i_draws, RHO_I_UPPER_CAP)
        elif param == 'delta':
            delta_t = sample_delta(1, rng_p)[0]

        eta_t = (1 + 0.51 * (rho_s_t if isinstance(rho_s_t, (int, float))
                 else rho_s_t) / 1000.0)**1.5
        hs_t = (is2_t - cs2_t) / (delta_t * eta_t)
        hi_samples[trial] = compute_ice_thickness(is2_t, hs_t, rho_s_t,
                                                   rho_i_t, RHO_W_CENTRAL)

    var_contributions[param] = np.var(hi_samples, axis=0)

total_var = sum(var_contributions.values())

results_B = {}
print(f"\n  {'Param':12s}  {'All':>7s}  {'Weddell':>7s}  {'Wed_W':>7s}  {'Wed_E':>7s}  {'Ross':>7s}")
print("  " + "-" * 60)
for param in params:
    row = {}
    for r in REGIONS:
        m = get_region_mask(sectors_sub, subsectors_sub, r)
        if m.sum() > 0:
            row[r] = float(np.mean(var_contributions[param][m]) /
                           np.mean(total_var[m]) * 100)
        else:
            row[r] = 0.0
    row['all'] = float(np.mean(var_contributions[param]) /
                       np.mean(total_var) * 100)
    results_B[param] = row
    print(f"  {param:12s}  {row['all']:6.1f}%  {row.get('Weddell',0):6.1f}%  "
          f"{row.get('Weddell_W',0):6.1f}%  {row.get('Weddell_E',0):6.1f}%  "
          f"{row.get('Ross',0):6.1f}%")

fb_all = results_B['fb_is2']['all'] + results_B['fb_cs2']['all']
print(f"  {'fb_combined':12s}  {fb_all:6.1f}%")

with open(f'{M3_OUTPUT}/R3_ExpB_d070_v4.json', 'w') as f:
    json.dump(results_B, f, indent=2)
print(f"\n  Saved: {M3_OUTPUT}/R3_ExpB_d070_v4.json")

print(f"\nStep 02 v4 complete in {time.time()-t0:.1f} s")

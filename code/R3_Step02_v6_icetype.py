#!/usr/bin/env python3
"""
R3_Step02_v6_icetype.py
=======================
Derived from R3_Step02_v5_coupled_20260830.py. Method unchanged.

Change: the sea-ice density prior is assigned per matchup from the observed
ice type (Melsheimer et al. 2023, collocated on the matchup date by Step 13)
instead of by longitude sub-sector. Experiment B is additionally reported for
the two observed ice-type strata (MYI_obs, FYI_obs) pooled across sectors.

Every difference between the _v6 and _v5 outputs is attributable to the
ice-type classification alone. Addresses Reviewer 3 major comment 7 and
Reviewer 2 other comments 3-4 (RSE-D-26-03119).

Input : {M3_OUTPUT}/R3_input_d070_icetype.pkl   (from R3_Step13)
Output: {M3_OUTPUT}/R3_ExpA_d070_v6.npz, {M3_OUTPUT}/R3_ExpB_d070_v6.json
Author: Xinlong Liu | September 2026
"""

import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

t0 = time.time()

print("=" * 72)
print("R3 Step 02 v6: Coupled delta-h_s, unnormalised shares, OBSERVED ice type")
print(f"  rho_i FYI: mu={RHO_I_FYI_MU}, sigma={RHO_I_FYI_SIGMA} kg/m^3")
print(f"  rho_i MYI: mu={RHO_I_MYI_MU}, sigma={RHO_I_MYI_SIGMA} kg/m^3")
print(f"  Prior assigned per matchup: MYI where Melsheimer MYI >= {MYI_THRESHOLD_PCT:.0f} %")
print(f"  Soft upper cap at {RHO_I_UPPER_CAP} kg/m^3 (no lower bound)")
print(f"  rho_s: Gaussian mu={RHO_S_CENTRAL}, sigma={RHO_S_SIGMA} (no clipping)")
print(f"  delta: Gaussian(mu={DELTA_BASELINE}, sigma={DELTA_SIGMA}), clipped {DELTA_CLIP}")
print(f"  rho_w = {RHO_W_CENTRAL} (FIXED), N_MC = {N_MC:,}")
print("=" * 72)

df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_icetype.pkl')
n_total = len(df)

if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df['lon'].values)
assert 'icetype' in df.columns, "Run R3_Step13_icetype_collocation.py first."

is2_fb = df['is2_fb_wm'].values
cs2_fb = df['cs2_rfb'].values
sectors = df['sector'].values
subsectors = df['subsector'].values
icetype = df['icetype'].values

assert np.all(is2_fb > 0), "ERROR: negative IS2 freeboards"
assert np.all(cs2_fb > 0), "ERROR: negative CS2 freeboards"

print(f"Loaded {n_total:,} matchups (positive-fb filtered)")
for r in REGIONS_V6:
    m = get_region_mask_v6(sectors, subsectors, icetype, r)
    print(f"  {r:12s}: {m.sum():,}")

# Prior assignment: observed ice type (v6) versus longitude rule (v5)
mu_seg, sigma_seg = get_icetype_rho_i_params(icetype)
mu_v5, _ = get_regional_rho_i_params(subsectors)
changed = mu_seg != mu_v5
print(f"\nPer-segment rho_i prior (observed ice type):")
print(f"  MYI prior: {(mu_seg == RHO_I_MYI_MU).sum():,}   FYI prior: {(mu_seg == RHO_I_FYI_MU).sum():,}")
print(f"  Unclassified (given FYI): {(icetype == 'unclassified').sum():,}")
print(f"  Segments whose prior differs from the v5 longitude rule: {changed.sum():,} "
      f"({100 * changed.mean():.1f} %)")
for r in REGIONS:
    m = get_region_mask(sectors, subsectors, r)
    print(f"    {r:12s}: {changed[m].sum():,} of {m.sum():,} changed "
          f"({100 * changed[m].mean():.1f} %)")

region_masks = {r: get_region_mask_v6(sectors, subsectors, icetype, r) for r in REGIONS_V6}

# ===================================================================
# EXPERIMENT A: Full simultaneous perturbation + histogram accumulation
# ===================================================================
print("\n" + "=" * 72)
print("EXPERIMENT A — Full simultaneous perturbation (5 params, coupled)")
print("  Per-matchup rho_i prior from observed ice type")
print("=" * 72)

rng = np.random.default_rng(42)
hi_mean = np.zeros(n_total)
hi_std = np.zeros(n_total)
hi_p05 = np.zeros(n_total)
hi_p95 = np.zeros(n_total)

n_bins = len(HIST_BINS) - 1
mc_hist_counts = {r: np.zeros(n_bins, dtype=np.int64) for r in REGIONS_V6}

batch = 10000
n_batches = (n_total + batch - 1) // batch

for b in range(n_batches):
    i0, i1 = b * batch, min((b + 1) * batch, n_total)
    nb = i1 - i0

    is2_mc = rng.normal(is2_fb[i0:i1], FB_SIGMA_RAW, size=(N_MC, nb))
    cs2_mc = rng.normal(cs2_fb[i0:i1], FB_SIGMA_RAW, size=(N_MC, nb))
    delta_mc = sample_delta(N_MC, rng).reshape(-1, 1)
    rho_s_mc = rng.normal(RHO_S_CENTRAL, RHO_S_SIGMA, size=(N_MC, 1))

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

    for r in REGIONS_V6:
        batch_mask = region_masks[r][i0:i1]
        if batch_mask.sum() == 0:
            continue
        draws = hi_mc[:, batch_mask].ravel()
        counts, _ = np.histogram(draws, bins=HIST_BINS)
        mc_hist_counts[r] += counts

    if (b + 1) % 5 == 0 or b == n_batches - 1:
        print(f"  Exp A: {i1:,}/{n_total:,} ({100*i1/n_total:.0f}%, {time.time()-t0:.0f}s)")

ci_width = hi_p95 - hi_p05

print(f"\n--- Experiment A Results (observed ice type, 940 cap) ---")
for r in REGIONS_V6:
    m = region_masks[r]
    print(f"  {r:12s} (n={m.sum():,}): h_i={hi_mean[m].mean():.3f} +/- {hi_std[m].mean():.3f} m, "
          f"90%CI=[{hi_p05[m].mean():.3f},{hi_p95[m].mean():.3f}], "
          f"width={ci_width[m].mean():.3f} m")

print(f"\n--- MC Draw Histogram Totals ---")
for r in REGIONS_V6:
    total_draws = mc_hist_counts[r].sum()
    expected = region_masks[r].sum() * N_MC
    print(f"  {r:12s}: {total_draws:,} draws accumulated (expected: {expected:,})")

save_dict = dict(
    hi_mean=hi_mean, hi_std=hi_std, hi_p05=hi_p05, hi_p95=hi_p95,
    ci_width=ci_width, sectors=sectors, subsectors=subsectors, icetype=icetype,
    myi_frac=df['myi_frac'].values,
    rho_i_mu_seg=mu_seg, rho_i_sigma_seg=sigma_seg,
    rho_i_fyi_mu=RHO_I_FYI_MU, rho_i_fyi_sigma=RHO_I_FYI_SIGMA,
    rho_i_myi_mu=RHO_I_MYI_MU, rho_i_myi_sigma=RHO_I_MYI_SIGMA,
    rho_i_upper_cap=RHO_I_UPPER_CAP,
    delta_sigma=DELTA_SIGMA,
    hist_bins=HIST_BINS,
    myi_threshold_pct=MYI_THRESHOLD_PCT,
    parameterisation_note=('Per-matchup ice type from Melsheimer et al. (2023), '
                           f'MYI prior at >= {MYI_THRESHOLD_PCT:.0f} % MYI concentration'),
)
for r in REGIONS_V6:
    save_dict[f'mc_hist_{r}'] = mc_hist_counts[r]

np.savez_compressed(f'{M3_OUTPUT}/R3_ExpA_d070_v6.npz', **save_dict)
print(f"  Saved: {M3_OUTPUT}/R3_ExpA_d070_v6.npz")


# ===================================================================
# EXPERIMENT B: Variance decomposition
# ===================================================================
print("\n" + "=" * 72)
print("EXPERIMENT B — Variance decomposition (5 params)")
print("  Per-matchup rho_i prior from observed ice type")
print("=" * 72)

n_sub = min(5000, n_total)
idx_sub = rng.choice(n_total, n_sub, replace=False)
idx_sub.sort()

is2_c = is2_fb[idx_sub]
cs2_c = cs2_fb[idx_sub]
sectors_sub = sectors[idx_sub]
subsectors_sub = subsectors[idx_sub]
icetype_sub = icetype[idx_sub]

mu_sub, sigma_sub = get_icetype_rho_i_params(icetype_sub)

params = ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta']

var_contributions = {p: np.zeros(n_sub) for p in params}

for ip, param in enumerate(params):
    rng_p = np.random.default_rng(300 + ip)
    hi_samples = np.zeros((N_MC, n_sub))

    for trial in range(N_MC):
        is2_t = is2_c.copy()
        cs2_t = cs2_c.copy()
        rho_s_t = RHO_S_CENTRAL
        rho_i_t = mu_sub.copy()
        delta_t = DELTA_BASELINE

        if param == 'fb_is2':
            is2_t += rng_p.normal(0, FB_SIGMA_RAW, n_sub)
        elif param == 'fb_cs2':
            cs2_t += rng_p.normal(0, FB_SIGMA_RAW, n_sub)
        elif param == 'rho_s':
            rho_s_t = rng_p.normal(RHO_S_CENTRAL, RHO_S_SIGMA)
        elif param == 'rho_i':
            rho_i_draws = rng_p.normal(loc=mu_sub, scale=sigma_sub)
            rho_i_t = np.minimum(rho_i_draws, RHO_I_UPPER_CAP)
        elif param == 'delta':
            delta_t = sample_delta(1, rng_p)[0]

        eta_t = (1 + 0.51 * rho_s_t / 1000.0)**1.5
        hs_t = (is2_t - cs2_t) / (delta_t * eta_t)
        hi_samples[trial] = compute_ice_thickness(is2_t, hs_t, rho_s_t,
                                                   rho_i_t, RHO_W_CENTRAL)

    var_contributions[param] = np.var(hi_samples, axis=0)

var_sim_sub = hi_std[idx_sub] ** 2
assert np.all(var_sim_sub > 0), \
    "Experiment A variance is zero — Experiment B cannot be normalised."

sum_of_parts = sum(var_contributions.values())

recovery_pct = float(np.mean(sum_of_parts) / np.mean(var_sim_sub) * 100)
interaction_pct = 100.0 - recovery_pct
print(f"\n  Single-parameter variances recover {recovery_pct:.1f}% of the "
      f"simultaneous variance.")
print(f"  Residual interaction term: {interaction_pct:.1f}% "
      f"(NOT redistributed).\n")

# ── Pairwise interaction attribution ──────────────────────────────────
PAIRS = [('delta', 'rho_i'), ('delta', 'rho_s'), ('rho_s', 'rho_i')]
print("  Pairwise interaction attribution:")
pair_pct = {}
for ipair, (pa, pb) in enumerate(PAIRS):
    rng_pair = np.random.default_rng(400 + ipair)
    hi_pair = np.zeros((N_MC, n_sub))
    for trial in range(N_MC):
        rho_s_t = RHO_S_CENTRAL
        rho_i_t = mu_sub.copy()
        delta_t = DELTA_BASELINE
        for p in (pa, pb):
            if p == 'rho_s':
                rho_s_t = rng_pair.normal(RHO_S_CENTRAL, RHO_S_SIGMA)
            elif p == 'rho_i':
                rho_i_t = np.minimum(
                    rng_pair.normal(loc=mu_sub, scale=sigma_sub),
                    RHO_I_UPPER_CAP)
            elif p == 'delta':
                delta_t = sample_delta(1, rng_pair)[0]
        eta_t = (1 + 0.51 * rho_s_t / 1000.0) ** 1.5
        hs_t = (is2_c - cs2_c) / (delta_t * eta_t)
        hi_pair[trial] = compute_ice_thickness(is2_c, hs_t, rho_s_t,
                                               rho_i_t, RHO_W_CENTRAL)
    inter = (np.var(hi_pair, axis=0)
             - var_contributions[pa] - var_contributions[pb])
    pair_pct[f'{pa}_{pb}'] = float(100 * np.mean(inter)
                                   / np.mean(var_sim_sub))
    print(f"    {pa:>6s} x {pb:<6s}: "
          f"{pair_pct[f'{pa}_{pb}']:6.2f}% of simultaneous variance")

print(f"    {'sum of pairs':>15s}: {sum(pair_pct.values()):6.2f}%")
print(f"    {'total residual':>15s}: {interaction_pct:6.2f}%")
print(f"    {'higher-order':>15s}: "
      f"{interaction_pct - sum(pair_pct.values()):6.2f}%\n")

# ── Shares by region and by observed ice type ─────────────────────────
results_B = {}
hdr = f"\n  {'Param':12s}  {'All':>7s}" + "".join(f"  {r:>9s}" for r in REGIONS_V6)
print(hdr)
print("  " + "-" * (len(hdr) - 3))
for param in params:
    row = {}
    for r in REGIONS_V6:
        m = get_region_mask_v6(sectors_sub, subsectors_sub, icetype_sub, r)
        row[r] = (float(np.mean(var_contributions[param][m]) /
                        np.mean(var_sim_sub[m]) * 100) if m.sum() > 0 else 0.0)
    row['all'] = float(np.mean(var_contributions[param]) /
                       np.mean(var_sim_sub) * 100)
    results_B[param] = row
    print(f"  {param:12s}  {row['all']:6.1f}%"
          + "".join(f"  {row[r]:8.1f}%" for r in REGIONS_V6))

fb_all = results_B['fb_is2']['all'] + results_B['fb_cs2']['all']
print(f"  {'fb_combined':12s}  {fb_all:6.1f}%")

results_B['n_sub_by_region'] = {
    r: int(get_region_mask_v6(sectors_sub, subsectors_sub, icetype_sub, r).sum())
    for r in REGIONS_V6}
results_B['recovery_pct'] = recovery_pct
results_B['interaction_pct'] = interaction_pct
results_B['pairwise_interaction_pct'] = pair_pct
results_B['higher_order_pct'] = interaction_pct - sum(pair_pct.values())
results_B['n_prior_changed_vs_v5'] = int(changed.sum())
results_B['note'] = ('Shares unnormalised: divided by the Experiment A simultaneous '
                     'variance. rho_i prior per matchup from observed ice type '
                     f'(Melsheimer et al. 2023, MYI at >= {MYI_THRESHOLD_PCT:.0f} %).')

with open(f'{M3_OUTPUT}/R3_ExpB_d070_v6.json', 'w') as f:
    json.dump(results_B, f, indent=2)
print(f"\n  Saved: {M3_OUTPUT}/R3_ExpB_d070_v6.json")

# ── v6 versus v5 (same seeds, same subsample; only the prior differs) ──
v5_path = f'{M3_OUTPUT}/R3_ExpB_d070_v5.json'
if os.path.exists(v5_path):
    with open(v5_path) as f:
        v5 = json.load(f)
    print(f"\n  Experiment B shares, v6 minus v5 (percentage points):")
    print(f"  {'Param':12s}  {'All':>7s}" + "".join(f"  {r:>9s}" for r in REGIONS))
    for param in params:
        print(f"  {param:12s}  {results_B[param]['all'] - v5[param]['all']:+6.1f} "
              + "".join(f"  {results_B[param][r] - v5[param][r]:+8.1f} " for r in REGIONS))
    print(f"  recovery_pct  v5 {v5['recovery_pct']:.1f}  ->  v6 {recovery_pct:.1f}")
else:
    print(f"\n  ({v5_path} not found; skipping v5 comparison)")

print(f"\nStep 02 v6 complete in {time.time()-t0:.1f} s")

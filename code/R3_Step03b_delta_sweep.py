#!/usr/bin/env python3
"""
R3_Step03b_delta_sweep.py — δ sensitivity sweep (full dataset, 10,000 MC draws)
================================================================================
Companion to R3_Step03_v3_coupled_sweep.py (which sweeps σ_ρᵢ).

This script sweeps the penetration factor uncertainty σ_δ while holding all
other parameter distributions at their primary values, using the same
one-at-a-time variance decomposition methodology.

*** REVISED 14 April 2026: ***
  - Extended from 5,000 subsampled segments to ALL 65,165 positive-fb segments
  - Extended from 5,000 MC draws to 10,000 MC draws per parameter perturbation
  - All four regions: Weddell, Western Weddell, Eastern Weddell, Ross
  - Memory-efficient batching (10,000 segments per batch)

σ_δ values tested: [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
  - 0.02: near-deterministic (99.7% of draws in [0.64, 0.76]); represents the
          precision achievable with dual-frequency Ka–Ku snow-depth constraint
  - 0.05: well-constrained; consistent with the empirical range δ* ≈ 0.55–0.70
          from Chapter 4, implying σ_δ ≈ 0.05–0.08 if δ can be measured
  - 0.10: previous analysis value before Alex increased to 0.20 (4 Apr meeting)
  - 0.15: moderate uncertainty
  - 0.20: current primary analysis value (Alex's instruction)
  - 0.25: substantial uncertainty
  - 0.30: large (68% of draws span [0.40, 1.00] before clipping)
  - 0.40: extreme (~50% of draws hit clip boundaries at [0.30, 1.00])

All other parameters fixed at primary values:
  IS2/CS2 freeboard: Gaussian σ = 0.030 m
  ρ_s: Gaussian μ = 330, σ = 70 kg/m³ (NO clipping)
  ρ_i: Trunc. normal μ = 885 (eff ≈ 880), σ = 25, [750, 917]
  ρ_w: FIXED at 1024

Methodology:
  One-at-a-time perturbation: for each σ_δ value, each of the 5 parameters
  is perturbed individually while all others are held at central values.
  When δ is perturbed, snow thickness is re-derived from the (unperturbed)
  freeboards at each drawn δ value (coupled δ–hₛ approach). The fractional
  variance contribution of each parameter is computed per segment and
  averaged within each region.

Expected runtime: ~30–60 minutes on Gadi (8 σ_δ values × 5 params ×
  10,000 draws × 65,165 segments). Memory requirement: ~1.2 GB peak
  (batched at 10,000 segments).

Input:  R3_input_d070_verified.pkl (from Step 01)
Output: R3_delta_sweep_v1.csv

Run: source ~/cryo2ice_env/bin/activate && python R3_Step03b_delta_sweep.py
"""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

t0 = time.time()

DELTA_SIGMAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
N_MC_SWEEP = 10_000
BATCH_SIZE = 10_000  # segments per batch to manage memory

total_configs = len(DELTA_SIGMAS)
print("=" * 70)
print("R3 Step 03b: δ Sensitivity Sweep (FULL dataset, 10,000 MC draws)")
print(f"  {total_configs} σ_δ values × {len(REGIONS)} regions")
print(f"  σ_δ range: {DELTA_SIGMAS}")
print(f"  N_MC = {N_MC_SWEEP:,} draws per parameter per segment")
print(f"  Batch size = {BATCH_SIZE:,} segments")
print(f"  ρ_i: truncnorm μ={RHO_I_CONFIGURED_MU}, σ={RHO_I_SIGMA_DEFAULT} (FIXED)")
print(f"  ρ_s: Gaussian μ={RHO_S_CENTRAL}, σ={RHO_S_SIGMA} (NO clipping, FIXED)")
print(f"  ρ_w = {RHO_W_CENTRAL} (FIXED)")
print("=" * 70)

# ── Load full dataset (positive-freeboard filtered) ──
df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_verified.pkl')
if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df['lon'].values)

n_total = len(df)
is2_fb = df['is2_fb_wm'].values
cs2_fb = df['cs2_rfb'].values
sectors = df['sector'].values
subsectors = df['subsector'].values

assert np.all(is2_fb > 0), "ERROR: negative IS2 freeboards detected"
assert np.all(cs2_fb > 0), "ERROR: negative CS2 freeboards detected"

print(f"\nLoaded {n_total:,} segments (positive-fb filtered)")
for r in REGIONS:
    m = get_region_mask(sectors, subsectors, r)
    print(f"  {r:12s}: {m.sum():,}")

n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
print(f"  Processing in {n_batches} batches of up to {BATCH_SIZE:,} segments")

# ── Precompute region masks ──
region_masks = {}
for r in REGIONS:
    region_masks[r] = get_region_mask(sectors, subsectors, r)

params = ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta']
results = []
config_count = 0

# ── Main sweep loop ──
for delta_sigma in DELTA_SIGMAS:
    config_count += 1
    t_config = time.time()
    print(f"\n{'='*70}")
    print(f"  Config {config_count}/{total_configs}: σ_δ = {delta_sigma:.2f}")
    print(f"{'='*70}")

    var_by_param = {p: np.zeros(n_total) for p in params}

    for ip, param in enumerate(params):
        t_param = time.time()
        rng_p = np.random.default_rng(3000 * config_count + ip)

        for b in range(n_batches):
            i0 = b * BATCH_SIZE
            i1 = min((b + 1) * BATCH_SIZE, n_total)
            nb = i1 - i0

            # hi_samples shape: (N_MC_SWEEP, nb) — ~0.8 GB max at nb=10000
            hi_samples = np.zeros((N_MC_SWEEP, nb))

            for trial in range(N_MC_SWEEP):
                is2_t = is2_fb[i0:i1].copy()
                cs2_t = cs2_fb[i0:i1].copy()
                rho_s_t = RHO_S_CENTRAL
                rho_i_t = float(RHO_I_CONFIGURED_MU)
                delta_t = DELTA_BASELINE

                if param == 'fb_is2':
                    is2_t += rng_p.normal(0, FB_SIGMA_RAW, nb)
                elif param == 'fb_cs2':
                    cs2_t += rng_p.normal(0, FB_SIGMA_RAW, nb)
                elif param == 'rho_s':
                    rho_s_t = rng_p.normal(RHO_S_CENTRAL, RHO_S_SIGMA)
                elif param == 'rho_i':
                    rho_i_t = sample_density('truncnorm', RHO_I_CONFIGURED_MU,
                                             RHO_I_SIGMA_DEFAULT, 1, rng_p)[0]
                elif param == 'delta':
                    # *** THIS IS THE SWEPT PARAMETER ***
                    delta_t = rng_p.normal(DELTA_BASELINE, delta_sigma)
                    delta_t = np.clip(delta_t, DELTA_CLIP[0], DELTA_CLIP[1])

                eta_t = (1 + 0.51 * (rho_s_t if isinstance(rho_s_t, (int, float))
                         else rho_s_t) / 1000.0)**1.5
                hs_t = (is2_t - cs2_t) / (delta_t * eta_t)
                hi_samples[trial] = compute_ice_thickness(
                    is2_t, hs_t, rho_s_t, rho_i_t, RHO_W_CENTRAL)

            var_by_param[param][i0:i1] = np.var(hi_samples, axis=0)

            # Free memory explicitly
            del hi_samples

        dt_param = time.time() - t_param
        print(f"    {param:8s}: {dt_param:.1f}s "
              f"({n_total:,} segments × {N_MC_SWEEP:,} draws)")

    total_var = sum(var_by_param.values())

    for region in REGIONS:
        mask = region_masks[region]
        n_region = mask.sum()
        if n_region < 10:
            continue
        tv = np.mean(total_var[mask])
        row = {
            'delta_sigma': delta_sigma,
            'region': region,
            'n': int(n_region),
        }
        for p in params:
            row[f'frac_{p}'] = float(
                np.mean(var_by_param[p][mask]) / tv * 100) if tv > 0 else 0
        row['frac_fb_combined'] = row['frac_fb_is2'] + row['frac_fb_cs2']
        results.append(row)

    dt_config = time.time() - t_config
    print(f"\n    Completed σ_δ={delta_sigma:.2f} in {dt_config:.1f}s "
          f"({dt_config/60:.1f} min)")

    # Print interim results for this σ_δ
    for region in REGIONS:
        row = [r for r in results if r['delta_sigma'] == delta_sigma
               and r['region'] == region]
        if row:
            r = row[0]
            print(f"      {region:12s}: δ={r['frac_delta']:5.1f}%  "
                  f"ρ_i={r['frac_rho_i']:5.1f}%  ρ_s={r['frac_rho_s']:5.1f}%  "
                  f"CS2={r['frac_fb_cs2']:5.1f}%  IS2={r['frac_fb_is2']:5.1f}%")

# ── Save results ──
df_results = pd.DataFrame(results)
csv_path = f'{M3_OUTPUT}/R3_delta_sweep_v1.csv'
df_results.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path} ({len(df_results)} rows)")

# ── Summary table ──
print("\n" + "=" * 70)
print(f"FULL SUMMARY: δ sweep (N_MC={N_MC_SWEEP:,}, n_segments={n_total:,})")
print("=" * 70)

for region in REGIONS:
    sub = df_results[df_results['region'] == region]
    if len(sub) == 0:
        continue
    print(f"\n  {REGION_LABELS[region]} (n={sub.iloc[0]['n']:,}):")
    print(f"  {'σ_δ':>6s}  {'fb_IS2':>6s}  {'fb_CS2':>6s}  {'ρ_s':>6s}  "
          f"{'ρ_i':>6s}  {'δ':>6s}  {'fb_comb':>7s}")
    print("  " + "-" * 55)
    for _, r in sub.sort_values('delta_sigma').iterrows():
        marker = " ◄ primary" if abs(r['delta_sigma'] - 0.20) < 0.001 else ""
        print(f"  {r['delta_sigma']:6.2f}  {r['frac_fb_is2']:5.1f}%  "
              f"{r['frac_fb_cs2']:5.1f}%  {r['frac_rho_s']:5.1f}%  "
              f"{r['frac_rho_i']:5.1f}%  {r['frac_delta']:5.1f}%  "
              f"{r['frac_fb_combined']:6.1f}%{marker}")

# ── Key interpretation ──
print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)
for region in REGIONS:
    sub = df_results[df_results['region'] == region].sort_values('delta_sigma')
    if len(sub) == 0:
        continue

    # Find the σ_δ at which δ first overtakes ρ_i
    crossover = None
    for _, r in sub.iterrows():
        if r['frac_delta'] > r['frac_rho_i']:
            crossover = r['delta_sigma']
            break

    # Find δ contribution at primary σ_δ = 0.20
    primary = sub[sub['delta_sigma'].between(0.195, 0.205)]
    d_pct = primary.iloc[0]['frac_delta'] if len(primary) > 0 else float('nan')
    r_pct = primary.iloc[0]['frac_rho_i'] if len(primary) > 0 else float('nan')

    # Find δ contribution at near-deterministic σ_δ = 0.02
    determ = sub[sub['delta_sigma'].between(0.015, 0.025)]
    d_determ = determ.iloc[0]['frac_delta'] if len(determ) > 0 else float('nan')
    r_determ = determ.iloc[0]['frac_rho_i'] if len(determ) > 0 else float('nan')

    print(f"\n  {REGION_LABELS[region]:18s}:")
    if crossover is not None:
        print(f"    δ overtakes ρ_i at σ_δ ≈ {crossover:.2f}")
    else:
        print(f"    δ never overtakes ρ_i in tested range (ρ_i dominates throughout)")
    print(f"    At σ_δ = 0.02 (near-deterministic): δ = {d_determ:.1f}%, ρ_i = {r_determ:.1f}%")
    print(f"    At σ_δ = 0.20 (primary):            δ = {d_pct:.1f}%, ρ_i = {r_pct:.1f}%")

total_elapsed = time.time() - t0
print(f"\nStep 03b complete in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

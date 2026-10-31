#!/usr/bin/env python3
"""
R3_Step03_v7_rho_sweep.py — sea-ice-density uncertainty sweep, mixture prior
==============================================================================
Replaces R3_Step03_v3_coupled_sweep.py (single FYI prior everywhere, shares
renormalised to 100 %). Sweeps the first year spread sigma_FYI over
RHO_I_SIGMAS = [15, 20, 25, 30, 35.7, 40] kg/m^3 with the multi-year spread
scaled in proportion (sigma_MYI = sigma_FYI x 23.0/35.7), so the mixture prior
of Step 02 v7 is recovered exactly at sigma_FYI = 35.7.

All other inputs at their primary values (delta ~ N(0.70, 0.20) clipped;
rho_s ~ N(330, 70) unbounded; freeboards sigma = 0.030 m; rho_w = 1024).
Shares are fractions of the simultaneous-perturbation variance, un-renormalised.
Subsample and seeds identical to Step 02 v7 (see R3_sweep_engine_v7).

The four single-parameter variances that do not depend on sigma_rho_i are
computed once; only the rho_i perturbation and the simultaneous variance are
repeated per sweep point.

Input : {M3_OUTPUT}/R3_input_d070_icetype.pkl, R3_ExpB_d070_v7.json (for check)
Output: {M3_OUTPUT}/R3_density_sweep_d070_v7.csv
Run   : source ~/cryo2ice_env/bin/activate && python R3_Step03_v7_rho_sweep.py
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *
from R3_sweep_engine_v7 import (PARAMS, BASE_SEED, load_subsample, single_param_variance,
                                simultaneous_variance, shares_by_region,
                                print_region_table, check_against_step02)

t0 = time.time()
SIGMA_FYI_GRID = list(RHO_I_SIGMAS)                 # [15, 20, 25, 30, 35.7, 40]

print("=" * 72)
print("R3 Step 03 v7: sigma_rho_i sweep with per-matchup MIXTURE rho_i prior")
print(f"  sigma_FYI grid: {SIGMA_FYI_GRID} kg/m^3 (sigma_MYI scaled by {RHO_I_MYI_SIGMA/RHO_I_FYI_SIGMA:.3f})")
print(f"  delta ~ N({DELTA_BASELINE}, {DELTA_SIGMA}) clipped {DELTA_CLIP}; rho_s ~ N({RHO_S_CENTRAL}, {RHO_S_SIGMA})")
print(f"  N_MC = {N_MC:,}; strata: {REGIONS_V6}")
print("=" * 72)

sub = load_subsample()

# Parameters whose one-at-a-time variance is independent of sigma_rho_i
print("\n  Fixed single-parameter perturbations (once):")
var_fixed = {}
for p in ['fb_is2', 'fb_cs2', 'rho_s', 'delta']:
    t = time.time()
    var_fixed[p] = single_param_variance(sub, p, BASE_SEED[p])
    print(f"    {p:7s}: {time.time() - t:5.0f}s")

rows = []
for k, s_fyi in enumerate(SIGMA_FYI_GRID):
    t = time.time()
    scale = s_fyi / RHO_I_FYI_SIGMA
    baseline = np.isclose(s_fyi, RHO_I_FYI_SIGMA)
    seed_i = BASE_SEED['rho_i'] if baseline else 2000 + k        # exact Step 02 match at baseline
    var_by_param = dict(var_fixed)
    var_by_param['rho_i'] = single_param_variance(sub, 'rho_i', seed_i, sigma_scale=scale)
    var_sim = simultaneous_variance(sub, 5000 + k, sigma_scale=scale)
    for r in shares_by_region(sub, var_by_param, var_sim):
        r.update(rho_sigma=s_fyi, sigma_scale=scale, rho_sigma_myi=RHO_I_MYI_SIGMA * scale,
                 dist_shape='mixture', rho_central='mixture')
        rows.append(r)
    a = [r for r in rows if r['rho_sigma'] == s_fyi and r['region'] == 'all'][0]
    print(f"  sigma_FYI={s_fyi:5.1f}: rho_i={a['frac_rho_i']:5.1f}%  delta={a['frac_delta']:5.1f}%  "
          f"inter={a['frac_interaction']:4.1f}%  ({time.time() - t:.0f}s)")

df_out = pd.DataFrame(rows)
csv_path = f'{M3_OUTPUT}/R3_density_sweep_d070_v7.csv'
df_out.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path} ({len(df_out)} rows)")

print("\n" + "=" * 72)
print("SUMMARY (un-renormalised shares; inter = interaction residual; d/rho_i = delta share / rho_i share)")
print("=" * 72)
print_region_table(df_out, 'rho_sigma', 'sig_FYI')
check_against_step02(df_out, 'rho_sigma', RHO_I_FYI_SIGMA)

print("\n  Ordering of the two leading terms by stratum:")
for region in ['all'] + list(REGIONS_V6):
    s = df_out[df_out.region == region].sort_values('rho_sigma')
    led = ['rho_i' if a >= b else 'delta' for a, b in zip(s.frac_rho_i, s.frac_delta)]
    print(f"    {region:10s}: " + "  ".join(f"{x:4.1f}->{l}" for x, l in zip(s.rho_sigma, led)))

print(f"\nStep 03 v7 complete in {(time.time() - t0) / 60:.1f} min")

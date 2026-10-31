#!/usr/bin/env python3
"""
R3_Step03b_v7_delta_sweep.py — penetration-fraction uncertainty sweep, mixture prior
=====================================================================================
Replaces R3_Step03b_delta_sweep.py (single FYI prior, shares renormalised).
Sweeps sigma_delta over DELTA_SIGMAS with the mixture rho_i prior of Step 02 v7
and all other inputs at their primary values. The sweep point 0.20 reproduces
Step 02 v7 exactly (same subsample and seeds).

Shares are fractions of the simultaneous-perturbation variance, un-renormalised.
The crossover reported is the sigma_delta below which rho_i exceeds delta, and
additionally the sigma_delta below which delta falls under half the rho_i share
and under the CryoSat-2 freeboard share, since under the mixture prior rho_i
already leads in every stratum at 0.20 (so the original 'crossover' may not
exist within the grid).

Input : {M3_OUTPUT}/R3_input_d070_icetype.pkl, R3_ExpB_d070_v7.json (for check)
Output: {M3_OUTPUT}/R3_delta_sweep_v7.csv
Run   : source ~/cryo2ice_env/bin/activate && python R3_Step03b_v7_delta_sweep.py
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
DELTA_SIGMAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

print("=" * 72)
print("R3 Step 03b v7: sigma_delta sweep with per-matchup MIXTURE rho_i prior")
print(f"  sigma_delta grid: {DELTA_SIGMAS}; delta mean {DELTA_BASELINE}, clipped {DELTA_CLIP}")
print(f"  rho_i: mixture (FYI {RHO_I_FYI_MU}+/-{RHO_I_FYI_SIGMA}, MYI {RHO_I_MYI_MU}+/-{RHO_I_MYI_SIGMA}, cap {RHO_I_UPPER_CAP})")
print(f"  N_MC = {N_MC:,}; strata: {REGIONS_V6}")
print("=" * 72)

sub = load_subsample()

print("\n  Fixed single-parameter perturbations (once):")
var_fixed = {}
for p in ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i']:
    t = time.time()
    var_fixed[p] = single_param_variance(sub, p, BASE_SEED[p])
    print(f"    {p:7s}: {time.time() - t:5.0f}s")

rows = []
for k, s_d in enumerate(DELTA_SIGMAS):
    t = time.time()
    baseline = np.isclose(s_d, DELTA_SIGMA)
    seed_d = BASE_SEED['delta'] if baseline else 3000 + k
    var_by_param = dict(var_fixed)
    var_by_param['delta'] = single_param_variance(sub, 'delta', seed_d, delta_sigma=s_d)
    var_sim = simultaneous_variance(sub, 6000 + k, delta_sigma=s_d)
    for r in shares_by_region(sub, var_by_param, var_sim):
        r.update(delta_sigma=s_d)
        rows.append(r)
    a = [r for r in rows if r['delta_sigma'] == s_d and r['region'] == 'all'][0]
    print(f"  sigma_delta={s_d:4.2f}: rho_i={a['frac_rho_i']:5.1f}%  delta={a['frac_delta']:5.1f}%  "
          f"fb_CS2={a['frac_fb_cs2']:4.1f}%  inter={a['frac_interaction']:4.1f}%  ({time.time() - t:.0f}s)")

df_out = pd.DataFrame(rows)
csv_path = f'{M3_OUTPUT}/R3_delta_sweep_v7.csv'
df_out.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path} ({len(df_out)} rows)")

print("\n" + "=" * 72)
print("SUMMARY (un-renormalised shares)")
print("=" * 72)
print_region_table(df_out, 'delta_sigma', 'sig_d')
check_against_step02(df_out, 'delta_sigma', DELTA_SIGMA)


def first_sigma_where(s, cond):
    """Largest sigma_delta in the grid at which cond holds, scanning upward (None if never)."""
    hit = [x for x, ok in zip(s.delta_sigma, cond) if ok]
    return max(hit) if hit else None


print("\n  Benchmarks by stratum (largest sigma_delta in grid satisfying each condition):")
print(f"  {'stratum':10s}  {'delta<rho_i':>12s}  {'delta<rho_i/2':>14s}  {'delta<fb_CS2':>13s}  {'delta@0.20':>10s}  {'rho_i@0.20':>10s}")
for region in ['all'] + list(REGIONS_V6):
    s = df_out[df_out.region == region].sort_values('delta_sigma')
    b1 = first_sigma_where(s, s.frac_delta < s.frac_rho_i)
    b2 = first_sigma_where(s, s.frac_delta < 0.5 * s.frac_rho_i)
    b3 = first_sigma_where(s, s.frac_delta < s.frac_fb_cs2)
    p = s[np.isclose(s.delta_sigma, 0.20)].iloc[0]
    f = lambda v: f"{v:.2f}" if v is not None else "never"
    print(f"  {region:10s}  {f(b1):>12s}  {f(b2):>14s}  {f(b3):>13s}  {p.frac_delta:9.1f}%  {p.frac_rho_i:9.1f}%")

print(f"\nStep 03b v7 complete in {(time.time() - t0) / 60:.1f} min")

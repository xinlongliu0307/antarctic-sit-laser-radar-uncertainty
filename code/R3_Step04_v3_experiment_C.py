#!/usr/bin/env python3
"""
R3_Step04_v3_experiment_C.py — Snow-source comparison + penetration sensitivity
================================================================================
Changes from v2 (Alex 08 Apr 2026):
  1. rho_i configured μ = 885 → effective mean ≈ 880 (Alexandrov 2010, Hutchings 2015)
  2. "Constant" label (not "Uniform") for fixed 0.15 m snow
  3. Positive freeboard filter assumed already applied in Step 01
  4. rho_w fixed at 1024 (not perturbed)

Run: source ~/cryo2ice_env/bin/activate && python R3_Step04_v3_experiment_C.py
"""

import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

t0 = time.time()

# *** Configured μ = 885 → effective truncated mean ≈ 880 ***
RHO_I_C = RHO_I_CONFIGURED_MU   # = 885.0

print("=" * 70)
print("R3 Step 04 v3: Experiment C (snow-source) + Penetration Sensitivity")
print(f"  rho_i: configured μ={RHO_I_C} (effective mean ≈ 880)")
print(f"  rho_w = {RHO_W_CENTRAL} (FIXED)")
print(f"  AMSR2 unit: METRES (no /100 conversion)")
print("=" * 70)

df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_verified.pkl')
print(f"Loaded {len(df):,} matchups (positive-fb filtered)")

# ── EXPERIMENT C: Snow-source comparison ──────────────────────────────

hs_c2i = df['hs_d070'].values
hs_amsr2 = df['AMSR2_snow_depth'].values
hs_const = np.full(len(df), CONSTANT_SNOW)

rho_s_c = RHO_S_CENTRAL
rho_w_c = RHO_W_CENTRAL
delta_c = DELTA_BASELINE

# *** Alex 08 Apr: "Constant" not "Uniform" ***
sources = {
    'C2I': hs_c2i,
    'AMSR2': hs_amsr2,
    'Constant_0.15m': hs_const,
}

is2_fb = df['is2_fb_wm'].values
sectors = df['sector'].values

print("\n--- Experiment C: Ice Thickness from Different Snow Sources ---")
print(f"    rho_i = {RHO_I_C} (effective ≈ 880), rho_s = {rho_s_c}, rho_w = {rho_w_c} (fixed), delta = {delta_c}")
print(f"    Hydrostatic: h_i = (rho_w * is2_fb - (rho_w - rho_s) * h_s) / (rho_w - rho_i)")

results_C = {}
for name, hs in sources.items():
    if name == 'AMSR2':
        valid = np.isfinite(hs) & (hs > 0) & (hs <= 1.5)
    elif name == 'C2I':
        valid = np.isfinite(hs)
    else:
        valid = np.ones(len(hs), dtype=bool)

    hi = compute_ice_thickness(is2_fb[valid], hs[valid], rho_s_c, RHO_I_C, rho_w_c, delta_c)

    for sector in ['Weddell', 'Ross']:
        mask = (sectors[valid] == sector)
        key = f"{name}_{sector}"
        hs_sec = hs[valid][mask]
        hi_sec = hi[mask]
        results_C[key] = {
            'mean_hs': float(np.nanmean(hs_sec)),
            'median_hs': float(np.nanmedian(hs_sec)),
            'mean_hi': float(np.nanmean(hi_sec)),
            'median_hi': float(np.nanmedian(hi_sec)),
            'std_hi': float(np.nanstd(hi_sec)),
            'n': int(mask.sum()),
        }
        print(f"  {name:15s} {sector:8s}: h_s={results_C[key]['mean_hs']:.3f} m "
              f"(median {results_C[key]['median_hs']:.3f}) -> "
              f"h_i={results_C[key]['mean_hi']:.3f} +/- {results_C[key]['std_hi']:.3f} m "
              f"(n={mask.sum():,})")

print("\n--- Impact of Snow Source on Ice Thickness ---")
for sector in ['Weddell', 'Ross']:
    c2i_hi = results_C[f'C2I_{sector}']['mean_hi']
    amsr2_hi = results_C[f'AMSR2_{sector}']['mean_hi']
    const_hi = results_C[f'Constant_0.15m_{sector}']['mean_hi']
    print(f"  {sector}: C2I={c2i_hi:.3f} m, AMSR2={amsr2_hi:.3f} m, "
          f"Constant={const_hi:.3f} m, range={max(c2i_hi,amsr2_hi,const_hi)-min(c2i_hi,amsr2_hi,const_hi):.3f} m")

with open(f'{M3_OUTPUT}/R3_ExpC_d070_v3.json', 'w') as f:
    json.dump(results_C, f, indent=2, default=str)
print(f"\n  Saved: {M3_OUTPUT}/R3_ExpC_d070_v3.json")


# ── PENETRATION SENSITIVITY ──────────────────────────────────────────
print("\n" + "=" * 70)
print("Penetration Sensitivity: h_i vs delta")
print(f"  rho_i = {RHO_I_C} (effective ≈ 880), rho_w = {rho_w_c} (fixed)")
print("=" * 70)

delta_range = np.arange(0.40, 1.01, 0.05)
pen_results = []
delta_f_arr = df['delta_f'].values

for delta_val in delta_range:
    hs_delta = delta_f_arr / (delta_val * ETA_S)

    for sector in ['Weddell', 'Ross']:
        mask = sectors == sector
        hi_delta = compute_ice_thickness(is2_fb[mask], hs_delta[mask],
                                          rho_s_c, RHO_I_C, rho_w_c, delta_val)
        hi_lo = compute_ice_thickness(is2_fb[mask], hs_delta[mask],
                                       rho_s_c - RHO_S_SIGMA, RHO_I_C, rho_w_c, delta_val)
        hi_hi = compute_ice_thickness(is2_fb[mask], hs_delta[mask],
                                       rho_s_c + RHO_S_SIGMA, RHO_I_C, rho_w_c, delta_val)
        pen_results.append({
            'delta': delta_val, 'sector': sector,
            'mean_hi': np.mean(hi_delta), 'median_hi': np.median(hi_delta),
            'mean_hi_rho_lo': np.mean(hi_lo), 'mean_hi_rho_hi': np.mean(hi_hi),
            'mean_hs': np.mean(hs_delta[mask]),
        })

    print(f"  delta={delta_val:.2f}: Wed h_i={pen_results[-2]['mean_hi']:.3f} m, "
          f"Ross h_i={pen_results[-1]['mean_hi']:.3f} m")

df_pen = pd.DataFrame(pen_results)
csv_path = f'{M3_OUTPUT}/R3_penetration_sensitivity_d070_v3.csv'
df_pen.to_csv(csv_path, index=False)
print(f"\n  Saved: {csv_path}")

print(f"\n  M2 empirical optima: delta*_median ~ 0.55, delta*_mean ~ 0.58")
for delta_star in [0.55, 0.58, 0.70]:
    row = df_pen[(df_pen['delta'].between(delta_star-0.01, delta_star+0.01))]
    for _, r in row.iterrows():
        print(f"    delta={delta_star:.2f} {r['sector']}: h_i={r['mean_hi']:.3f} m, h_s={r['mean_hs']:.3f} m")

print(f"\nStep 04 v3 complete in {time.time()-t0:.1f} s")

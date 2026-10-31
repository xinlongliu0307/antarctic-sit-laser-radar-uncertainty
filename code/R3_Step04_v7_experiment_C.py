#!/usr/bin/env python3
"""
R3_Step04_v7_experiment_C.py — Snow-source comparison + penetration sensitivity (v7)
====================================================================================
Derived from R3_Step04_v3_experiment_C.py. Changes:
  * bulk sea-ice density per matchup = expected value of the mixture prior,
    f*mu_MYI + (1-f)*mu_FYI with f the collocated Melsheimer et al. (2023) MYI
    concentration (Step 13), instead of one FYI value for every matchup
  * results reported for REGIONS_V6 (four sectors + observed MYI / FYI) as well
    as the two sectors of v3
  * input is the ice-type pickle from Step 13

Everything else (snow sources, delta grid, rho_s band, filters) is unchanged.
Deterministic; runs in seconds on a login node.

Input : {M3_OUTPUT}/R3_input_d070_icetype.pkl
Output: {M3_OUTPUT}/R3_ExpC_d070_v7.json
        {M3_OUTPUT}/R3_penetration_sensitivity_d070_v7.csv
Run   : source ~/cryo2ice_env/bin/activate && python R3_Step04_v7_experiment_C.py
"""
import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

t0 = time.time()
print("=" * 72)
print("R3 Step 04 v7: Experiment C (snow-source) + penetration sensitivity")
print("  rho_i per matchup = mixture expected value (Melsheimer MYI weight)")
print(f"  rho_s = {RHO_S_CENTRAL}, rho_w = {RHO_W_CENTRAL} (fixed), delta = {DELTA_BASELINE}")
print("=" * 72)

df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_icetype.pkl')
if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df['lon'].values)
n = len(df)
sectors, subsectors, icetype = df['sector'].values, df['subsector'].values, df['icetype'].values
is2_fb = df['is2_fb_wm'].values
rho_i_c = mixture_rho_i_central(df['myi_frac'].values)        # per-matchup central density
masks = {r: get_region_mask_v6(sectors, subsectors, icetype, r) for r in REGIONS_V6}
print(f"Loaded {n:,} matchups; mean central rho_i = {rho_i_c.mean():.1f} kg/m^3")

# ── Experiment C: snow-source comparison ─────────────────────────────────
sources = {'C2I': df['hs_d070'].values,
           'AMSR2': df['AMSR2_snow_depth'].values,
           'Constant_0.15m': np.full(n, CONSTANT_SNOW)}
results_C = {}
print("\n--- Experiment C: sea-ice thickness from different snow sources ---")
for name, hs in sources.items():
    if name == 'AMSR2':
        valid = np.isfinite(hs) & (hs > 0) & (hs <= 1.5)
    else:
        valid = np.isfinite(hs)
    hi = np.full(n, np.nan)
    hi[valid] = compute_ice_thickness(is2_fb[valid], hs[valid], RHO_S_CENTRAL,
                                      rho_i_c[valid], RHO_W_CENTRAL)
    for r in REGIONS_V6:
        m = masks[r] & valid
        if m.sum() == 0:
            continue
        results_C[f'{name}_{r}'] = dict(
            mean_hs=float(np.mean(hs[m])), median_hs=float(np.median(hs[m])),
            mean_hi=float(np.mean(hi[m])), median_hi=float(np.median(hi[m])),
            std_hi=float(np.std(hi[m])), mean_rho_i=float(np.mean(rho_i_c[m])), n=int(m.sum()))
        print(f"  {name:15s} {r:10s}: h_s={np.mean(hs[m]):.3f} m (med {np.median(hs[m]):.3f}) -> "
              f"h_i={np.mean(hi[m]):.3f} +/- {np.std(hi[m]):.3f} m (n={m.sum():,})")

print("\n--- Snow-source impact (mean h_i) ---")
for r in REGIONS_V6:
    v = [results_C.get(f'{s}_{r}', {}).get('mean_hi', np.nan) for s in sources]
    print(f"  {r:10s}: C2I={v[0]:.3f}  AMSR2={v[1]:.3f}  Constant={v[2]:.3f}  range={np.nanmax(v) - np.nanmin(v):.3f} m")

with open(f'{M3_OUTPUT}/R3_ExpC_d070_v7.json', 'w') as f:
    json.dump(results_C, f, indent=2)
print(f"  Saved: {M3_OUTPUT}/R3_ExpC_d070_v7.json")

# ── Penetration sensitivity: mean h_i versus delta ───────────────────────
print("\n" + "=" * 72)
print("Penetration sensitivity: h_i vs delta (rho_i = mixture central per matchup)")
print("=" * 72)
delta_range = np.arange(0.40, 1.01, 0.05)
delta_f = df['delta_f'].values
rows = []
for dv in delta_range:
    hs_d = delta_f / (dv * ETA_S)
    for r in REGIONS_V6:
        m = masks[r]
        hi = compute_ice_thickness(is2_fb[m], hs_d[m], RHO_S_CENTRAL, rho_i_c[m], RHO_W_CENTRAL)
        hi_lo = compute_ice_thickness(is2_fb[m], hs_d[m], RHO_S_CENTRAL - RHO_S_SIGMA, rho_i_c[m], RHO_W_CENTRAL)
        hi_hi = compute_ice_thickness(is2_fb[m], hs_d[m], RHO_S_CENTRAL + RHO_S_SIGMA, rho_i_c[m], RHO_W_CENTRAL)
        rows.append(dict(delta=round(float(dv), 2), sector=r, n=int(m.sum()),
                         mean_hi=float(np.mean(hi)), median_hi=float(np.median(hi)),
                         mean_hi_rho_lo=float(np.mean(hi_lo)), mean_hi_rho_hi=float(np.mean(hi_hi)),
                         mean_hs=float(np.mean(hs_d[m]))))
    w = [x for x in rows if x['delta'] == round(float(dv), 2)]
    print(f"  delta={dv:.2f}: " + "  ".join(f"{x['sector']}={x['mean_hi']:.3f}" for x in w))

df_pen = pd.DataFrame(rows)
csv_path = f'{M3_OUTPUT}/R3_penetration_sensitivity_d070_v7.csv'
df_pen.to_csv(csv_path, index=False)
print(f"\n  Saved: {csv_path}")

print("\n  Sensitivity of sector mean h_i per 0.1 change in delta (0.60 -> 0.70 -> 0.80):")
for r in REGIONS_V6:
    s = df_pen[df_pen.sector == r].set_index('delta')['mean_hi']
    print(f"    {r:10s}: {s[0.6] - s[0.7]:+.3f} m (0.60), {s[0.8] - s[0.7]:+.3f} m (0.80)")
print("\n  At the M2 empirical optima:")
for dstar in [0.55, 0.60, 0.70]:
    w = df_pen[np.isclose(df_pen.delta, dstar)]
    print(f"    delta={dstar:.2f}: " + "  ".join(f"{x.sector}={x.mean_hi:.3f}" for x in w.itertuples()))

print(f"\nStep 04 v7 complete in {time.time() - t0:.1f} s")

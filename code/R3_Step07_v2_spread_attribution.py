#!/usr/bin/env python3
"""
R3_Step07_v2_spread_attribution.py — Corrected Spread Attribution
==================================================================
FIX from v1: S_total now uses the same IS2-freeboard-based h_i as S_snow,
ensuring consistent denominators. Also reports attribution on both h_i and
h_fi (freeboard-level) for direct comparison to Chapter 3.

Method:
  For each C2I segment, compute h_i using IS2 total freeboard under:
    - Product k's snow (for S_snow): h_i^k = f(is2_fb, hs_clim_k)
    - C2I snow + product k's hfr (for S_retrack): h_i^k = f(hfr_k + hsc_c2i, hs_c2i)
    - Product k's snow + product k's hfr (for S_total): h_i^k = f(hfr_k + hsc_k, hs_k)

  f_snow   = S_snow / S_total
  f_retrack = S_retrack / S_total

Run: source ~/cryo2ice_env/bin/activate && python R3_Step07_v2_spread_attribution.py
"""

import os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v3 import *

t0 = time.time()
print("=" * 70)
print("R3 Step 07 v2: Spread Attribution (corrected S_total)")
print("=" * 70)

df = pd.read_pickle(f'{M3_OUTPUT}/R3_bridge_matched.pkl')
sectors = df['sector'].values
subsectors = df['subsector'].values
n_total = len(df)
print(f"Loaded {n_total:,} matchups")

# Identify products
prod_labels = []
for col in df.columns:
    if col.startswith('hs_clim_') and not col.endswith('_'):
        label = col.replace('hs_clim_', '')
        if label in ['CCI', 'LEGOS_I', 'LEGOS_II', 'CSAO', 'CryoTEMPO']:
            prod_labels.append(label)
prod_labels = sorted(set(prod_labels))
n_prod = len(prod_labels)
print(f"Products: {prod_labels}")

is2_fb = df['is2_fb_wm'].values
hs_c2i = df['hs_d070'].values
rho_i_base = 900.0

# ═══════════════════════════════════════════════════════════════════
# ATTRIBUTION ON ICE FREEBOARD (h_fi) — direct comparison to Ch. 3
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ATTRIBUTION ON SEA-ICE FREEBOARD (h_fi)")
print("  (direct comparison to Chapter 3: f_snow_hsc = 78-100%)")
print("=" * 70)

# Each product's h_fi from its own snow and freeboard
hfi_product = np.full((n_total, n_prod), np.nan)
# h_fi using same hfr but different (product) snow corrections
hfi_snow_varied = np.full((n_total, n_prod), np.nan)
# h_fi using same C2I snow but different hfr
hfi_retrack_varied = np.full((n_total, n_prod), np.nan)

# C2I snow -> speed correction
eta_s_val = (1 + 0.51 * RHO_S_CENTRAL / 1000.0)**1.5
hsc_c2i = (eta_s_val - 1) * hs_c2i

for j, prod in enumerate(prod_labels):
    hfr_k = df[f'hfr_clim_{prod}'].values
    hfi_k = df[f'hfi_clim_{prod}'].values
    hs_k = df[f'hs_clim_{prod}'].values
    hsc_k = df[f'hsc_clim_{prod}'].values

    # Product's own h_fi (from its own hfr + its own hsc)
    hfi_product[:, j] = hfi_k

    # S_snow: use CCI's hfr (reference) + each product's snow correction
    # This isolates the snow effect while holding hfr constant
    # Use the mean hfr across products as a common reference
    hsc_k_recomputed = (eta_s_val - 1) * hs_k  # recompute hsc using product's snow
    # We need a common hfr — use the C2I segment's CS2 radar freeboard as reference
    cs2_rfb = df['cs2_rfb'].values
    hfi_snow_varied[:, j] = cs2_rfb + hsc_k_recomputed

    # S_retrack: use each product's hfr + C2I snow correction
    hfi_retrack_varied[:, j] = hfr_k + hsc_c2i

# Compute spreads on h_fi
S_hfi_total = np.nanmax(hfi_product, axis=1) - np.nanmin(hfi_product, axis=1)
S_hfi_snow = np.nanmax(hfi_snow_varied, axis=1) - np.nanmin(hfi_snow_varied, axis=1)
S_hfi_retrack = np.nanmax(hfi_retrack_varied, axis=1) - np.nanmin(hfi_retrack_varied, axis=1)

# Fractions
valid_hfi = S_hfi_total > 0.001
f_snow_hfi = np.where(valid_hfi, S_hfi_snow / S_hfi_total * 100, np.nan)
f_retrack_hfi = np.where(valid_hfi, S_hfi_retrack / S_hfi_total * 100, np.nan)

print(f"\nSegments with valid h_fi spread: {valid_hfi.sum():,}")

for region in REGIONS:
    mask = get_region_mask(sectors, subsectors, region) & valid_hfi
    if mask.sum() < 10:
        continue
    print(f"\n  {REGION_LABELS[region]} (n={mask.sum():,}):")
    print(f"    S_hfi_total:   mean={np.nanmean(S_hfi_total[mask]):.4f} m, "
          f"median={np.nanmedian(S_hfi_total[mask]):.4f} m")
    print(f"    S_hfi_snow:    mean={np.nanmean(S_hfi_snow[mask]):.4f} m")
    print(f"    S_hfi_retrack: mean={np.nanmean(S_hfi_retrack[mask]):.4f} m")
    print(f"    f_snow (h_fi):    mean={np.nanmean(f_snow_hfi[mask]):.1f}%, "
          f"median={np.nanmedian(f_snow_hfi[mask]):.1f}%")
    print(f"    f_retrack (h_fi): mean={np.nanmean(f_retrack_hfi[mask]):.1f}%, "
          f"median={np.nanmedian(f_retrack_hfi[mask]):.1f}%")


# ═══════════════════════════════════════════════════════════════════
# ATTRIBUTION ON ICE THICKNESS (h_i) — the Chapter 5 result
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ATTRIBUTION ON SEA-ICE THICKNESS (h_i)")
print("  (using IS2 total freeboard, K&K 2020 hydrostatic equation)")
print("=" * 70)

# S_snow on h_i: use IS2 fb + each product's climatological snow
hi_snow_varied = np.full((n_total, n_prod), np.nan)
for j, prod in enumerate(prod_labels):
    hs_k = df[f'hs_clim_{prod}'].values
    hi_snow_varied[:, j] = compute_ice_thickness(is2_fb, hs_k,
                                                  RHO_S_CENTRAL, rho_i_base, RHO_W_CENTRAL)

# S_retrack on h_i: use each product's hfr + C2I hsc, then hydrostatic
hi_retrack_varied = np.full((n_total, n_prod), np.nan)
for j, prod in enumerate(prod_labels):
    hfr_k = df[f'hfr_clim_{prod}'].values
    hfi_k_c2i = hfr_k + hsc_c2i  # product's hfr + C2I snow correction
    # h_i from ice-freeboard form: (rho_w * hfi + (rho_s - rho_w) * hs) / (rho_w - rho_i)
    hi_retrack_varied[:, j] = (RHO_W_CENTRAL * hfi_k_c2i +
                                (RHO_S_CENTRAL - RHO_W_CENTRAL) * hs_c2i) / \
                               (RHO_W_CENTRAL - rho_i_base)

# S_total on h_i: use each product's hfr + each product's own hsc + each product's hs
hi_total_varied = np.full((n_total, n_prod), np.nan)
for j, prod in enumerate(prod_labels):
    hfi_k = df[f'hfi_clim_{prod}'].values
    hs_k = df[f'hs_clim_{prod}'].values
    hi_total_varied[:, j] = (RHO_W_CENTRAL * hfi_k +
                              (RHO_S_CENTRAL - RHO_W_CENTRAL) * hs_k) / \
                             (RHO_W_CENTRAL - rho_i_base)

S_hi_total = np.nanmax(hi_total_varied, axis=1) - np.nanmin(hi_total_varied, axis=1)
S_hi_snow = np.nanmax(hi_snow_varied, axis=1) - np.nanmin(hi_snow_varied, axis=1)
S_hi_retrack = np.nanmax(hi_retrack_varied, axis=1) - np.nanmin(hi_retrack_varied, axis=1)

valid_hi = S_hi_total > 0.001
f_snow_hi = np.where(valid_hi, S_hi_snow / S_hi_total * 100, np.nan)
f_retrack_hi = np.where(valid_hi, S_hi_retrack / S_hi_total * 100, np.nan)
f_residual_hi = np.where(valid_hi, 100 - f_snow_hi - f_retrack_hi, np.nan)

# Store in DataFrame
df['S_hi_total'] = S_hi_total
df['S_hi_snow'] = S_hi_snow
df['S_hi_retrack'] = S_hi_retrack
df['f_snow_hi'] = f_snow_hi
df['f_retrack_hi'] = f_retrack_hi
df['f_residual_hi'] = f_residual_hi

print(f"\nSegments with valid h_i spread: {valid_hi.sum():,}")

results_attrib = []
for region in REGIONS:
    mask = get_region_mask(sectors, subsectors, region) & valid_hi
    if mask.sum() < 10:
        continue

    row = {
        'region': region,
        'n': int(mask.sum()),
        'S_hi_total_mean': float(np.nanmean(S_hi_total[mask])),
        'S_hi_total_median': float(np.nanmedian(S_hi_total[mask])),
        'S_hi_snow_mean': float(np.nanmean(S_hi_snow[mask])),
        'S_hi_retrack_mean': float(np.nanmean(S_hi_retrack[mask])),
        'f_snow_hi_mean': float(np.nanmean(f_snow_hi[mask])),
        'f_snow_hi_median': float(np.nanmedian(f_snow_hi[mask])),
        'f_retrack_hi_mean': float(np.nanmean(f_retrack_hi[mask])),
        'f_retrack_hi_median': float(np.nanmedian(f_retrack_hi[mask])),
        'f_residual_hi_mean': float(np.nanmean(f_residual_hi[mask])),
        # Also store h_fi-level attribution
        'f_snow_hfi_mean': float(np.nanmean(f_snow_hfi[mask & valid_hfi])) if (mask & valid_hfi).sum() > 0 else np.nan,
        'f_retrack_hfi_mean': float(np.nanmean(f_retrack_hfi[mask & valid_hfi])) if (mask & valid_hfi).sum() > 0 else np.nan,
    }
    results_attrib.append(row)

    print(f"\n  {REGION_LABELS[region]} (n={mask.sum():,}):")
    print(f"    S_hi_total:   mean={row['S_hi_total_mean']:.3f} m, "
          f"median={row['S_hi_total_median']:.3f} m")
    print(f"    S_hi_snow:    mean={row['S_hi_snow_mean']:.3f} m")
    print(f"    S_hi_retrack: mean={row['S_hi_retrack_mean']:.3f} m")
    print(f"    f_snow (h_i):    mean={row['f_snow_hi_mean']:.1f}%, "
          f"median={row['f_snow_hi_median']:.1f}%")
    print(f"    f_retrack (h_i): mean={row['f_retrack_hi_mean']:.1f}%, "
          f"median={row['f_retrack_hi_median']:.1f}%")
    print(f"    f_residual:      mean={row['f_residual_hi_mean']:.1f}%")


# ── Comparison to Chapter 3 ──
print("\n" + "=" * 70)
print("COMPARISON TO CHAPTER 3")
print("=" * 70)
print("  Chapter 3 (gridded h_fi, 2013-2018):")
print("    f_snow(h_sc):  78-100%")
print("    f_snow(h_fi):  7-22%")
print("    f_retrack(h_fi): 78-94%")
print("  Chapter 5 (along-track, C2I-informed):")
for r in results_attrib:
    print(f"    {r['region']:12s}: f_snow(h_i)={r['f_snow_hi_mean']:.1f}%, "
          f"f_retrack(h_i)={r['f_retrack_hi_mean']:.1f}%, "
          f"f_snow(h_fi)={r.get('f_snow_hfi_mean', 0):.1f}%, "
          f"f_retrack(h_fi)={r.get('f_retrack_hfi_mean', 0):.1f}%")


# ── Per-product thickness summary for Table 5.2 ──
print("\n" + "=" * 70)
print("TABLE 5.2 VALUES: Per-Product Summary by Sector")
print("=" * 70)
for region in REGIONS:
    mask = get_region_mask(sectors, subsectors, region)
    print(f"\n  {REGION_LABELS[region]} (n={mask.sum():,}):")
    print(f"    {'Product':12s}  {'hs_C2I':>8s}  {'hs_clim':>8s}  {'dhs':>8s}  "
          f"{'hi_C2I':>8s}  {'hi_clim':>8s}  {'dhi':>8s}")
    print("    " + "-" * 68)
    sub_mask = mask
    for prod in prod_labels:
        hs_col = f'hs_clim_{prod}'
        hi_col = f'hi_clim_{prod}'
        dhi_col = f'dhi_{prod}'
        dhs_col = f'dhs_{prod}'
        valid = sub_mask & np.isfinite(df[hs_col].values)
        if valid.sum() == 0:
            continue
        s = df[valid]
        print(f"    {prod:12s}  {s['hs_d070'].median():8.3f}  {s[hs_col].median():8.3f}  "
              f"{s[dhs_col].median():+8.3f}  {s['hi_c2i'].median():8.3f}  "
              f"{s[hi_col].median():8.3f}  {s[dhi_col].median():+8.3f}")


# ── Save ──
with open(f'{M3_OUTPUT}/R3_spread_attribution_v2.json', 'w') as f:
    json.dump(results_attrib, f, indent=2)
print(f"\nSaved: {M3_OUTPUT}/R3_spread_attribution_v2.json")

df.to_pickle(f'{M3_OUTPUT}/R3_bridge_attributed_v2.pkl')
print(f"Saved: {M3_OUTPUT}/R3_bridge_attributed_v2.pkl")

print(f"\nStep 07 v2 complete in {time.time()-t0:.1f} s")

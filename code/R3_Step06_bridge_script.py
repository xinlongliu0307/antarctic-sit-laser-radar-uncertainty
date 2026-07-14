#!/usr/bin/env python3
"""
R3_Step06_bridge_script.py — Bridge C2I segments to Chapter 3 gridded products
===============================================================================
Maps each of the 82,341 C2I matchup points to the nearest grid cell in the
216x216 EASE-2 50km grid from the Chapter 3 regridded products, extracts the
multi-year winter-mean climatological snow thickness (hs) and radar freeboard
(hfr) for each product, and stores matched pairs for the controlled
substitution experiment (Chapter 5 Stages A-C).

Approach: Climatological-mean (Option 1)
  - Computes multi-year mean hs for each product across 2013-2018 winter months
  - Applies these as climatological references at each C2I location
  - Justified because inter-product differences are systematic and persistent
    (Chapter 3: snow assumptions explain 78-100% of hsc spread)

Products (CS2 era only, 2013-2018):
  - CCI_CS2       : ESA CCI v4.0
  - LEGOS_I_CS2   : LEGOS with AMSR2 snow
  - LEGOS_II_CS2  : LEGOS with SARAL Ka/CS2 Ku snow
  - CSAO_CS2      : CryoSat-2 Antarctic Ocean
  - CryoTEMPO_CS2 : CryoSat ThEMatic Products

NPZ variables:
  - hfr : radar freeboard (m)
  - hfi : sea-ice freeboard (m)
  - hs  : climatological snow thickness used by each product (m)
  - hsc : radar speed correction = hfi - hfr (m)
  - lat, lon : grid coordinates (216x216)

Run: source ~/cryo2ice_env/bin/activate && python R3_Step06_bridge_script.py
"""

import os, sys, time, glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v3 import *

t0 = time.time()
print("=" * 70)
print("R3 Step 06: Bridge C2I Segments to Chapter 3 Gridded Products")
print("  Approach: Climatological-mean (2013-2018 winter)")
print("=" * 70)

# ── Paths ──
REGRID_DIR = '/g/data/gv90/xl1657/phd/M1_workspace/processed/regridded'
MASK_DIR = '/g/data/gv90/xl1657/phd/M1_workspace/processed/masks'

# ── Load C2I matchup data ──
df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_verified.pkl')
if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df['lon'].values)
n_total = len(df)
print(f"\nLoaded {n_total:,} C2I matchups")

c2i_lats = df['lat'].values
c2i_lons = df['lon'].values

# ── Load grid coordinates from sector_map.npz ──
sm = np.load(f'{MASK_DIR}/sector_map.npz', allow_pickle=True)
grid_lat = sm['lat']   # (216, 216)
grid_lon = sm['lon']   # (216, 216)
ny, nx = grid_lat.shape
print(f"Grid: {ny} x {nx}")

# ── Build KD-tree on the grid for nearest-neighbour lookup ──
# Convert lat/lon to Cartesian for proper spherical NN
def latlon_to_xyz(lat, lon):
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

grid_xyz = latlon_to_xyz(grid_lat.ravel(), grid_lon.ravel())
tree = cKDTree(grid_xyz)

# Find nearest grid cell for each C2I point
c2i_xyz = latlon_to_xyz(c2i_lats, c2i_lons)
distances, indices = tree.query(c2i_xyz)

# Convert flat indices to (row, col) in the 216x216 grid
grid_rows = indices // nx
grid_cols = indices % nx

# Distance check: Earth radius ~ 6371 km, max acceptable distance ~ 50 km
# Cartesian distance for 50 km on unit sphere: 50/6371 ~ 0.00785
max_dist = 50.0 / 6371.0  # ~50 km on unit sphere
valid_nn = distances < max_dist
print(f"  Valid NN matches (< 50 km): {valid_nn.sum():,} / {n_total:,} "
      f"({100*valid_nn.sum()/n_total:.1f}%)")
print(f"  Max NN distance: {distances.max() * 6371:.1f} km")
print(f"  Median NN distance: {np.median(distances) * 6371:.1f} km")

# ── Define products to extract ──
PRODUCTS = {
    'CCI':        'CCI_CS2',
    'LEGOS_I':    'LEGOS_I_CS2',
    'LEGOS_II':   'LEGOS_II_CS2',
    'CSAO':       'CSAO_CS2',
    'CryoTEMPO':  'CryoTEMPO_CS2',
}

# Winter months in CS2 era
YEARS = range(2013, 2019)
MONTHS = range(5, 11)  # May-October

# ── Compute multi-year winter-mean climatological fields for each product ──
print("\n--- Computing Multi-Year Winter-Mean Climatologies ---")

climatologies = {}
for prod_label, prod_prefix in PRODUCTS.items():
    hs_stack = []
    hfr_stack = []
    hfi_stack = []
    hsc_stack = []
    n_files = 0

    for year in YEARS:
        for month in MONTHS:
            fname = f'{prod_prefix}_{year}{month:02d}.npz'
            fpath = f'{REGRID_DIR}/{fname}'
            if not os.path.exists(fpath):
                continue
            d = np.load(fpath, allow_pickle=True)
            hs_grid = d['hs'].astype(float)
            hfr_grid = d['hfr'].astype(float)
            hfi_grid = d['hfi'].astype(float)
            hsc_grid = d['hsc'].astype(float)

            # Replace fill/invalid with NaN
            for arr in [hs_grid, hfr_grid, hfi_grid, hsc_grid]:
                arr[arr < -9] = np.nan
                arr[arr > 9] = np.nan

            hs_stack.append(hs_grid)
            hfr_stack.append(hfr_grid)
            hfi_stack.append(hfi_grid)
            hsc_stack.append(hsc_grid)
            n_files += 1

    if n_files == 0:
        print(f"  {prod_label:12s}: NO FILES FOUND — skipping")
        continue

    # Compute mean across all winter months (ignoring NaN)
    hs_mean = np.nanmean(np.array(hs_stack), axis=0)
    hfr_mean = np.nanmean(np.array(hfr_stack), axis=0)
    hfi_mean = np.nanmean(np.array(hfi_stack), axis=0)
    hsc_mean = np.nanmean(np.array(hsc_stack), axis=0)

    climatologies[prod_label] = {
        'hs': hs_mean,
        'hfr': hfr_mean,
        'hfi': hfi_mean,
        'hsc': hsc_mean,
        'n_files': n_files,
    }

    valid_cells = np.isfinite(hs_mean).sum()
    print(f"  {prod_label:12s}: {n_files} files, "
          f"mean hs={np.nanmean(hs_mean):.3f} m, "
          f"mean hfr={np.nanmean(hfr_mean):.3f} m, "
          f"valid cells={valid_cells:,}/{ny*nx:,}")


# ── Extract climatological values at each C2I segment location ──
print("\n--- Extracting Values at C2I Segment Locations ---")

for prod_label in climatologies:
    clim = climatologies[prod_label]

    # Extract at grid (row, col) for each C2I point
    hs_at_c2i = clim['hs'][grid_rows, grid_cols]
    hfr_at_c2i = clim['hfr'][grid_rows, grid_cols]
    hfi_at_c2i = clim['hfi'][grid_rows, grid_cols]
    hsc_at_c2i = clim['hsc'][grid_rows, grid_cols]

    # Mark invalid NN matches
    hs_at_c2i[~valid_nn] = np.nan
    hfr_at_c2i[~valid_nn] = np.nan
    hfi_at_c2i[~valid_nn] = np.nan
    hsc_at_c2i[~valid_nn] = np.nan

    # Store in DataFrame
    df[f'hs_clim_{prod_label}'] = hs_at_c2i
    df[f'hfr_clim_{prod_label}'] = hfr_at_c2i
    df[f'hfi_clim_{prod_label}'] = hfi_at_c2i
    df[f'hsc_clim_{prod_label}'] = hsc_at_c2i

    valid = np.isfinite(hs_at_c2i)
    print(f"  {prod_label:12s}: {valid.sum():,} valid ({100*valid.sum()/n_total:.1f}%), "
          f"mean hs_clim={np.nanmean(hs_at_c2i):.3f} m")


# ── Compute C2I-informed vs climatology-informed SIT ──
print("\n--- Computing SIT Under Scenario O and Scenario C ---")

is2_fb = df['is2_fb_wm'].values
hs_c2i = df['hs_d070'].values
sectors = df['sector'].values
subsectors = df['subsector'].values

# Scenario O: SIT using C2I-observed snow
rho_i_base = 900.0
hi_c2i = compute_ice_thickness(is2_fb, hs_c2i, RHO_S_CENTRAL, rho_i_base, RHO_W_CENTRAL)
df['hi_c2i'] = hi_c2i

# Scenario C: SIT using each climatological product's snow
for prod_label in climatologies:
    hs_clim = df[f'hs_clim_{prod_label}'].values
    hi_clim = compute_ice_thickness(is2_fb, hs_clim, RHO_S_CENTRAL, rho_i_base, RHO_W_CENTRAL)
    df[f'hi_clim_{prod_label}'] = hi_clim

    # Compute thickness difference: Delta_h_i = h_i^C2I - h_i^clim
    df[f'dhi_{prod_label}'] = hi_c2i - hi_clim

    # Compute snow difference: Delta_h_s = h_s^C2I - h_s^clim
    df[f'dhs_{prod_label}'] = hs_c2i - hs_clim


# ── Report sector-stratified snow and thickness differences ──
print("\n" + "=" * 70)
print("SECTOR-STRATIFIED SNOW AND THICKNESS DIFFERENCES")
print("=" * 70)

for region in REGIONS:
    mask = get_region_mask(sectors, subsectors, region)
    print(f"\n  {REGION_LABELS[region]} (n={mask.sum():,}):")
    print(f"    {'Product':12s}  {'hs_C2I':>7s}  {'hs_clim':>7s}  {'dhs':>7s}  "
          f"{'hi_C2I':>7s}  {'hi_clim':>7s}  {'dhi':>7s}")
    print("    " + "-" * 65)

    for prod_label in climatologies:
        hs_clim_col = f'hs_clim_{prod_label}'
        hi_clim_col = f'hi_clim_{prod_label}'
        dhi_col = f'dhi_{prod_label}'
        dhs_col = f'dhs_{prod_label}'

        sub = df[mask & np.isfinite(df[hs_clim_col])]
        if len(sub) == 0:
            print(f"    {prod_label:12s}  --- no valid matchups ---")
            continue

        print(f"    {prod_label:12s}  {sub['hs_d070'].mean():7.3f}  "
              f"{sub[hs_clim_col].mean():7.3f}  {sub[dhs_col].mean():+7.3f}  "
              f"{sub['hi_c2i'].mean():7.3f}  {sub[hi_clim_col].mean():7.3f}  "
              f"{sub[dhi_col].mean():+7.3f}")


# ── Save the augmented DataFrame ──
out_path = f'{M3_OUTPUT}/R3_bridge_matched.pkl'
df.to_pickle(out_path)
print(f"\nSaved: {out_path}")

# Also save a summary CSV for quick inspection
summary_rows = []
for region in REGIONS:
    mask = get_region_mask(sectors, subsectors, region)
    for prod_label in climatologies:
        sub = df[mask & np.isfinite(df[f'hs_clim_{prod_label}'])]
        if len(sub) == 0:
            continue
        summary_rows.append({
            'region': region,
            'product': prod_label,
            'n': len(sub),
            'hs_c2i_mean': sub['hs_d070'].mean(),
            'hs_c2i_median': sub['hs_d070'].median(),
            'hs_clim_mean': sub[f'hs_clim_{prod_label}'].mean(),
            'hs_clim_median': sub[f'hs_clim_{prod_label}'].median(),
            'dhs_mean': sub[f'dhs_{prod_label}'].mean(),
            'dhs_median': sub[f'dhs_{prod_label}'].median(),
            'hi_c2i_mean': sub['hi_c2i'].mean(),
            'hi_clim_mean': sub[f'hi_clim_{prod_label}'].mean(),
            'dhi_mean': sub[f'dhi_{prod_label}'].mean(),
            'dhi_median': sub[f'dhi_{prod_label}'].median(),
        })

df_summary = pd.DataFrame(summary_rows)
csv_path = f'{M3_OUTPUT}/R3_bridge_summary.csv'
df_summary.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

print(f"\nStep 06 complete in {time.time()-t0:.1f} s")

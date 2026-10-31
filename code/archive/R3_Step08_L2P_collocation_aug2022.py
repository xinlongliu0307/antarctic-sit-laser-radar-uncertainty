#!/usr/bin/env python3
"""
R3_Step08_L2P_collocation_aug2022.py
====================================
Proof-of-concept: Collocate CCI v4.0 L2P and Cryo-TEMPO L2P along-track
data with CRYO2ICE (C2I) segments for August 2022.

Method: Nearest-neighbour from C2I CS2 location to L2P CS2 location
        (per Alex's instruction, 8 Apr 2026).

Input:
  - C2I matchup CSV: snow_thickness_with_amsr2_all.csv
  - CCI v4.0 L2P daily files (Aug 2022)
  - Cryo-TEMPO L2P orbit files (Aug 2022)

Output:
  - CSV with collocated C2I + CCI + Cryo-TEMPO values
  - Summary statistics printed to terminal

Author: Xinlong Liu
Date:   09 Apr 2026
Environment: cryo2ice_env on NCI Gadi (project gv90)
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
import time as timer
from scipy.spatial import cKDTree

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
# Paths
C2I_CSV = '/g/data/gv90/xl1657/phd/M2_workspace/output/snow_thickness/snow_thickness_with_amsr2_all.csv'
CCI_DIR = '/g/data/gv90/xl1657/phd/M1_workspace/raw_data/CCI_CS2/v4p0_L2P'
CT_DIR  = '/g/data/gv90/xl1657/phd/M1_workspace/raw_data/CryoTEMPO/TEMPO_POCA_SI/ALONGTRACK'
OUT_DIR = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation'

# Collocation parameters
MAX_DIST_KM = 25.0   # Maximum nearest-neighbour distance (km)
EARTH_RADIUS_KM = 6371.0

# Study period for this proof-of-concept
POC_YEAR = 2022
POC_MONTH = 8  # August

# Sector definitions (matching M2)
SECTOR_DEFS = {
    'Weddell':         {'lon_min': -62.0, 'lon_max': 15.0},
    'Western_Weddell': {'lon_min': -62.0, 'lon_max': -45.0},
    'Eastern_Weddell': {'lon_min': -45.0, 'lon_max': 15.0},
    'Ross':            {'lon_min': 160.0, 'lon_max': 220.0},  # 160E to 140W = 220
}

# ===========================================================================
#  HELPER FUNCTIONS
# ===========================================================================

def latlon_to_cartesian(lat_deg, lon_deg):
    """Convert lat/lon (degrees) to 3D Cartesian for cKDTree."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = EARTH_RADIUS_KM * np.cos(lat) * np.cos(lon)
    y = EARTH_RADIUS_KM * np.cos(lat) * np.sin(lon)
    z = EARTH_RADIUS_KM * np.sin(lat)
    return np.column_stack([x, y, z])


def assign_sector(lon):
    """Assign sector label based on longitude. Handles wrap-around for Ross."""
    lon = lon % 360  # normalise to [0, 360)
    if 298.0 <= lon <= 375.0 or lon <= 15.0:
        # Weddell: -62 to 15 => 298 to 360 and 0 to 15
        lon_orig = lon if lon <= 180 else lon - 360
        if lon_orig < -45.0:
            return 'Western_Weddell'
        else:
            return 'Eastern_Weddell'
    elif 160.0 <= lon <= 220.0:
        return 'Ross'
    else:
        return 'Other'


def load_c2i_august2022(csv_path, year, month):
    """Load C2I matchups and filter to the target month and Weddell/Ross sectors."""
    print(f'Loading C2I matchup data from:\n  {csv_path}')
    df = pd.read_csv(csv_path)
    print(f'  Total rows: {len(df):,}')

    # Parse time — adapt column name to your CSV structure
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()
                 or 'year' in c.lower() or 'month' in c.lower()]
    print(f'  Time-related columns: {time_cols}')

    # Diagnostic: show first few values of date-related columns
    for tc in ['date', 'year_month']:
        if tc in df.columns:
            print(f'  {tc} — first 5 values: {df[tc].head().tolist()}')
            print(f'  {tc} — dtype: {df[tc].dtype}, unique count: {df[tc].nunique()}')

    # Filter to target month using the most reliable method available
    target_ym = f'{year}-{month:02d}'       # e.g. "2022-08"
    target_ym_alt = f'{year}{month:02d}'    # e.g. "202208"

    if 'year_month' in df.columns:
        # Use year_month column directly — most reliable
        ym_vals = df['year_month'].astype(str)
        mask_time = (ym_vals == target_ym) | (ym_vals == target_ym_alt)
        print(f'  Filtering via year_month column, target: {target_ym} or {target_ym_alt}')
    elif 'date' in df.columns:
        # Try parsing date column
        try:
            df['datetime'] = pd.to_datetime(df['date'])
            mask_time = (df['datetime'].dt.year == year) & (df['datetime'].dt.month == month)
        except Exception:
            # date might be a year-month string
            date_str = df['date'].astype(str)
            mask_time = (date_str == target_ym) | (date_str == target_ym_alt)
    else:
        raise ValueError(f'Cannot identify time column. Available: {list(df.columns)}')

    df_month = df[mask_time].copy()
    print(f'  Rows for {year}-{month:02d}: {len(df_month):,}')

    # Identify lat/lon columns
    lat_cols = [c for c in df.columns if 'lat' in c.lower()]
    lon_cols = [c for c in df.columns if 'lon' in c.lower()]
    print(f'  Lat columns: {lat_cols}')
    print(f'  Lon columns: {lon_cols}')

    # Use CS2 position for nearest-neighbour matching (per Alex's instruction)
    cs2_lat_col = [c for c in lat_cols if 'cs2' in c.lower()]
    cs2_lon_col = [c for c in lon_cols if 'cs2' in c.lower()]

    if cs2_lat_col and cs2_lon_col:
        lat_col = cs2_lat_col[0]
        lon_col = cs2_lon_col[0]
    else:
        # Fallback to first available
        lat_col = lat_cols[0] if lat_cols else 'lat'
        lon_col = lon_cols[0] if lon_cols else 'lon'

    print(f'  Using for NN matching: {lat_col}, {lon_col}')

    # Assign sectors
    df_month['sector'] = df_month[lon_col].apply(assign_sector)
    mask_sector = df_month['sector'].isin(['Western_Weddell', 'Eastern_Weddell', 'Ross'])
    df_out = df_month[mask_sector].copy()
    print(f'  Rows in Weddell+Ross sectors: {len(df_out):,}')
    print(f'  Sector breakdown:')
    print(df_out['sector'].value_counts().to_string(header=False))

    return df_out, lat_col, lon_col


def load_cci_month(cci_dir, year, month):
    """Load all CCI v4.0 L2P files for a given month."""
    # Directory structure: CCI_DIR/YYYY/MM/*.nc
    pattern = os.path.join(cci_dir, str(year), f'{month:02d}',
                           f'ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-SH-{year}{month:02d}*-fv4p0.nc')
    files = sorted(glob.glob(pattern))
    if not files:
        # Fallback: try CCI_DIR/MM/*.nc
        pattern = os.path.join(cci_dir, f'{month:02d}',
                               f'ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-SH-{year}{month:02d}*-fv4p0.nc')
        files = sorted(glob.glob(pattern))
    if not files:
        # Fallback: try flat directory
        pattern = os.path.join(cci_dir,
                               f'ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-SH-{year}{month:02d}*-fv4p0.nc')
        files = sorted(glob.glob(pattern))

    print(f'\nLoading CCI v4.0 L2P for {year}-{month:02d}: {len(files)} files')

    all_lat = []
    all_lon = []
    all_rfb = []
    all_sifb = []
    all_snow = []
    all_sit = []
    all_rfb_unc = []
    all_snow_unc = []
    all_time = []

    for f in files:
        ds = xr.open_dataset(f)
        all_lat.append(ds['lat'].values)
        all_lon.append(ds['lon'].values)
        all_rfb.append(ds['radar_freeboard'].values)
        all_sifb.append(ds['sea_ice_freeboard'].values)
        all_snow.append(ds['snow_depth'].values)
        all_sit.append(ds['sea_ice_thickness'].values)
        all_rfb_unc.append(ds['radar_freeboard_uncertainty'].values)
        all_snow_unc.append(ds['snow_depth_uncertainty'].values)
        all_time.append(ds['time'].values)
        ds.close()

    cci = {
        'lat':      np.concatenate(all_lat),
        'lon':      np.concatenate(all_lon),
        'radar_freeboard':     np.concatenate(all_rfb),
        'sea_ice_freeboard':   np.concatenate(all_sifb),
        'snow_depth':          np.concatenate(all_snow),
        'sea_ice_thickness':   np.concatenate(all_sit),
        'radar_freeboard_unc': np.concatenate(all_rfb_unc),
        'snow_depth_unc':      np.concatenate(all_snow_unc),
        'time':                np.concatenate(all_time),
    }
    print(f'  Total CCI points loaded: {len(cci["lat"]):,}')
    return cci


def load_cryotempo_month(ct_dir, year, month):
    """Load all Cryo-TEMPO L2P files for a given month."""
    month_dir = os.path.join(ct_dir, str(year), f'{month:02d}')
    files = sorted(glob.glob(os.path.join(month_dir, '*.nc')))
    print(f'\nLoading Cryo-TEMPO L2P for {year}-{month:02d}: {len(files)} files')

    all_lat = []
    all_lon = []
    all_rfb = []
    all_sifb = []
    all_snow = []
    all_rfb_unc = []
    all_snow_unc = []
    all_time = []

    for f in files:
        ds = xr.open_dataset(f)
        # Cryo-TEMPO stores lat/lon as coordinates
        all_lat.append(ds['latitude'].values)
        all_lon.append(ds['longitude'].values)
        all_rfb.append(ds['radar_freeboard'].values)
        all_sifb.append(ds['sea_ice_freeboard'].values)
        all_snow.append(ds['snow_depth'].values)
        all_rfb_unc.append(ds['radar_freeboard_uncertainty'].values)
        all_snow_unc.append(ds['snow_depth_uncertainty'].values)
        all_time.append(ds['time'].values)
        ds.close()

    ct = {
        'lat':      np.concatenate(all_lat),
        'lon':      np.concatenate(all_lon),
        'radar_freeboard':     np.concatenate(all_rfb),
        'sea_ice_freeboard':   np.concatenate(all_sifb),
        'snow_depth':          np.concatenate(all_snow),
        'radar_freeboard_unc': np.concatenate(all_rfb_unc),
        'snow_depth_unc':      np.concatenate(all_snow_unc),
        'time':                np.concatenate(all_time),
    }
    print(f'  Total Cryo-TEMPO points loaded: {len(ct["lat"]):,}')
    return ct


def collocate_nearest_neighbour(c2i_lat, c2i_lon, product_data, max_dist_km,
                                 product_name='Product'):
    """
    For each C2I segment, find the nearest product point using cKDTree.

    Returns:
      match_idx:  index into product arrays (-1 if no match within max_dist)
      match_dist: distance in km to nearest match
    """
    print(f'\nCollocating {len(c2i_lat):,} C2I segments with {len(product_data["lat"]):,} {product_name} points...')
    t0 = timer.time()

    # Build tree from product points
    prod_xyz = latlon_to_cartesian(product_data['lat'], product_data['lon'])
    tree = cKDTree(prod_xyz)

    # Query for each C2I segment
    c2i_xyz = latlon_to_cartesian(c2i_lat, c2i_lon)
    distances, indices = tree.query(c2i_xyz, k=1)

    # distances are in km (because we used Earth radius in km for Cartesian)
    # But cKDTree returns Euclidean distance in 3D — convert chord to arc
    # For small distances, chord ≈ arc, but let's be precise
    chord_km = distances
    arc_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(chord_km / (2.0 * EARTH_RADIUS_KM))

    # Apply max distance filter
    valid = arc_km <= max_dist_km
    match_idx = np.where(valid, indices, -1)
    match_dist = np.where(valid, arc_km, np.nan)

    n_matched = np.sum(valid)
    elapsed = timer.time() - t0
    print(f'  Matched: {n_matched:,} / {len(c2i_lat):,} '
          f'({100*n_matched/len(c2i_lat):.1f}%) within {max_dist_km} km')
    print(f'  Median match distance: {np.nanmedian(match_dist):.2f} km')
    print(f'  Mean  match distance:  {np.nanmean(match_dist):.2f} km')
    print(f'  Elapsed: {elapsed:.1f} s')

    return match_idx, match_dist


def extract_matched_values(product_data, match_idx, prefix):
    """Extract product values at matched indices into a dict with prefixed keys."""
    n = len(match_idx)
    valid = match_idx >= 0

    result = {}
    for key in ['radar_freeboard', 'sea_ice_freeboard', 'snow_depth',
                'radar_freeboard_unc', 'snow_depth_unc']:
        if key in product_data:
            arr = np.full(n, np.nan)
            arr[valid] = product_data[key][match_idx[valid]]
            result[f'{prefix}_{key}'] = arr

    # SIT (CCI only)
    if 'sea_ice_thickness' in product_data:
        arr = np.full(n, np.nan)
        arr[valid] = product_data['sea_ice_thickness'][match_idx[valid]]
        result[f'{prefix}_sea_ice_thickness'] = arr

    return result


# ===========================================================================
#  MAIN
# ===========================================================================
if __name__ == '__main__':
    t_start = timer.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load C2I segments for August 2022
    # ------------------------------------------------------------------
    c2i_df, lat_col, lon_col = load_c2i_august2022(C2I_CSV, POC_YEAR, POC_MONTH)

    if len(c2i_df) == 0:
        print('\nERROR: No C2I segments found for August 2022. Check CSV column names.')
        print('Available columns:')
        df_tmp = pd.read_csv(C2I_CSV, nrows=1)
        print(list(df_tmp.columns))
        raise SystemExit(1)

    c2i_lat = c2i_df[lat_col].values
    c2i_lon = c2i_df[lon_col].values

    # ------------------------------------------------------------------
    # 2. Load CCI v4.0 L2P for August 2022
    # ------------------------------------------------------------------
    cci_data = load_cci_month(CCI_DIR, POC_YEAR, POC_MONTH)

    # ------------------------------------------------------------------
    # 3. Load Cryo-TEMPO L2P for August 2022
    # ------------------------------------------------------------------
    ct_data = load_cryotempo_month(CT_DIR, POC_YEAR, POC_MONTH)

    # ------------------------------------------------------------------
    # 4. Collocate: C2I → CCI nearest neighbour
    # ------------------------------------------------------------------
    cci_idx, cci_dist = collocate_nearest_neighbour(
        c2i_lat, c2i_lon, cci_data, MAX_DIST_KM, 'CCI v4.0')

    # ------------------------------------------------------------------
    # 5. Collocate: C2I → Cryo-TEMPO nearest neighbour
    # ------------------------------------------------------------------
    ct_idx, ct_dist = collocate_nearest_neighbour(
        c2i_lat, c2i_lon, ct_data, MAX_DIST_KM, 'Cryo-TEMPO')

    # ------------------------------------------------------------------
    # 6. Build output DataFrame
    # ------------------------------------------------------------------
    print('\nBuilding output DataFrame...')
    out_df = c2i_df.copy()
    out_df['cci_match_dist_km'] = cci_dist
    out_df['ct_match_dist_km'] = ct_dist

    # Extract matched CCI values
    cci_vals = extract_matched_values(cci_data, cci_idx, 'cci')
    for k, v in cci_vals.items():
        out_df[k] = v

    # Extract matched Cryo-TEMPO values
    ct_vals = extract_matched_values(ct_data, ct_idx, 'ct')
    for k, v in ct_vals.items():
        out_df[k] = v

    # ------------------------------------------------------------------
    # 7. Summary statistics
    # ------------------------------------------------------------------
    print(f'\n{"="*70}')
    print('COLLOCATION SUMMARY — AUGUST 2022 PROOF OF CONCEPT')
    print(f'{"="*70}')

    for product, dist_col, snow_col, rfb_col in [
        ('CCI v4.0',    'cci_match_dist_km', 'cci_snow_depth', 'cci_radar_freeboard'),
        ('Cryo-TEMPO',  'ct_match_dist_km',  'ct_snow_depth',  'ct_radar_freeboard'),
    ]:
        valid = out_df[dist_col].notna()
        print(f'\n{product}:')
        print(f'  Total C2I segments:     {len(out_df):,}')
        print(f'  Matched (≤{MAX_DIST_KM} km):     {valid.sum():,} ({100*valid.mean():.1f}%)')
        print(f'  Match distance — median: {out_df.loc[valid, dist_col].median():.2f} km, '
              f'mean: {out_df.loc[valid, dist_col].mean():.2f} km')

        if valid.sum() > 0:
            print(f'  Snow depth — median: {out_df.loc[valid, snow_col].median():.3f} m, '
                  f'mean: {out_df.loc[valid, snow_col].mean():.3f} m')
            print(f'  Radar FB  — median: {out_df.loc[valid, rfb_col].median():.3f} m, '
                  f'mean: {out_df.loc[valid, rfb_col].mean():.3f} m')

        # Per-sector breakdown
        for sector in ['Western_Weddell', 'Eastern_Weddell', 'Ross']:
            mask_s = (out_df['sector'] == sector) & valid
            n_s = mask_s.sum()
            if n_s > 0:
                print(f'  {sector:20s}: {n_s:>6,} matched, '
                      f'snow median={out_df.loc[mask_s, snow_col].median():.3f} m, '
                      f'rfb median={out_df.loc[mask_s, rfb_col].median():.3f} m')

    # Compare C2I snow vs product snow for matched segments
    print(f'\n{"="*70}')
    print('C2I vs PRODUCT SNOW DEPTH COMPARISON (matched segments)')
    print(f'{"="*70}')

    # Identify C2I snow column
    snow_c2i_cols = [c for c in out_df.columns if 'snow' in c.lower() and 'cci' not in c.lower()
                     and 'ct_' not in c.lower() and 'amsr' not in c.lower()
                     and 'unc' not in c.lower()]
    print(f'  C2I snow columns found: {snow_c2i_cols}')

    if snow_c2i_cols:
        c2i_snow_col = snow_c2i_cols[0]
        for product, dist_col, prod_snow_col in [
            ('CCI v4.0',   'cci_match_dist_km', 'cci_snow_depth'),
            ('Cryo-TEMPO', 'ct_match_dist_km',  'ct_snow_depth'),
        ]:
            valid = out_df[dist_col].notna()
            if valid.sum() > 0:
                c2i_s = out_df.loc[valid, c2i_snow_col].values
                prod_s = out_df.loc[valid, prod_snow_col].values
                both_valid = np.isfinite(c2i_s) & np.isfinite(prod_s)
                if both_valid.sum() > 0:
                    diff = c2i_s[both_valid] - prod_s[both_valid]
                    corr = np.corrcoef(c2i_s[both_valid], prod_s[both_valid])[0, 1]
                    print(f'\n  {product} (n={both_valid.sum():,}):')
                    print(f'    C2I snow — median: {np.median(c2i_s[both_valid]):.3f} m, '
                          f'mean: {np.mean(c2i_s[both_valid]):.3f} m')
                    print(f'    {product} snow — median: {np.median(prod_s[both_valid]):.3f} m, '
                          f'mean: {np.mean(prod_s[both_valid]):.3f} m')
                    print(f'    Difference (C2I - {product}) — median: {np.median(diff):.3f} m, '
                          f'mean: {np.mean(diff):.3f} m')
                    print(f'    Pearson r: {corr:.4f}')

    # ------------------------------------------------------------------
    # 8. Save output
    # ------------------------------------------------------------------
    out_file = os.path.join(OUT_DIR, f'R3_L2P_collocation_aug{POC_YEAR}.csv')
    out_df.to_csv(out_file, index=False)
    print(f'\nOutput saved to:\n  {out_file}')
    print(f'\nTotal elapsed: {timer.time() - t_start:.1f} s')

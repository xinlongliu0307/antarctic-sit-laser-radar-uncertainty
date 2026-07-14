#!/usr/bin/env python3
"""
R3_Step08c_L2P_collocation_full.py
===================================
Scale the August 2022 proof-of-concept to all available winter months:
  - CCI v4.0 L2P:  May–Oct 2022, May–Oct 2023  (12 months)
  - Cryo-TEMPO L2P: May–Oct 2022–2025           (24 months)

The CCI v4.0 L2P product extends only to April 2024, covering winters
2022 and 2023 within the C2I study period (2022–2025). Cryo-TEMPO
extends through 2025, providing full four-winter coverage. This
asymmetry is a data-availability constraint, not a methodological
limitation — CCI and Cryo-TEMPO are collocated independently.

Method:
  Nearest-neighbour matching from C2I CS2 segment location to
  CCI/Cryo-TEMPO along-track CS2 location, with a maximum distance
  cutoff of 25 km (per Alex's instruction, 8 Apr 2026).

Input:
  - C2I matchup CSV: snow_thickness_with_amsr2_all.csv
  - CCI v4.0 L2P daily files (2022–2023 winters)
  - Cryo-TEMPO L2P orbit files (2022–2025 winters)

Output:
  - Per-month CSVs:  R3_L2P_collocation_YYYYMM.csv
  - Combined CSV:    R3_L2P_collocation_all_winters.csv
  - Summary report:  R3_L2P_collocation_summary.txt

Author: Xinlong Liu
Date:   09 Apr 2026
Environment: cryo2ice_env on NCI Gadi (project gv90)
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
import sys
import time as timer
from scipy.spatial import cKDTree

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
C2I_CSV = '/g/data/gv90/xl1657/phd/M2_workspace/output/snow_thickness/snow_thickness_with_amsr2_all.csv'
CCI_DIR = '/g/data/gv90/xl1657/phd/M1_workspace/raw_data/CCI_CS2/v4p0_L2P'
CT_DIR  = '/g/data/gv90/xl1657/phd/M1_workspace/raw_data/CryoTEMPO/TEMPO_POCA_SI/ALONGTRACK'
OUT_DIR = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation'

MAX_DIST_KM = 25.0
EARTH_RADIUS_KM = 6371.0

# Define which months each product covers
CCI_MONTHS = [(2022, m) for m in range(5, 11)] + \
             [(2023, m) for m in range(5, 11)]       # 12 months

CT_MONTHS  = [(y, m) for y in range(2022, 2026)
                      for m in range(5, 11)]          # 24 months

WINTER_MONTHS = sorted(set(CCI_MONTHS + CT_MONTHS))  # Union: 24 unique months


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
    """Assign sector label based on longitude."""
    lon = lon % 360
    if 298.0 <= lon <= 375.0 or lon <= 15.0:
        lon_orig = lon if lon <= 180 else lon - 360
        if lon_orig < -45.0:
            return 'Western_Weddell'
        else:
            return 'Eastern_Weddell'
    elif 160.0 <= lon <= 220.0:
        return 'Ross'
    else:
        return 'Other'


def load_c2i_all(csv_path):
    """Load entire C2I dataset, assign sectors, and index by year_month."""
    print(f'Loading C2I matchup data from:\n  {csv_path}')
    df = pd.read_csv(csv_path)
    print(f'  Total rows: {len(df):,}')

    # year_month is integer (e.g., 202208)
    df['ym_int'] = df['year_month'].astype(int)

    # Assign sectors
    df['sector'] = df['lon'].apply(assign_sector)
    mask_sector = df['sector'].isin(['Western_Weddell', 'Eastern_Weddell', 'Ross'])
    df = df[mask_sector].copy()
    print(f'  Rows in Weddell+Ross sectors: {len(df):,}')
    print(f'  Sector breakdown:')
    print(f'    {df["sector"].value_counts().to_string()}')

    # Report available year-months
    ym_counts = df['ym_int'].value_counts().sort_index()
    print(f'  Available year-months ({len(ym_counts)}):')
    for ym, cnt in ym_counts.items():
        print(f'    {str(ym)[:4]}-{str(ym)[4:]}: {cnt:,}')

    return df


def load_cci_month(cci_dir, year, month):
    """Load all CCI v4.0 L2P files for a given month."""
    # Try YYYY/MM/ structure first, then fallbacks
    for pattern in [
        os.path.join(cci_dir, str(year), f'{month:02d}',
                     f'ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-SH-{year}{month:02d}*-fv4p0.nc'),
        os.path.join(cci_dir, f'{month:02d}',
                     f'ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-SH-{year}{month:02d}*-fv4p0.nc'),
        os.path.join(cci_dir,
                     f'ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-SH-{year}{month:02d}*-fv4p0.nc'),
    ]:
        files = sorted(glob.glob(pattern))
        if files:
            break

    if not files:
        return None

    all_lat, all_lon = [], []
    all_rfb, all_sifb, all_snow, all_sit = [], [], [], []
    all_rfb_unc, all_snow_unc = [], []

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
        ds.close()

    return {
        'lat':  np.concatenate(all_lat),
        'lon':  np.concatenate(all_lon),
        'radar_freeboard':     np.concatenate(all_rfb),
        'sea_ice_freeboard':   np.concatenate(all_sifb),
        'snow_depth':          np.concatenate(all_snow),
        'sea_ice_thickness':   np.concatenate(all_sit),
        'radar_freeboard_unc': np.concatenate(all_rfb_unc),
        'snow_depth_unc':      np.concatenate(all_snow_unc),
    }


def load_cryotempo_month(ct_dir, year, month):
    """Load all Cryo-TEMPO L2P files for a given month."""
    month_dir = os.path.join(ct_dir, str(year), f'{month:02d}')
    files = sorted(glob.glob(os.path.join(month_dir, '*.nc')))

    if not files:
        return None

    all_lat, all_lon = [], []
    all_rfb, all_sifb, all_snow = [], [], []
    all_rfb_unc, all_snow_unc = [], []

    for f in files:
        ds = xr.open_dataset(f)
        all_lat.append(ds['latitude'].values)
        all_lon.append(ds['longitude'].values)
        all_rfb.append(ds['radar_freeboard'].values)
        all_sifb.append(ds['sea_ice_freeboard'].values)
        all_snow.append(ds['snow_depth'].values)
        all_rfb_unc.append(ds['radar_freeboard_uncertainty'].values)
        all_snow_unc.append(ds['snow_depth_uncertainty'].values)
        ds.close()

    return {
        'lat':  np.concatenate(all_lat),
        'lon':  np.concatenate(all_lon),
        'radar_freeboard':     np.concatenate(all_rfb),
        'sea_ice_freeboard':   np.concatenate(all_sifb),
        'snow_depth':          np.concatenate(all_snow),
        'radar_freeboard_unc': np.concatenate(all_rfb_unc),
        'snow_depth_unc':      np.concatenate(all_snow_unc),
    }


def collocate_nn(c2i_lat, c2i_lon, product_data, max_dist_km):
    """Nearest-neighbour collocation using cKDTree. Returns (indices, distances_km)."""
    prod_xyz = latlon_to_cartesian(product_data['lat'], product_data['lon'])
    tree = cKDTree(prod_xyz)

    c2i_xyz = latlon_to_cartesian(c2i_lat, c2i_lon)
    chord_dist, indices = tree.query(c2i_xyz, k=1)

    # Convert chord distance to arc distance
    arc_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(
        np.clip(chord_dist / (2.0 * EARTH_RADIUS_KM), 0, 1))

    valid = arc_km <= max_dist_km
    match_idx = np.where(valid, indices, -1)
    match_dist = np.where(valid, arc_km, np.nan)

    return match_idx, match_dist


def extract_matched(product_data, match_idx, prefix):
    """Extract product values at matched indices."""
    n = len(match_idx)
    valid = match_idx >= 0
    result = {}

    for key in ['radar_freeboard', 'sea_ice_freeboard', 'snow_depth',
                'radar_freeboard_unc', 'snow_depth_unc']:
        if key in product_data:
            arr = np.full(n, np.nan)
            arr[valid] = product_data[key][match_idx[valid]]
            result[f'{prefix}_{key}'] = arr

    if 'sea_ice_thickness' in product_data:
        arr = np.full(n, np.nan)
        arr[valid] = product_data['sea_ice_thickness'][match_idx[valid]]
        result[f'{prefix}_sea_ice_thickness'] = arr

    return result


# ===========================================================================
#  MAIN
# ===========================================================================
if __name__ == '__main__':
    t_global = timer.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Open summary report file
    summary_path = os.path.join(OUT_DIR, 'R3_L2P_collocation_summary.txt')
    summary_lines = []

    def log(msg):
        """Print and record to summary."""
        print(msg)
        summary_lines.append(msg)

    # ------------------------------------------------------------------
    # 1. Load all C2I segments (once)
    # ------------------------------------------------------------------
    c2i_all = load_c2i_all(C2I_CSV)

    # ------------------------------------------------------------------
    # 2. Process each month
    # ------------------------------------------------------------------
    all_monthly_dfs = []

    log(f'\n{"="*70}')
    log(f'PROCESSING {len(WINTER_MONTHS)} WINTER MONTHS')
    log(f'{"="*70}')
    log(f'CCI months:        {len(CCI_MONTHS)} (2022–2023)')
    log(f'Cryo-TEMPO months: {len(CT_MONTHS)} (2022–2025)')

    for year, month in WINTER_MONTHS:
        ym_int = year * 100 + month
        t_month = timer.time()

        log(f'\n{"─"*70}')
        log(f'{year}-{month:02d}')
        log(f'{"─"*70}')

        # Filter C2I to this month
        mask_ym = c2i_all['ym_int'] == ym_int
        c2i_month = c2i_all[mask_ym].copy()

        if len(c2i_month) == 0:
            log(f'  No C2I segments for {year}-{month:02d}, skipping.')
            continue

        log(f'  C2I segments: {len(c2i_month):,}')
        c2i_lat = c2i_month['lat'].values
        c2i_lon = c2i_month['lon'].values

        # --- CCI collocation (only for 2022–2023) ---
        has_cci = (year, month) in CCI_MONTHS
        if has_cci:
            cci_data = load_cci_month(CCI_DIR, year, month)
            if cci_data is not None:
                log(f'  CCI points loaded: {len(cci_data["lat"]):,}')
                cci_idx, cci_dist = collocate_nn(
                    c2i_lat, c2i_lon, cci_data, MAX_DIST_KM)
                n_cci = np.sum(cci_idx >= 0)
                log(f'  CCI matched: {n_cci:,} / {len(c2i_month):,} '
                    f'({100*n_cci/len(c2i_month):.1f}%), '
                    f'median dist: {np.nanmedian(cci_dist):.2f} km')

                c2i_month['cci_match_dist_km'] = cci_dist
                cci_vals = extract_matched(cci_data, cci_idx, 'cci')
                for k, v in cci_vals.items():
                    c2i_month[k] = v

                del cci_data  # free memory
            else:
                log(f'  CCI: no files found for {year}-{month:02d}')
                has_cci = False

        if not has_cci:
            # Fill CCI columns with NaN for months without CCI data
            for col in ['cci_match_dist_km', 'cci_radar_freeboard',
                        'cci_sea_ice_freeboard', 'cci_snow_depth',
                        'cci_radar_freeboard_unc', 'cci_snow_depth_unc',
                        'cci_sea_ice_thickness']:
                c2i_month[col] = np.nan

        # --- Cryo-TEMPO collocation (all 2022–2025) ---
        ct_data = load_cryotempo_month(CT_DIR, year, month)
        if ct_data is not None:
            log(f'  Cryo-TEMPO points loaded: {len(ct_data["lat"]):,}')
            ct_idx, ct_dist = collocate_nn(
                c2i_lat, c2i_lon, ct_data, MAX_DIST_KM)
            n_ct = np.sum(ct_idx >= 0)
            log(f'  Cryo-TEMPO matched: {n_ct:,} / {len(c2i_month):,} '
                f'({100*n_ct/len(c2i_month):.1f}%), '
                f'median dist: {np.nanmedian(ct_dist):.2f} km')

            c2i_month['ct_match_dist_km'] = ct_dist
            ct_vals = extract_matched(ct_data, ct_idx, 'ct')
            for k, v in ct_vals.items():
                c2i_month[k] = v

            del ct_data
        else:
            log(f'  Cryo-TEMPO: no files found for {year}-{month:02d}')
            for col in ['ct_match_dist_km', 'ct_radar_freeboard',
                        'ct_sea_ice_freeboard', 'ct_snow_depth',
                        'ct_radar_freeboard_unc', 'ct_snow_depth_unc']:
                c2i_month[col] = np.nan

        # Save per-month CSV
        month_csv = os.path.join(OUT_DIR, f'R3_L2P_collocation_{year}{month:02d}.csv')
        c2i_month.to_csv(month_csv, index=False)
        log(f'  Saved: {month_csv}')
        log(f'  Elapsed: {timer.time() - t_month:.1f} s')

        all_monthly_dfs.append(c2i_month)

    # ------------------------------------------------------------------
    # 3. Combine all months into a single CSV
    # ------------------------------------------------------------------
    log(f'\n{"="*70}')
    log('COMBINING ALL MONTHS')
    log(f'{"="*70}')

    df_all = pd.concat(all_monthly_dfs, ignore_index=True)
    combined_csv = os.path.join(OUT_DIR, 'R3_L2P_collocation_all_winters.csv')
    df_all.to_csv(combined_csv, index=False)
    log(f'Combined CSV saved: {combined_csv}')
    log(f'Total rows: {len(df_all):,}')

    # ------------------------------------------------------------------
    # 4. Summary statistics
    # ------------------------------------------------------------------
    log(f'\n{"="*70}')
    log('FULL-DATASET COLLOCATION SUMMARY')
    log(f'{"="*70}')

    # CCI summary (2022–2023 only)
    cci_valid = df_all['cci_match_dist_km'].notna()
    log(f'\nCCI v4.0 L2P (2022–2023):')
    log(f'  Total C2I segments in CCI period: '
        f'{df_all["ym_int"].isin([y*100+m for y,m in CCI_MONTHS]).sum():,}')
    log(f'  Matched (≤{MAX_DIST_KM} km): {cci_valid.sum():,}')
    if cci_valid.sum() > 0:
        log(f'  Match distance — median: {df_all.loc[cci_valid, "cci_match_dist_km"].median():.2f} km, '
            f'mean: {df_all.loc[cci_valid, "cci_match_dist_km"].mean():.2f} km')
        log(f'  CCI snow depth — median: {df_all.loc[cci_valid, "cci_snow_depth"].median():.3f} m, '
            f'mean: {df_all.loc[cci_valid, "cci_snow_depth"].mean():.3f} m')
        log(f'  CCI radar FB   — median: {df_all.loc[cci_valid, "cci_radar_freeboard"].median():.3f} m, '
            f'mean: {df_all.loc[cci_valid, "cci_radar_freeboard"].mean():.3f} m')

    # Cryo-TEMPO summary (2022–2025)
    ct_valid = df_all['ct_match_dist_km'].notna()
    log(f'\nCryo-TEMPO L2P (2022–2025):')
    log(f'  Total C2I segments: {len(df_all):,}')
    log(f'  Matched (≤{MAX_DIST_KM} km): {ct_valid.sum():,}')
    if ct_valid.sum() > 0:
        log(f'  Match distance — median: {df_all.loc[ct_valid, "ct_match_dist_km"].median():.2f} km, '
            f'mean: {df_all.loc[ct_valid, "ct_match_dist_km"].mean():.2f} km')
        log(f'  CT snow depth  — median: {df_all.loc[ct_valid, "ct_snow_depth"].median():.3f} m, '
            f'mean: {df_all.loc[ct_valid, "ct_snow_depth"].mean():.3f} m')
        log(f'  CT radar FB    — median: {df_all.loc[ct_valid, "ct_radar_freeboard"].median():.3f} m, '
            f'mean: {df_all.loc[ct_valid, "ct_radar_freeboard"].mean():.3f} m')

    # C2I snow comparison
    log(f'\nC2I snow thickness (all matched):')
    both_valid = cci_valid | ct_valid
    if both_valid.sum() > 0:
        log(f'  C2I snow — median: {df_all.loc[both_valid, "snow_thickness"].median():.3f} m, '
            f'mean: {df_all.loc[both_valid, "snow_thickness"].mean():.3f} m')

    # Per-sector, per-year breakdown
    log(f'\n{"="*70}')
    log('PER-SECTOR, PER-YEAR BREAKDOWN')
    log(f'{"="*70}')

    log(f'\n  {"Sector":<22s} {"Year":>5s} {"n_C2I":>7s} '
        f'{"n_CCI":>7s} {"n_CT":>7s} '
        f'{"C2I_snow":>9s} {"CCI_snow":>9s} {"CT_snow":>9s} '
        f'{"CCI_rfb":>9s} {"CT_rfb":>9s}')
    log(f'  {"-"*105}')

    for sector in ['Western_Weddell', 'Eastern_Weddell', 'Ross']:
        for year in [2022, 2023, 2024, 2025]:
            mask = ((df_all['sector'] == sector) &
                    (df_all['ym_int'] >= year * 100 + 5) &
                    (df_all['ym_int'] <= year * 100 + 10))
            n_c2i = mask.sum()
            if n_c2i == 0:
                continue

            n_cci = (mask & cci_valid).sum()
            n_ct = (mask & ct_valid).sum()

            c2i_snow_med = df_all.loc[mask, 'snow_thickness'].median()
            cci_snow_med = df_all.loc[mask & cci_valid, 'cci_snow_depth'].median() \
                if n_cci > 0 else np.nan
            ct_snow_med = df_all.loc[mask & ct_valid, 'ct_snow_depth'].median() \
                if n_ct > 0 else np.nan
            cci_rfb_med = df_all.loc[mask & cci_valid, 'cci_radar_freeboard'].median() \
                if n_cci > 0 else np.nan
            ct_rfb_med = df_all.loc[mask & ct_valid, 'ct_radar_freeboard'].median() \
                if n_ct > 0 else np.nan

            log(f'  {sector:<22s} {year:>5d} {n_c2i:>7,d} '
                f'{n_cci:>7,d} {n_ct:>7,d} '
                f'{c2i_snow_med:>9.3f} {cci_snow_med:>9.3f} {ct_snow_med:>9.3f} '
                f'{cci_rfb_med:>9.3f} {ct_rfb_med:>9.3f}')

    # ------------------------------------------------------------------
    # 5. Save summary report
    # ------------------------------------------------------------------
    elapsed_total = timer.time() - t_global
    log(f'\nTotal elapsed: {elapsed_total:.1f} s ({elapsed_total/60:.1f} min)')

    with open(summary_path, 'w') as fout:
        fout.write('\n'.join(summary_lines))
    print(f'\nSummary report saved: {summary_path}')
    print('Done.')

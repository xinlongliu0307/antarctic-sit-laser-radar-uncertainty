#!/usr/bin/env python3
"""
R3_Step01_verify_input_data.py
==============================
Verify M2 output CSV as R3 input — confirms matchup counts and snow
thickness statistics match the M2 manuscript before Monte Carlo runs.

Reads from:  /g/data/gv90/xl1657/phd/M2_workspace/output/snow_thickness/
Writes to:   /g/data/gv90/xl1657/phd/M3_workspace/output/

Actual CSV columns (confirmed 01 Apr 2026):
  lat, lon, cs2_rfb, is2_fb_wm, is2_fb_mean, is2_fb_std,
  is2_fb_unc_mean, n_is2, n_strong, n_weak, date, year_month,
  snow_thickness, snow_thickness_unc, AMSR2_SIC, AMSR2_snow_depth

Note: 'month' and 'sector' are NOT in the CSV — derived here from
      'date' (YYYYMMDD integer) and 'lon' respectively.

Environment: ~/cryo2ice_env/ on NCI Gadi (Python 3.9.2)
Author: Xinlong Liu | 01 Apr 2026
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────
# 0. Configuration
# ────────────────────────────────────────────────────────────────
M2_BASE = '/g/data/gv90/xl1657/phd/M2_workspace'
M3_BASE = '/g/data/gv90/xl1657/phd/M3_workspace'

CSV_PATH = os.path.join(M2_BASE, 'output', 'snow_thickness',
                        'snow_thickness_with_amsr2_all.csv')

OUT_DIR = os.path.join(M3_BASE, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# --- Column name mapping (verified from Gadi output 01 Apr 2026) ---
COL_LAT     = 'lat'
COL_LON     = 'lon'
COL_CS2_FB  = 'cs2_rfb'               # CS2 radar freeboard [m]
COL_IS2_FB  = 'is2_fb_wm'             # IS2 weighted-mean freeboard [m]
COL_IS2_STD = 'is2_fb_std'            # IS2 freeboard std within matchup
COL_HS      = 'snow_thickness'        # Retrieved snow thickness [m]
COL_HS_UNC  = 'snow_thickness_unc'    # Per-retrieval uncertainty [m]
COL_DATE    = 'date'                   # YYYYMMDD integer
COL_YM      = 'year_month'            # YYYYMM integer
COL_AMSR2   = 'AMSR2_snow_depth'      # AMSR2 snow depth [cm]
COL_SIC     = 'AMSR2_SIC'             # AMSR2 sea ice concentration [%]
COL_N_IS2   = 'n_is2'
COL_NSTRONG = 'n_strong'
COL_NWEAK   = 'n_weak'

# Study region longitude bounds
# Weddell Sea: 62°W to 15°E  →  lon in [-62, 15]
# Ross Sea:    160°E to 140°W →  lon in [160, 180] or [-180, -140]
WEDDELL_LON = (-62.0, 15.0)
ROSS_LON_POS = (160.0, 180.0)
ROSS_LON_NEG = (-180.0, -140.0)

t0 = time.time()

# ────────────────────────────────────────────────────────────────
# 1. Load M2 output
# ────────────────────────────────────────────────────────────────
print('=' * 65)
print('R3 Step 01: Verify M2 Output as R3 Input')
print('=' * 65)
print(f'  M2 source:  {M2_BASE}')
print(f'  M3 output:  {M3_BASE}')

if not os.path.isfile(CSV_PATH):
    sys.exit(f'ERROR: CSV not found at {CSV_PATH}')

df = pd.read_csv(CSV_PATH)
print(f'\nLoaded: {CSV_PATH}')
print(f'Total rows: {len(df):,}')
print(f'Columns ({len(df.columns)}): {list(df.columns)}\n')

# ────────────────────────────────────────────────────────────────
# 2. Derive month and sector from existing columns
# ────────────────────────────────────────────────────────────────
print('Deriving month and sector...')

# Month from date (YYYYMMDD) — extract via integer arithmetic
df['month'] = (df[COL_DATE] // 100) % 100
df['year']  = df[COL_DATE] // 10000

print(f'  Month range: {df["month"].min()} – {df["month"].max()}')
print(f'  Year  range: {df["year"].min()} – {df["year"].max()}')

# Sector from longitude
lon = df[COL_LON].values
weddell_mask = (lon >= WEDDELL_LON[0]) & (lon <= WEDDELL_LON[1])
ross_mask    = ((lon >= ROSS_LON_POS[0]) & (lon <= ROSS_LON_POS[1])) | \
               ((lon >= ROSS_LON_NEG[0]) & (lon <= ROSS_LON_NEG[1]))
other_mask   = ~weddell_mask & ~ross_mask

df['sector'] = 'Other'
df.loc[weddell_mask, 'sector'] = 'Weddell'
df.loc[ross_mask, 'sector']    = 'Ross'

print(f'\n  Sector assignment from longitude:')
print(f'    Weddell ({WEDDELL_LON[0]}° to {WEDDELL_LON[1]}°): '
      f'{weddell_mask.sum():,}')
print(f'    Ross ({ROSS_LON_POS[0]}° to {ROSS_LON_POS[1]}° / '
      f'{ROSS_LON_NEG[0]}° to {ROSS_LON_NEG[1]}°): {ross_mask.sum():,}')
print(f'    Other: {other_mask.sum():,}')

# ────────────────────────────────────────────────────────────────
# 3. Filter to winter (May–Oct) and study sectors (Weddell, Ross)
# ────────────────────────────────────────────────────────────────
winter_mask = df['month'].between(5, 10)
study_mask  = winter_mask & (df['sector'].isin(['Weddell', 'Ross']))

d = df[study_mask].copy().reset_index(drop=True)

n_wed = ((df['sector'] == 'Weddell') & winter_mask).sum()
n_ros = ((df['sector'] == 'Ross') & winter_mask).sum()
print(f'\nWinter (May–Oct) study-sector matchups: {len(d):,}')
print(f'  Weddell: {n_wed:,}')
print(f'  Ross:    {n_ros:,}')
print(f'  Sum:     {n_wed + n_ros:,}')

# ────────────────────────────────────────────────────────────────
# 4. Report key statistics — cross-check against M2 manuscript
# ────────────────────────────────────────────────────────────────
def report_col(label, series, unit='m'):
    v = series.dropna()
    if len(v) == 0:
        print(f'  {label}: NO DATA')
        return
    print(f'\n--- {label} (n={len(v):,}, unit={unit}) ---')
    print(f'  Mean (median): {v.mean():.4f} ({v.median():.4f})')
    print(f'  Std:           {v.std():.4f}')
    print(f'  Min / Max:     {v.min():.4f} / {v.max():.4f}')
    print(f'  5th / 95th %%:  {v.quantile(0.05):.4f} / {v.quantile(0.95):.4f}')
    if (v < 0).any():
        neg_frac = 100 * (v < 0).sum() / len(v)
        print(f'  Negative fraction: {neg_frac:.1f}%')

report_col('Snow thickness (h_s)', d[COL_HS])
report_col('Per-retrieval uncertainty', d[COL_HS_UNC])
report_col('CS2 radar freeboard', d[COL_CS2_FB])
report_col('IS2 total freeboard (wm)', d[COL_IS2_FB])
report_col('AMSR2 snow depth', d[COL_AMSR2], unit='cm')

# ────────────────────────────────────────────────────────────────
# 5. By-sector breakdown
# ────────────────────────────────────────────────────────────────
print('\n' + '=' * 50)
print('BY-SECTOR BREAKDOWN')
print('=' * 50)

for sector_name in ['Weddell', 'Ross']:
    sub = d[d['sector'] == sector_name]
    print(f'\n{sector_name} (n={len(sub):,}):')
    hs = sub[COL_HS].dropna()
    print(f'  h_s mean (median): {hs.mean():.4f} ({hs.median():.4f}) m')
    print(f'  h_s std:           {hs.std():.4f} m')
    unc = sub[COL_HS_UNC].dropna()
    print(f'  Uncertainty median: {unc.median():.4f} m')
    fb = sub[COL_CS2_FB].dropna()
    print(f'  CS2 fb mean:       {fb.mean():.4f} m')
    is2 = sub[COL_IS2_FB].dropna()
    print(f'  IS2 fb mean:       {is2.mean():.4f} m')

# ────────────────────────────────────────────────────────────────
# 6. Year × month coverage table
# ────────────────────────────────────────────────────────────────
print('\n--- Year × Month coverage (study sectors, winter) ---')
pivot = d.groupby(['year', 'month']).size().unstack(fill_value=0)
print(pivot.to_string())

# ────────────────────────────────────────────────────────────────
# 7. Save verified subset to M3_workspace
# ────────────────────────────────────────────────────────────────
out_pkl = os.path.join(OUT_DIR, 'R3_input_verified.pkl')
d.to_pickle(out_pkl)
print(f'\nSaved R3 input ({len(d):,} rows): {out_pkl}')

# Also save column mapping for reference
print('\n--- Column mapping used by R3 Steps 02–03 ---')
print(f'  CS2 freeboard:   {COL_CS2_FB}')
print(f'  IS2 freeboard:   {COL_IS2_FB}')
print(f'  Snow thickness:  {COL_HS}')
print(f'  h_s uncertainty: {COL_HS_UNC}')
print(f'  Sector:          sector (derived from {COL_LON})')
print(f'  Month:           month (derived from {COL_DATE})')

elapsed = time.time() - t0
print(f'\nStep 01 complete in {elapsed:.1f} seconds.')

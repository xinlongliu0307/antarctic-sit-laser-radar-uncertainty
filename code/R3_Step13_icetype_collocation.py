#!/usr/bin/env python3
"""
R3_Step13_icetype_collocation.py — Per-matchup sea-ice type from Melsheimer et al. (2023)
===========================================================================================
Answers Reviewer 3 major comment 7 and Reviewer 2 other comments 3-4 (RSE-D-26-03119):
ice type is assigned to each CRYO2ICE matchup from the Bremen ASCAT+AMSR2 product on the
matchup date, instead of by longitude sub-sector.

Inputs
  {M3_OUTPUT}/R3_input_d070_verified.pkl                      matchups (lat, lon, date, sector)
  {MELSHEIMER_DIR}/final/YYYY/MultiYearIce-Antarctic-YYYYMMDD.nc   MYI (drift-corrected), CR-FLAG
  {MELSHEIMER_DIR}/raw/YYYY/ECICE-IcetypesUncorrected-YYYYMMDD.nc  FYI, YI, TOTAL_ICE, LAT, LON

Grid: NSIDC south polar stereographic 12.5 km, EPSG:3412, 664 rows x 632 cols.
NOTE the NetCDF dimension names are misleading: dim "X" is the ROW axis (y decreasing
from +4343.75 km), dim "Y" is the COLUMN axis (x increasing from -3943.75 km).

Outputs
  {M3_OUTPUT}/R3_input_d070_icetype.pkl        input + myi_frac, fyi_frac, yi_frac, total_ice,
                                               cr_flag, icetype_date, icetype_day_offset, icetype
  {M3_OUTPUT}/R3_icetype_sector_composition.csv   -> Table tab:icetype
  {M3_FIGURES}/R3_Fig_icetype_map.png/.pdf

Run: source ~/cryo2ice_env/bin/activate && python R3_Step13_icetype_collocation.py
"""
import os, sys, time, glob
import datetime as dt
import numpy as np
import pandas as pd
import netCDF4
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *          # M3_OUTPUT, M3_FIGURES, MELSHEIMER_DIR, REGIONS, ...

t0 = time.time()

# ── Grid constants (verified from LAT/LON of a raw file) ──────────────────────
NROW, NCOL = 664, 632
CELL = 12_500.0                     # m
X0 = -3_943_750.0                   # x of column 0 centre
Y0 = 4_343_750.0                    # y of row 0 centre
MYI_THRESHOLD = 50.0                # % MYI at/above which a matchup is classed MYI
MAX_DAY_OFFSET = 2                  # fall back +-1, +-2 days if the date file is missing
NEIGHBOURHOOD = 1                   # 3x3 median as a robustness column (0 = off)

# ── Column detection ─────────────────────────────────────────────────────────
LAT_CANDIDATES = ['lat', 'is2_lat', 'lat_is2', 'latitude', 'lat_mean']
LON_CANDIDATES = ['lon', 'is2_lon', 'lon_is2', 'longitude', 'lon_mean']
DATE_CANDIDATES = ['date', 'yyyymmdd', 'date_int', 'is2_date', 'datetime', 'time']


def pick(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    sys.exit(f"Cannot find a {what} column. Tried {candidates}.\n"
             f"Columns present: {list(df.columns)}\n"
             f"Add the right name to {what.upper()}_CANDIDATES and rerun.")


def to_yyyymmdd(series):
    """Return integer YYYYMMDD from int, str or datetime-like columns."""
    if np.issubdtype(series.dtype, np.datetime64):
        return series.dt.strftime('%Y%m%d').astype(int).values
    s = series.astype(str).str.replace('-', '', regex=False).str.slice(0, 8)
    return s.astype(int).values


# ── Load matchups ────────────────────────────────────────────────────────────
df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_verified.pkl')
n = len(df)
lat_col, lon_col, date_col = (pick(df, LAT_CANDIDATES, 'lat'),
                              pick(df, LON_CANDIDATES, 'lon'),
                              pick(df, DATE_CANDIDATES, 'date'))
if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df[lon_col].values)
dates = to_yyyymmdd(df[date_col])
print(f"Loaded {n:,} matchups; columns used: lat={lat_col}, lon={lon_col}, date={date_col}")
print(f"Date span: {dates.min()} - {dates.max()}, {len(np.unique(dates))} unique days")

# ── Lon/lat -> row/col ───────────────────────────────────────────────────────
tr = Transformer.from_crs('EPSG:4326', 'EPSG:3412', always_xy=True)
x, y = tr.transform(df[lon_col].values.astype(float), df[lat_col].values.astype(float))
row = np.rint((Y0 - y) / CELL).astype(int)
col = np.rint((x - X0) / CELL).astype(int)
inb = (row >= 0) & (row < NROW) & (col >= 0) & (col < NCOL)
assert inb.all(), f"{(~inb).sum()} matchups fall outside the 664x632 grid"
df['ps_x_km'] = x / 1e3
df['ps_y_km'] = y / 1e3

# ── File lookup with +-day fallback ──────────────────────────────────────────
def find_file(product, ymd, max_off=MAX_DAY_OFFSET):
    d0 = dt.date(ymd // 10000, (ymd // 100) % 100, ymd % 100)
    tmpl = {'final': 'final/{y}/MultiYearIce-Antarctic-{d}.nc',
            'raw':   'raw/{y}/ECICE-IcetypesUncorrected-{d}.nc'}[product]
    for off in sorted(range(-max_off, max_off + 1), key=abs):
        d = d0 + dt.timedelta(days=off)
        p = os.path.join(MELSHEIMER_DIR, tmpl.format(y=d.year, d=d.strftime('%Y%m%d')))
        if os.path.exists(p):
            return p, off
    return None, None


def read_var(path, name):
    with netCDF4.Dataset(path) as ds:
        v = ds.variables[name][:]
        return np.ma.filled(v.astype(float), np.nan)


def sample(field, r, c, k=0):
    """Nearest cell (k=0) or nanmedian of a (2k+1)^2 neighbourhood."""
    if k == 0:
        return field[r, c]
    out = np.full(len(r), np.nan)
    for i, (ri, ci) in enumerate(zip(r, c)):
        blk = field[max(ri - k, 0):ri + k + 1, max(ci - k, 0):ci + k + 1]
        if np.isfinite(blk).any():
            out[i] = np.nanmedian(blk)
    return out


# ── Collocate, one date at a time ────────────────────────────────────────────
cols = {k: np.full(n, np.nan) for k in
        ['myi_frac', 'myi_frac_3x3', 'fyi_frac', 'yi_frac', 'total_ice', 'cr_flag']}
icetype_date = np.zeros(n, dtype=int)
day_offset = np.full(n, 99, dtype=int)

uniq = np.unique(dates)
missing = []
for k, ymd in enumerate(uniq):
    idx = np.where(dates == ymd)[0]
    pf, off_f = find_file('final', ymd)
    pr, off_r = find_file('raw', ymd)
    if pf is None or pr is None:
        missing.append(ymd)
        continue
    myi = read_var(pf, 'MYI')
    crf = read_var(pf, 'CR-FLAG')
    fyi = read_var(pr, 'FYI')
    yi = read_var(pr, 'YI')
    tot = read_var(pr, 'TOTAL_ICE')
    r, c = row[idx], col[idx]
    cols['myi_frac'][idx] = sample(myi, r, c)
    cols['myi_frac_3x3'][idx] = sample(myi, r, c, NEIGHBOURHOOD)
    cols['cr_flag'][idx] = sample(crf, r, c)
    cols['fyi_frac'][idx] = sample(fyi, r, c)
    cols['yi_frac'][idx] = sample(yi, r, c)
    cols['total_ice'][idx] = sample(tot, r, c)
    icetype_date[idx] = int(os.path.basename(pf)[-11:-3])
    day_offset[idx] = max(abs(off_f), abs(off_r))
    if (k + 1) % 50 == 0 or k == len(uniq) - 1:
        print(f"  {k + 1}/{len(uniq)} days ({time.time() - t0:.0f}s)")

for k_, v in cols.items():
    df[k_] = v
df['icetype_date'] = icetype_date
df['icetype_day_offset'] = day_offset
df['icetype'] = np.where(np.isnan(df['myi_frac']), 'unclassified',
                         np.where(df['myi_frac'] >= MYI_THRESHOLD, 'MYI', 'FYI'))

print(f"\nDays with no product file within +-{MAX_DAY_OFFSET} d: {len(missing)} "
      f"({missing[:5]}{' ...' if len(missing) > 5 else ''})")
print(f"Matchups: {(df.icetype != 'unclassified').sum():,} classified, "
      f"{(df.icetype == 'unclassified').sum():,} unclassified "
      f"(NaN cell or missing day); {(day_offset > 0).sum():,} used a +-day fallback")

df.to_pickle(f'{M3_OUTPUT}/R3_input_d070_icetype.pkl')
print(f"Saved: {M3_OUTPUT}/R3_input_d070_icetype.pkl")

# ── Sector composition table (tab:icetype) ───────────────────────────────────
df['winter'] = dates // 10000
rows = []
for region in REGIONS + ['All']:
    m_reg = (np.ones(n, bool) if region == 'All'
             else get_region_mask(df['sector'].values, df['subsector'].values, region))
    for winter in ['all'] + sorted(df['winter'].unique().tolist()):
        m = m_reg & (np.ones(n, bool) if winter == 'all' else (df['winter'] == winter).values)
        if m.sum() == 0:
            continue
        sub = df[m]
        cl = sub[sub.icetype != 'unclassified']
        rows.append(dict(
            region=region, winter=winter, n=int(m.sum()), n_classified=len(cl),
            pct_MYI=100 * (cl.icetype == 'MYI').mean() if len(cl) else np.nan,
            mean_myi_frac=cl.myi_frac.mean(), median_myi_frac=cl.myi_frac.median(),
            mean_fyi_frac=cl.fyi_frac.mean(), mean_yi_frac=cl.yi_frac.mean(),
            mean_total_ice=cl.total_ice.mean(),
            pct_corrected=100 * (cl.cr_flag > 0).mean() if len(cl) else np.nan,
            pct_day_fallback=100 * (sub.icetype_day_offset > 0).mean(),
            # what the longitude rule (config v4) would have assigned
            pct_MYI_by_longitude=100 * (sub.subsector == 'Weddell_W').mean(),
        ))
comp = pd.DataFrame(rows)
comp.to_csv(f'{M3_OUTPUT}/R3_icetype_sector_composition.csv', index=False, float_format='%.1f')
print(f"Saved: {M3_OUTPUT}/R3_icetype_sector_composition.csv\n")
pd.set_option('display.width', 160)
print(comp[comp.winter == 'all'][['region', 'n', 'n_classified', 'pct_MYI', 'mean_myi_frac',
                                  'mean_fyi_frac', 'pct_MYI_by_longitude', 'pct_corrected']]
      .to_string(index=False))

# ── Map ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 7.5))
cl = df[df.icetype != 'unclassified']
sc = ax.scatter(cl.ps_x_km, cl.ps_y_km, c=cl.myi_frac, s=1.5, cmap='viridis', vmin=0, vmax=100,
                rasterized=True)
un = df[df.icetype == 'unclassified']
ax.scatter(un.ps_x_km, un.ps_y_km, s=1.5, c='lightgrey', label=f'unclassified (n={len(un):,})')
cb = fig.colorbar(sc, ax=ax, shrink=0.8, label='Drift-corrected MYI concentration (%)')
ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
ax.set_aspect('equal'); ax.set_xlabel('x (km, EPSG:3412)'); ax.set_ylabel('y (km, EPSG:3412)')
ax.set_title(f'CRYO2ICE matchups coloured by Melsheimer et al. (2023) MYI on the matchup date\n'
             f'n = {len(cl):,} classified; MYI class at $\\geq$ {MYI_THRESHOLD:.0f} %', fontsize=10)
ax.legend(loc='lower left', fontsize=8, markerscale=4)
fig.tight_layout()
fig.savefig(f'{M3_FIGURES}/R3_Fig_icetype_map.png', dpi=200)
fig.savefig(f'{M3_FIGURES}/R3_Fig_icetype_map.pdf')
print(f"Saved: {M3_FIGURES}/R3_Fig_icetype_map.png/pdf")

print(f"\nStep 13 complete in {time.time() - t0:.1f} s")

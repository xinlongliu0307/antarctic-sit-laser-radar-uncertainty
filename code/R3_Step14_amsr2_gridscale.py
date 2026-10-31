#!/usr/bin/env python3
"""
R3_Step14_amsr2_gridscale.py — CRYO2ICE vs AMSR2 snow at the AMSR2 grid scale
==============================================================================
Answers Reviewer 1 comment 3, Reviewer 2 other comments 6-7 and Reviewer 3
major comment 10 (RSE-D-26-03119): the along-track CRYO2ICE snow thickness is
aggregated to the AMSR2 grid and matched to the same day before comparison,
so that many matchups are no longer compared with one grid value.

Scales
  retrieval : one matchup vs the AMSR2 cell value it fell in (as submitted)
  12.5 km   : matchups averaged within each AU_SI12 cell on the matchup date
  25 km     : matchups averaged within 2x2 blocks of AU_SI12 cells on the date

The AU_SI12 product is on the NSIDC 12.5 km south polar stereographic grid
(EPSG:3412, 632 x 664), the same grid as the Melsheimer product, so the cell
index is obtained by projecting the matchup lon/lat exactly as in Step 13.
The AMSR2 value per cell-day is the mean of the (identical) per-matchup values
that Step 01 attached to the matchups from that cell.

Statistics per sector and scale: n, mean C2I, mean AMSR2, mean bias
(AMSR2 - C2I), RMSD, Pearson r and r^2, Spearman rho, and the sector-level
Weddell - Ross contrast in each product.

Input : {M3_OUTPUT}/R3_input_d070_icetype.pkl
        (needs lat, lon, date, hs_d070, AMSR2_snow_depth, sector, subsector)
Output: {M3_OUTPUT}/R3_amsr2_gridscale_v7.csv           statistics table
        {M3_OUTPUT}/R3_amsr2_gridscale_cells_v7.csv     aggregated cell-day pairs
        {M3_FIGURES}/R3_Fig_amsr2_gridscale.{png,pdf}
Run   : source ~/cryo2ice_env/bin/activate && python R3_Step14_amsr2_gridscale.py
"""
import os, sys, time
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

t0 = time.time()
CELL, X0, Y0 = 12_500.0, -3_943_750.0, 4_343_750.0      # as in Step 13
MIN_PER_CELL = 3                                         # matchups per cell-day
AMSR2_VALID = (0.0, 1.5)                                 # m, as in Step 04
SCALES = ['retrieval', '12.5km', '25km']
SECTORS = ['Weddell', 'Weddell_W', 'Weddell_E', 'Ross', 'All']

print("=" * 72)
print("R3 Step 14: CRYO2ICE vs AMSR2 snow thickness at the AMSR2 grid scale")
print("=" * 72)

df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_icetype.pkl')
if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df['lon'].values)
am = df['AMSR2_snow_depth'].values.astype(float)
flag = np.isin(np.round(am * 100), [110, 120, 130, 140, 150, 160])   # v8: AU_SI12 flag codes (cm)
print(f'AU_SI12 flag values masked: {int(flag.sum()):,}')
am[flag] = np.nan
df['AMSR2_snow_depth'] = am
ok = np.isfinite(am) & (am > AMSR2_VALID[0]) & (am <= AMSR2_VALID[1]) & np.isfinite(df['hs_d070'].values)
df = df[ok].copy()
print(f"Matchups with a valid AMSR2 snow value: {len(df):,} of {ok.size:,}")

tr = Transformer.from_crs('EPSG:4326', 'EPSG:3412', always_xy=True)
x, y = tr.transform(df['lon'].values.astype(float), df['lat'].values.astype(float))
df['row'] = np.rint((Y0 - y) / CELL).astype(int)
df['col'] = np.rint((x - X0) / CELL).astype(int)
df['row25'] = df['row'] // 2
df['col25'] = df['col'] // 2
df['region_key'] = np.where(df['subsector'] != '', df['subsector'], df['sector'])


def aggregate(d, keys):
    g = d.groupby(keys)
    out = g.agg(n=('hs_d070', 'size'), c2i=('hs_d070', 'mean'), c2i_std=('hs_d070', 'std'),
                amsr2=('AMSR2_snow_depth', 'mean'), amsr2_std=('AMSR2_snow_depth', 'std'),
                sector=('sector', lambda s: s.mode().iloc[0]),
                subsector=('subsector', lambda s: s.mode().iloc[0]),
                myi_frac=('myi_frac', 'mean')).reset_index()
    return out[out.n >= MIN_PER_CELL].copy()


pairs = {'retrieval': df[['date', 'hs_d070', 'AMSR2_snow_depth', 'sector', 'subsector', 'myi_frac']]
                        .rename(columns={'hs_d070': 'c2i', 'AMSR2_snow_depth': 'amsr2'}).assign(n=1),
         '12.5km': aggregate(df, ['date', 'row', 'col']),
         '25km': aggregate(df, ['date', 'row25', 'col25'])}
for k, v in pairs.items():
    print(f"  {k:10s}: {len(v):,} pairs" + ("" if k == 'retrieval' else f", median {v.n.median():.0f} matchups per cell-day"))


def region_mask(d, reg):
    if reg == 'All':
        return np.ones(len(d), bool)
    if reg in ('Weddell_W', 'Weddell_E'):
        return d['subsector'].values == reg
    return d['sector'].values == reg


rows = []
for scale in SCALES:
    d = pairs[scale]
    for reg in SECTORS:
        m = region_mask(d, reg)
        if m.sum() < 10:
            continue
        a, c = d['amsr2'].values[m], d['c2i'].values[m]
        r, p = pearsonr(a, c)
        rs, ps = spearmanr(a, c)
        rows.append(dict(scale=scale, region=reg, n=int(m.sum()),
                         mean_c2i=c.mean(), mean_amsr2=a.mean(), median_c2i=np.median(c), median_amsr2=np.median(a),
                         bias=(a - c).mean(), rmsd=np.sqrt(((a - c) ** 2).mean()),
                         pearson_r=r, r2=r ** 2, pearson_p=p, spearman_rho=rs, spearman_p=ps,
                         amsr2_excess_pct=100 * (np.median(a) / np.median(c) - 1)))
stats = pd.DataFrame(rows)
stats.to_csv(f'{M3_OUTPUT}/R3_amsr2_gridscale_v7.csv', index=False, float_format='%.4f')
pd.concat([pairs[k].assign(scale=k) for k in ['12.5km', '25km']]).to_csv(
    f'{M3_OUTPUT}/R3_amsr2_gridscale_cells_v7.csv', index=False, float_format='%.4f')

pd.set_option('display.width', 200)
print("\n" + stats[['scale', 'region', 'n', 'mean_c2i', 'mean_amsr2', 'bias', 'rmsd', 'pearson_r', 'r2', 'spearman_rho']]
      .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

print("\nSector-level Weddell - Ross contrast (mean snow, m):")
for scale in SCALES:
    s = stats[stats.scale == scale].set_index('region')
    print(f"  {scale:10s}: C2I {s.loc['Weddell', 'mean_c2i'] - s.loc['Ross', 'mean_c2i']:+.3f}   "
          f"AMSR2 {s.loc['Weddell', 'mean_amsr2'] - s.loc['Ross', 'mean_amsr2']:+.3f}")

# ── Figure: scatter at three scales + r^2 by sector ───────────────────
fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw={'width_ratios': [1, 1, 1, 0.9]})
for ax, scale in zip(axes[:3], SCALES):
    d = pairs[scale]
    for reg, col in [('Weddell', C_WED), ('Ross', C_ROSS)]:
        m = region_mask(d, reg)
        ax.scatter(d['c2i'].values[m], d['amsr2'].values[m], s=2 if scale == 'retrieval' else 6,
                   color=col, alpha=0.25 if scale == 'retrieval' else 0.5, label=reg, rasterized=True)
    ax.plot([0, 1], [0, 1], 'k:', lw=0.8)
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8); ax.set_aspect('equal')
    ax.set_xlabel('CRYO2ICE snow thickness (m)')
    ax.set_ylabel('AMSR2 snow depth (m)' if scale == 'retrieval' else '')
    s = stats[(stats.scale == scale) & (stats.region == 'All')].iloc[0]
    ax.set_title(f'{scale} (n={s.n:,}; r$^2$={s.r2:.3f}, bias={s.bias:+.3f} m)', fontsize=9.5)
    ax.legend(fontsize=8, loc='upper left', frameon=False, markerscale=4)
ax = axes[3]
xr = np.arange(len(SECTORS))
w = 0.27
for i, scale in enumerate(SCALES):
    s = stats[stats.scale == scale].set_index('region').reindex(SECTORS)
    ax.bar(xr + (i - 1) * w, s['r2'], w, label=scale, alpha=0.85, edgecolor='white')
ax.set_xticks(xr)
ax.set_xticklabels([REGION_LABELS.get(r, r).replace(' Sea', '') for r in SECTORS], fontsize=8, rotation=20)
ax.set_ylabel('Pearson r$^2$')
ax.set_title('Explained variance by sector and scale', fontsize=9.5)
ax.legend(fontsize=8, frameon=False)
ax.grid(axis='y', alpha=0.25)
fig.suptitle('CRYO2ICE against AMSR2 (AU_SI12) snow on the matchup date at the retrieval, 12.5 km and 25 km scales',
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ('png', 'pdf'):
    fig.savefig(f'{M3_FIGURES}/R3_Fig_amsr2_gridscale.{ext}')
print(f"\nSaved: R3_amsr2_gridscale_v7.csv, R3_amsr2_gridscale_cells_v7.csv, R3_Fig_amsr2_gridscale.png/pdf")
print(f"Step 14 complete in {time.time() - t0:.1f} s")

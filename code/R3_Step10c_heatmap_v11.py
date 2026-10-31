#!/usr/bin/env python3
"""
R3_Step10c_heatmap_v11.py - redraws Figure R3_Fig_C2_03 (sector x year snow-replacement heatmap) only,
from the v8-fixed snow-swap CSV (CRYO2ICE snow at delta = 0.70).

Layout (matches the manuscript caption):
  row 1  Snow     (a) CCI 2022-2023        (b) Cryo-TEMPO 2022-2025    AMSR-E/AMSR2 excess over CRYO2ICE (%)
  row 2  Sea ice  (c) CCI 2022-2023        (d) Cryo-TEMPO 2022-2025    median per-matchup dh_i (m)
Definitions are identical to Step 10 v9:
  excess (%) = 100 * (median L2 snow - median C2I snow) / median C2I snow, per sector-year
  dh_i (m)   = median over matchups of (h_i with C2I snow - h_i with AMSR2 snow), per sector-year
Before plotting, the pooled values are checked against Table 3 of the manuscript.

Input : {L2P}/R3_SIT_snow_swap_all_winters_v7.csv   (overwritten in place by the v8 rerun)
Output: {OUT}/R3_Fig_C2_03_sector_year_heatmap.{pdf,png}  (same name, so \\includegraphics is unchanged;
        any existing file is first copied to *_pre_v11.*)
        {OUT}/R3_Fig_C2_03_sector_year_heatmap_values.csv  (all plotted numbers, for Zenodo)
Run   : login node, a few seconds (see bottom of file).
"""
import os, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize

L2P = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation'
SWAP = os.environ.get('SWAP_CSV', f'{L2P}/R3_SIT_snow_swap_all_winters_v7.csv')
OUT = os.environ.get('FIG_DIR', f'{L2P}/figures_v7')
NAME = 'R3_Fig_C2_03_sector_year_heatmap'

SECT = ['Western_Weddell', 'Eastern_Weddell', 'Ross']
SLAB = ['Western\nWeddell', 'Eastern\nWeddell', 'Ross']
PROD = [('CCI', 'CCI', 'cci_snow_depth', 'h_i_Imp_CCI', 'h_i_Base_CCI', [2022, 2023]),
        ('CT', 'Cryo-TEMPO', 'ct_snow_depth', 'h_i_Imp_CT', 'h_i_Base_CT', [2022, 2023, 2024, 2025])]
# Table 3 (pooled-winter) values: excess %, median dh_i m
TABLE3 = {'CCI': {'Western_Weddell': (23, 0.308), 'Eastern_Weddell': (36, 0.392), 'Ross': (42, 0.319)},
          'CT':  {'Western_Weddell': (51, 0.766), 'Eastern_Weddell': (44, 0.406), 'Ross': (43, 0.327)}}

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'axes.labelsize': 11,
                     'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})

# ---------------------------------------------------------------- data
df = pd.read_csv(SWAP)
df['year'] = (df['year_month'] // 100).astype(int)
print(f'Loaded {len(df):,} rows from {SWAP}')

rows, M = [], {}
for key, lab, l2, imp, base, years in PROD:
    ok = df[imp].notna() & df[base].notna()
    ex = np.full((len(SECT), len(years)), np.nan)
    dh = np.full_like(ex, np.nan)
    for i, s in enumerate(SECT):
        for j, y in enumerate(years + ['pooled']):
            m = ok & (df['sector'] == s)
            if y != 'pooled':
                m &= df['year'] == y
            n = int(m.sum())
            if n == 0:
                continue
            c2i, l2s = df.loc[m, 'snow_thickness'].median(), df.loc[m, l2].median()
            e = 100 * (l2s - c2i) / c2i
            d = (df.loc[m, imp] - df.loc[m, base]).median()
            rows.append(dict(product=lab, sector=s, year=y, n=n, c2i_snow_median_m=round(c2i, 4),
                             l2_snow_median_m=round(l2s, 4), excess_pct=round(e, 2), dh_i_median_m=round(d, 4)))
            if y == 'pooled':
                te, td = TABLE3[key][s]
                flag = '' if (abs(round(e) - te) <= 1 and abs(d - td) <= 0.002) else '   <-- CHECK vs Table 3'
                print(f'  pooled {lab:10s} {s:16s} excess {e:+5.1f}% (Table 3 {te:+d}%)  '
                      f'dh_i {d:+.3f} m (Table 3 {td:+.3f} m){flag}')
            else:
                ex[i, j], dh[i, j] = e, d
    M[key] = (lab, years, ex, dh)

tab = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
tab.to_csv(os.path.join(OUT, f'{NAME}_values.csv'), index=False)
print('\nValues plotted (excess % | median dh_i m):')
print(tab[tab.year != 'pooled'].pivot_table(index=['product', 'sector'], columns='year',
      values=['excess_pct', 'dh_i_median_m']).round(2).to_string())

# ---------------------------------------------------------------- colour scales
all_ex = np.concatenate([M[k][2][np.isfinite(M[k][2])] for k in M])
all_dh = np.concatenate([M[k][3][np.isfinite(M[k][3])] for k in M])
ex_max = 25 * np.ceil(all_ex.max() / 25)                       # 98 % -> 100 %
dh_max = 0.2 * np.ceil(all_dh.max() / 0.2)                     # 1.09 m -> 1.2 m
if all_dh.min() < 0:                                            # diverging only if a cell is negative
    dh_norm, dh_cmap = TwoSlopeNorm(vmin=min(all_dh.min(), -0.1), vcenter=0, vmax=dh_max), 'RdBu_r'
else:
    dh_norm, dh_cmap = Normalize(vmin=0, vmax=dh_max), 'Blues'
ex_norm, ex_cmap = Normalize(vmin=0, vmax=ex_max), 'YlOrRd'

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 2, figsize=(13, 6.6),
                         gridspec_kw=dict(width_ratios=[2, 4], hspace=0.28, wspace=0.06))

def draw(ax, mat, years, norm, cmap, fmt, show_y):
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=norm, aspect='auto')
    rgba = plt.get_cmap(cmap)(norm(np.nan_to_num(mat)))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                lum = 0.299 * rgba[i, j, 0] + 0.587 * rgba[i, j, 1] + 0.114 * rgba[i, j, 2]
                ax.text(j, i, fmt(mat[i, j]), ha='center', va='center', fontsize=11,
                        fontweight='bold', color='white' if lum < 0.5 else 'black')
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years])
    ax.set_yticks(range(len(SECT)))
    ax.set_yticklabels(SLAB if show_y else [])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(years)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SECT)), minor=True)
    ax.grid(which='minor', color='white', lw=2)
    ax.tick_params(which='minor', length=0)
    return im

letters = iter('abcd')
for r, (kind, norm, cmap, fmt) in enumerate([
        ('ex', ex_norm, ex_cmap, lambda v: f'{v:+.0f}%'),
        ('dh', dh_norm, dh_cmap, lambda v: f'{v:+.2f} m')]):
    for c, key in enumerate(['CCI', 'CT']):
        lab, years, ex, dh = M[key]
        ax = axes[r, c]
        im = draw(ax, ex if kind == 'ex' else dh, years, norm, cmap, fmt, show_y=(c == 0))
        ax.set_title(f'({next(letters)}) {lab} ({years[0]}–{years[-1]})', loc='left',
                     fontweight='bold', fontsize=11.5)
    cb = fig.colorbar(im, ax=axes[r, :].tolist(), shrink=0.92, pad=0.015, fraction=0.03)
    cb.set_label('AMSR-E/AMSR2 excess\nover CRYO2ICE (%)' if kind == 'ex'
                 else '$\\Delta h_i$ (m)\n(positive: thicker with CRYO2ICE)', fontsize=10)
    axes[r, 0].annotate('Snow' if kind == 'ex' else 'Sea ice', xy=(-0.42, 0.5), xycoords='axes fraction',
                        rotation=90, ha='center', va='center', fontsize=12.5, fontweight='bold')

for ext in ('pdf', 'png'):
    f = os.path.join(OUT, f'{NAME}.{ext}')
    if os.path.exists(f) and not os.path.exists(f.replace(f'.{ext}', f'_pre_v11.{ext}')):
        shutil.copy2(f, f.replace(f'.{ext}', f'_pre_v11.{ext}'))
    fig.savefig(f)
print(f'\nSaved {OUT}/{NAME}.pdf/.png  (+ _values.csv)')

# ---------------------------------------------------------------- how to run on NCI Gadi
# ssh xl1657@gadi.nci.org.au
# cd /g/data/gv90/xl1657/phd/M3_workspace/code
# module load python3/3.9.2
# source ~/cryo2ice_env/bin/activate
# python R3_Step10c_heatmap_v11.py

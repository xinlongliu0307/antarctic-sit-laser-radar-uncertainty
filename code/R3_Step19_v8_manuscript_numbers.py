#!/usr/bin/env python3
"""
R3_Step19_v8_manuscript_numbers.py — every number the v8 snow fix changes, keyed for Ctrl+F
===========================================================================================
Run AFTER the v8 reruns (patch -> Step 09 -> Step 14 -> Step 10 v9 -> Step 10b).
Prints   @@KEY@@ = value   for every placeholder in MS3_v8_snowfix_plan_20260927.md,
writes them to ../logs/step19_keys.txt, and runs consistency checks (lines starting
'WARNING' mean a sentence of the plan may not hold: send me the output).

Optional: python R3_Step19_v8_manuscript_numbers.py --tex manuscript3_V2_clean.tex
          fills every @@KEY@@ in that file and writes manuscript3_V2_clean_filled.tex
          (then reports any token left unfilled). This is the same as doing the
          Ctrl+F Replace All for each key by hand.
"""
import os, sys, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

L2P = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation'
SWAP = f'{L2P}/R3_SIT_snow_swap_all_winters_v7.csv'
SWEEP = f'{L2P}/R3_snow_swap_delta_sweep_v7.csv'
GRID = f'{M3_OUTPUT}/R3_amsr2_gridscale_v7.csv'
CELLS = f'{M3_OUTPUT}/R3_amsr2_gridscale_cells_v7.csv'
PKL = f'{M3_OUTPUT}/R3_input_d070_icetype.pkl'
ETA = (1 + 0.51 * RHO_S_CENTRAL / 1000.0) ** 1.5
SECT = ['Western_Weddell', 'Eastern_Weddell', 'Ross']
LAB = {'Western_Weddell': 'Western Weddell', 'Eastern_Weddell': 'Eastern Weddell', 'Ross': 'Ross'}
K = {}
WARN = []


def f(x, d=3, sign=False):
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def rng_str(a, b, d=2):
    lo, hi = sorted([round(a, d), round(b, d)])
    return f"{lo:.{d}f}" if lo == hi else f"{lo:.{d}f}--{hi:.{d}f}"


# ── 0. AU_SI12 flag audit (raw pickle values) ─────────────────────────────
print('=' * 90)
print('0. AU_SI12 flag audit (pickle, before masking)')
p = pd.read_pickle(PKL)
if 'subsector' not in p.columns:
    p['subsector'] = assign_subsector_vec(p['lon'].values)
a = p['AMSR2_snow_depth'].astype(float).values
codes = np.round(a * 100)
for c in [110, 120, 130, 140, 150, 160]:
    n = int(np.sum(codes == c))
    if n:
        print(f'   {c} cm flag stored as {c / 100:.2f} m: {n:,} matchups')
print(f'   values > 1.00 m: {int(np.sum(a > 1.0)):,} of {np.isfinite(a).sum():,}')
for r in ['Weddell_W', 'Weddell_E', 'Ross']:
    m = get_region_mask(p['sector'].values, p['subsector'].values, r)
    fl = np.isin(codes[m], [110, 120, 130, 140, 150, 160])
    ok = np.isfinite(a[m]) & (a[m] > 0) & ~fl
    print(f'   {r:10s} flagged {100 * fl.mean():5.1f}%   AMSR2 median after masking {np.median(a[m][ok]):.3f} m, mean {a[m][ok].mean():.3f} m')

# ── 1. Snow-replacement numbers ───────────────────────────────────────────
print('=' * 90)
print('1. Snow replacement (Step 09 v8 output)')
s = pd.read_csv(SWAP, low_memory=False)
s['year'] = (s['year_month'] // 100).astype(int)
dfb = s['is2_fb_wm'] - s['cs2_rfb']
ok = np.abs(s['snow_thickness']) > 0.02
imp = float(np.median(dfb[ok] / (ETA * s.loc[ok, 'snow_thickness'])))
print(f'   implied delta of snow_thickness now: {imp:.3f}')
if abs(imp - 0.70) > 0.01:
    WARN.append(f'implied delta is {imp:.3f}, not 0.70: Step 09 patch did not take effect')
wed = s['sector'].isin(['Western_Weddell', 'Eastern_Weddell'])
pos = s['snow_thickness'] > 0
print(f'   ESS-style check (h_s > 0, delta 0.70): Weddell median {s.loc[wed & pos, "snow_thickness"].median():.3f} '
      f'(ESS 0.188), Ross {s.loc[(s.sector == "Ross") & pos, "snow_thickness"].median():.3f} (ESS 0.155)')
print(f'   negative C2I fraction: {100 * (s["snow_thickness"] <= 0).mean():.1f}%')

rows_tex, xs_all = {}, {}
for tag, snow, base, impc, dcol in [('CCI', 'cci_snow_depth', 'h_i_Base_CCI', 'h_i_Imp_CCI', 'Delta_h_i_CCI'),
                                    ('CT', 'ct_snow_depth', 'h_i_Base_CT', 'h_i_Imp_CT', 'Delta_h_i_CT')]:
    m = s[base].notna() & s[impc].notna()
    d = s.loc[m, impc] - s.loc[m, base]
    K[f'C2I_MED_{tag}'] = f(s.loc[m, 'snow_thickness'].median())
    K[f'L2_MED_{tag}'] = f(s.loc[m, snow].median())
    K[f'HB_{tag}'] = f(s.loc[m, base].median())
    K[f'HI_{tag}'] = f(s.loc[m, impc].median())
    K[f'DHI_{tag}'] = f(d.median(), sign=True)
    K[f'DHI_{tag}_RAW'] = d.median()
    K[f'DMED_{tag}'] = f(s.loc[m, impc].median() - s.loc[m, base].median())
    b = s.loc[m, base].values
    K[f'REL_{tag}'] = f(np.nanmedian(100 * d.values / np.where(b != 0, b, np.nan)), 1)
    for sec in SECT:
        ms = m & (s['sector'] == sec)
        am, cm = s.loc[ms, snow].median(), s.loc[ms, 'snow_thickness'].median()
        hb, hi = s.loc[ms, base].median(), s.loc[ms, impc].median()
        dd = (s.loc[ms, impc] - s.loc[ms, base]).median()
        xs = 100 * (am - cm) / cm
        xs_all[(tag, sec)] = xs
        K[f'DHI_{tag}_{sec}'] = dd
        K[f'SNOWDIFF_{tag}_{sec}'] = am - cm
        rows_tex[f'T3_{tag}_{sec}'] = (f"{LAB[sec]} & {int(ms.sum()):,} & {am:.3f} & {cm:.3f} & {xs:+.0f} & "
                                       f"{hb:.3f} & {hi:.3f} & {hi - hb:+.3f} & {dd:+.3f} \\\\")
    # Western Weddell per-year (heatmap numbers)
    yrs = []
    for y in sorted(s['year'].unique()):
        my = m & (s['sector'] == 'Western_Weddell') & (s['year'] == y)
        if my.sum():
            yrs.append((y, (s.loc[my, impc] - s.loc[my, base]).median(),
                        s.loc[my, snow].median(), s.loc[my, 'snow_thickness'].median()))
    print(f'   {tag} Western Weddell by year (dh_i, L2 snow, C2I snow):',
          ', '.join(f'{y}: {dd:+.3f} {l2:.3f} {c2:.3f}' for y, dd, l2, c2 in yrs))
    if tag == 'CT':
        K['WWYR_MIN'] = f(min(v[1] for v in yrs), sign=True)
        K['WWYR_MAX'] = f(max(v[1] for v in yrs), sign=True)
        y25 = [v for v in yrs if v[0] == 2025]
        if y25:
            K['CT_WW25_L2'] = f(y25[0][2], 2)
            K['CT_WW25_C2I'] = f(y25[0][3], 2)

K.update(rows_tex)
K['DHI_RANGE2'] = rng_str(K.pop('DHI_CCI_RAW'), K.pop('DHI_CT_RAW'), 2)
lo, hi = [float(x) for x in (K['DHI_RANGE2'].split('--') * 2)[:2]]
K['DHI_RANGE1'] = rng_str(lo, hi, 1)
K['XS_MIN'] = f"{min(xs_all.values()):.0f}"
K['XS_MAX'] = f"{max(xs_all.values()):.0f}"
who = max(xs_all, key=xs_all.get)
print(f'   largest climatological excess: {who} ({xs_all[who]:.0f}%)')
if who != ('CT', 'Western_Weddell'):
    WARN.append(f'largest excess is {who}, not Cryo-TEMPO Western Weddell: Section 4.1 paragraph 3 needs rewording')
for tag in ['CCI', 'CT']:
    ww = K[f'DHI_{tag}_Western_Weddell']
    if ww < max(K[f'DHI_{tag}_Eastern_Weddell'], K[f'DHI_{tag}_Ross']):
        WARN.append(f'{tag}: Western Weddell is NOT the largest median dh_i: Section 4.5 regional sentence needs rewording')
K['WW_DHI_CT'] = f(K['DHI_CT_Western_Weddell'], sign=True)
K['WW_DHI_CCI'] = f(K['DHI_CCI_Western_Weddell'], sign=True)
K['WW_SNOWDIFF_CT'] = f(K['SNOWDIFF_CT_Western_Weddell'])
K['OTHER_SNOWDIFF_CT'] = f(np.mean([K['SNOWDIFF_CT_Eastern_Weddell'], K['SNOWDIFF_CT_Ross']]), 2)
for k in [k for k in K if k.startswith(('DHI_CCI_', 'DHI_CT_', 'SNOWDIFF_'))]:
    K.pop(k)

# ── 2. delta sweep ────────────────────────────────────────────────────────
print('=' * 90)
print('2. Penetration-fraction sweep of the replacement (all sectors and by sector)')
sw = pd.read_csv(SWEEP)


def zero_cross(g):
    g = g.sort_values('delta')
    x, y = g['delta'].values, g['median_dh_i'].values
    for i in range(len(y) - 1):
        if y[i] <= 0 < y[i + 1] or y[i] >= 0 > y[i + 1]:
            return x[i] + (0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    return np.nan


zcs = []
for tag in ['CCI', 'CT']:
    for sec in ['All'] + SECT:
        g = sw[(sw['product'] == tag) & (sw['sector'] == sec)]
        zc = zero_cross(g)
        v = g.set_index(g['delta'].round(2))['median_dh_i']
        print(f'   {tag:3s} {sec:16s} dh_i at 0.40 {v[0.40]:+.3f}  0.55 {v[0.55]:+.3f}  0.70 {v[0.70]:+.3f}  '
              f'1.00 {v[1.00]:+.3f}   zero crossing {zc:.2f}' if np.isfinite(zc) else
              f'   {tag:3s} {sec:16s} dh_i at 0.40 {v[0.40]:+.3f}  0.55 {v[0.55]:+.3f}  0.70 {v[0.70]:+.3f}  '
              f'1.00 {v[1.00]:+.3f}   no zero crossing in 0.40-1.00')
        if sec == 'All':
            for dv, key in [(0.40, 'D040'), (0.55, 'D055'), (1.00, 'D100')]:
                K[f'{key}_{tag}'] = f(v[dv], 2, sign=True)
            K[f'ZC_{tag}'] = f(zc, 2) if np.isfinite(zc) else 'NONE'
            zcs.append(zc)
            if abs(v[0.70] - float(K[f'DHI_{tag}'])) > 0.02:
                WARN.append(f'{tag}: sweep at 0.70 ({v[0.70]:+.3f}) differs from per-matchup median {K[f"DHI_{tag}"]}')
if not all(np.isfinite(zcs)):
    WARN.append('no pooled zero crossing for at least one product: the new sign-change sentence in Section 4.5 does not hold')
elif not all(0.45 <= z <= 0.65 for z in zcs):
    WARN.append(f'pooled zero crossings {zcs} are not near the empirical optimum 0.55-0.58: reword "close to the empirical optimum"')
K['ZC_MIN'] = f(np.nanmin(zcs), 2) if np.isfinite(np.nanmin(zcs)) else 'NONE'

# ── 3. Grid scale (Step 14 v8 output) ─────────────────────────────────────
print('=' * 90)
print('3. Grid-scale comparison (Step 14 after flag masking)')
g = pd.read_csv(GRID).set_index(['scale', 'region'])
K['R2_RET'] = f(g.loc[('retrieval', 'All'), 'r2'])
K['R2_125'] = f(g.loc[('12.5km', 'All'), 'r2'])
K['R2_25'] = f(g.loc[('25km', 'All'), 'r2'])
K['NCD_125'] = f"{int(g.loc[('12.5km', 'All'), 'n']):,}".replace(',', '{,}')
K['NCD_25'] = f"{int(g.loc[('25km', 'All'), 'n']):,}".replace(',', '{,}')
wed = g[g.index.get_level_values('region').isin(['Weddell', 'Weddell_W', 'Weddell_E'])]
K['R2MAX_WED'] = f(wed['r2'].max())
K['R2MAX_ROSS'] = f(g[g.index.get_level_values('region') == 'Ross']['r2'].max())
b = g[g.index.get_level_values('region') == 'All']['bias']
K['BIAS_RANGE'] = rng_str(b.min(), b.max(), 2)
cc, ca = [], []
for sc in ['retrieval', '12.5km', '25km']:
    cc.append(g.loc[(sc, 'Weddell'), 'mean_c2i'] - g.loc[(sc, 'Ross'), 'mean_c2i'])
    ca.append(g.loc[(sc, 'Weddell'), 'mean_amsr2'] - g.loc[(sc, 'Ross'), 'mean_amsr2'])
    print(f'   {sc:10s} r2 {g.loc[(sc, "All"), "r2"]:.4f}  p {g.loc[(sc, "All"), "pearson_p"]:.1e}  '
          f'bias {g.loc[(sc, "All"), "bias"]:.3f}  contrast C2I {cc[-1]:+.3f} AMSR2 {ca[-1]:+.3f}')
K['CON_C2I'] = rng_str(min(cc), max(cc), 2)
K['CON_AM'] = rng_str(min(ca), max(ca), 2)
if min(ca) <= 0 or abs(np.mean(cc) - np.mean(ca)) > 0.02:
    WARN.append('the Weddell-Ross contrast is no longer similar in C2I and AMSR2: "Only the sector contrast is shared" needs rewording')
cells = pd.read_csv(CELLS)
K['CELLMED_125'] = f"{cells.loc[cells.scale == '12.5km', 'n'].median():.0f}"
K['CELLMED_25'] = f"{cells.loc[cells.scale == '25km', 'n'].median():.0f}"

# ── 4. Output ─────────────────────────────────────────────────────────────
print('=' * 90)
print('4. KEYS (Ctrl+F each @@KEY@@, Replace All with the value)')
lines = [f'@@{k}@@ = {v}' for k, v in K.items()]
print('\n'.join(lines))
os.makedirs('../logs', exist_ok=True)
open('../logs/step19_keys.txt', 'w').write('\n'.join(lines) + '\n')
print('=' * 90)
print('\n'.join('WARNING: ' + w for w in WARN) if WARN else 'All consistency checks passed: every plan sentence holds.')

if '--tex' in sys.argv:
    src = sys.argv[sys.argv.index('--tex') + 1]
    t = open(src).read()
    for k, v in K.items():
        t = t.replace(f'@@{k}@@', v)
    out = src.replace('.tex', '_filled.tex')
    open(out, 'w').write(t)
    left = sorted(set(re.findall(r'@@[A-Z0-9_a-z]+@@', t)))
    print(f'\nWrote {out}; unfilled tokens: {left if left else "none"}')

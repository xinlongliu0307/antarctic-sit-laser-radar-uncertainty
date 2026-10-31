#!/usr/bin/env python3
"""
R3_Step18_c2i_snow_consistency.py — why is the MS3 C2I snow thinner than in Liu et al. (2026)?
==============================================================================================
Alex, comments 13 and 22 (27 Sep 2026): MS3 reports C2I median snow of 0.105-0.107 m in the
snow-replacement experiment, whereas Liu et al. (2026, ESS) report 0.188 m (Weddell) and
0.155 m (Ross) medians at the same delta = 0.70 and rho_s = 330 kg m-3.

Arithmetic from the ESS paper: median freeboard difference over all 82,341 matchups = 0.132 m.
  delta = 0.70  ->  median h_s = 0.132 / (0.70 * 1.263) = 0.149 m
  delta = 1.00  ->  median h_s = 0.132 / 1.263          = 0.105 m   <-- the MS3 value
So either (A) the difference is only a matter of which matchups are summarised, or
(B) the C2I snow column used in Step 09 carries no 0.70 penetration factor.

This script decides between A and B. Read-only; < 2 min on a login node.
  1. Paper-2 dataset (M2_CSV): medians over all / h_s > 0 / AMSR2-matched, by sector
  2. Step 09 input and output: implied delta = (is2 - cs2) / (eta * snow_thickness)
  3. Step 14 input (pickle): implied delta of hs_d070; AMSR2 daily snow statistics
  4. Pearson r with p-values (nominal n and n_eff = 0.44 n), pooled and within sector,
     for the level-2 comparison (Alex comment 14)
  5. Grid-scale r, r^2 and p-values from R3_amsr2_gridscale_v7.csv

Run: module load python3/3.9.2; source ~/cryo2ice_env/bin/activate
     cd /g/data/gv90/xl1657/phd/M3_workspace/code && python R3_Step18_c2i_snow_consistency.py | tee ../logs/step18.txt
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

pd.set_option('display.width', 220)
ETA = (1 + 0.51 * RHO_S_CENTRAL / 1000.0) ** 1.5          # 1.263
L2P = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation'
COLL_CSV = f'{L2P}/R3_L2P_collocation_all_winters.csv'
SWAP_CSV = f'{L2P}/R3_SIT_snow_swap_all_winters_v7.csv'
PKL = f'{M3_OUTPUT}/R3_input_d070_icetype.pkl'
GRID_CSV = f'{M3_OUTPUT}/R3_amsr2_gridscale_v7.csv'
NEFF_FRAC = 0.44                                           # n_eff / n, Liu et al. (2026) Table 1


def d(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 'n=0'
    p = x[x > 0]
    return (f"n={len(x):6,d} mean={x.mean():.3f} median={np.median(x):.3f} neg={100 * np.mean(x <= 0):4.1f}%  "
            f"| h>0: n={len(p):6,d} mean={p.mean():.3f} median={np.median(p):.3f}")


def implied_delta(is2, cs2, hs):
    is2, cs2, hs = map(lambda a: np.asarray(a, float), (is2, cs2, hs))
    ok = np.isfinite(is2) & np.isfinite(cs2) & np.isfinite(hs) & (np.abs(hs) > 0.02)
    r = (is2[ok] - cs2[ok]) / (ETA * hs[ok])
    return f"implied delta: median {np.median(r):.3f}, IQR {np.percentile(r, 25):.3f}-{np.percentile(r, 75):.3f} (n={ok.sum():,})"


def pearson_block(x, y, label):
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 10:
        return f"  {label:24s} n={n}"
    r, p = stats.pearsonr(x[ok], y[ok])
    neff = max(int(NEFF_FRAC * n), 3)
    t = r * np.sqrt((neff - 2) / max(1 - r ** 2, 1e-12))
    p_eff = 2 * stats.t.sf(abs(t), neff - 2)
    rs, ps = stats.spearmanr(x[ok], y[ok])
    return (f"  {label:24s} n={n:6,d}  r={r:+.3f}  r2={r * r:.4f}  p={p:.1e}  "
            f"p(n_eff={neff:,})={p_eff:.1e}  spearman={rs:+.3f}")


print("=" * 100)
print("0. Liu et al. (2026) ESS reference values (delta 0.70, rho_s 330)")
print("   Table 2 (AMSR2-matched, n = 40,431 / 28,510): C2I mean (median) Weddell 0.256 (0.188), Ross 0.216 (0.155)")
print("   AMSR2 daily mean (median): Weddell 0.252 (0.200), Ross 0.195 (0.170); r = 0.250 / 0.027")
print("   Section 4.1: 82,341 matchups, 88-89% with h_s > 0; freeboard difference median 0.132 m, 11.5% < 0")
print(f"   => all-matchup median h_s at delta 0.70 = {0.132 / (0.70 * ETA):.3f} m; at delta 1.00 = {0.132 / ETA:.3f} m")

print("=" * 100)
print("1. Paper-2 dataset (M2_CSV)")
print("=" * 100)
if os.path.exists(M2_CSV):
    m2 = pd.read_csv(M2_CSV, low_memory=False)
    print("   columns:", list(m2.columns))
    sc = next((c for c in ['snow_thickness', 'hs', 'h_s', 'snow_depth'] if c in m2.columns), None)
    sec = next((c for c in ['sector', 'region'] if c in m2.columns), None)
    am = next((c for c in ['AMSR2_snow_depth', 'amsr2_snow_depth', 'amsr2'] if c in m2.columns), None)
    if sc:
        print(f"   {sc} all:", d(m2[sc]))
        if am:
            mm = m2[am].notna() & (m2[am] > 0)
            print(f"   {sc} AMSR2-matched:", d(m2.loc[mm, sc]))
        if sec:
            for s_, g in m2.groupby(sec):
                print(f"   {sc} {str(s_):16s}", d(g[sc]))
                if am:
                    gm = g[g[am].notna() & (g[am] > 0)]
                    print(f"      {'AMSR2-matched':16s}", d(gm[sc]).split('|')[0], "| AMSR2", d(gm[am]).split('|')[0])
        for a, b in [('is2_fb_wm', 'cs2_rfb'), ('is2_fb', 'cs2_fb'), ('is2_freeboard', 'cs2_radar_freeboard')]:
            if a in m2.columns and b in m2.columns:
                print(f"   ({a} - {b}) median {np.nanmedian(m2[a] - m2[b]):.3f} m;", implied_delta(m2[a], m2[b], m2[sc]))
else:
    print("   M2_CSV not found:", M2_CSV)

for tag, path in [('2. Step 09 INPUT (collocation CSV)', COLL_CSV), ('2b. Step 09 OUTPUT (snow-swap CSV)', SWAP_CSV)]:
    print("=" * 100)
    print(tag)
    print("=" * 100)
    if not os.path.exists(path):
        print("   not found:", path)
        continue
    s = pd.read_csv(path, low_memory=False)
    print(f"   rows {len(s):,}; columns with 'snow' or 'fb':", [c for c in s.columns if 'snow' in c or 'fb' in c])
    df_ = s['is2_fb_wm'] - s['cs2_rfb']
    print("   freeboard difference :", d(df_).split('|')[0])
    print("   snow_thickness       :", d(s['snow_thickness']))
    print("   Dfb/(0.70 eta)       :", d(df_ / (0.70 * ETA)))
    print("   Dfb/eta (delta 1.00) :", d(df_ / ETA))
    print("  ", implied_delta(s['is2_fb_wm'], s['cs2_rfb'], s['snow_thickness']))
    for sec, g in s.groupby('sector'):
        print(f"   {sec:16s} snow_thickness", d(g['snow_thickness']))
    if 'cci_snow_depth' in s.columns:
        for prod, col in [('CCI', 'cci_snow_depth'), ('CT', 'ct_snow_depth')]:
            m = s[col].notna() & s['snow_thickness'].notna()
            print(f"   {prod} subset: C2I", d(s.loc[m, 'snow_thickness']).split('|')[0], "| L2 snow", d(s.loc[m, col]).split('|')[0])

print("=" * 100)
print("3. Monte Carlo / grid-scale input (pickle): positive-freeboard matchups")
print("=" * 100)
if os.path.exists(PKL):
    p = pd.read_pickle(PKL)
    if 'subsector' not in p.columns:
        p['subsector'] = assign_subsector_vec(p['lon'].values)
    hs_rt = (p['is2_fb_wm'] - p['cs2_rfb']) / (DELTA_BASELINE * ETA)
    print(f"   rows {len(p):,}")
    print("   runtime h_s (MC)   :", d(hs_rt))
    if 'hs_d070' in p.columns:
        print("   hs_d070 (Step 14)  :", d(p['hs_d070']))
        print("  ", implied_delta(p['is2_fb_wm'], p['cs2_rfb'], p['hs_d070']))
    if 'snow_thickness' in p.columns:
        print("   snow_thickness     :", d(p['snow_thickness']))
    if 'AMSR2_snow_depth' in p.columns:
        a = p['AMSR2_snow_depth'].astype(float)
        ok = a.notna() & (a > 0) & (a <= 1.5)
        print("   AMSR2 daily (valid):", d(a[ok]).split('|')[0])
        for r in ['Weddell', 'Ross']:
            m = get_region_mask(p['sector'].values, p['subsector'].values, r) & ok.values
            print(f"   {r:8s} AMSR2 {d(a[m]).split('|')[0]}  | hs_d070 {d(p.loc[m, 'hs_d070']).split('|')[0] if 'hs_d070' in p.columns else ''}")
else:
    print("   not found:", PKL)

print("=" * 100)
print("4. Level-2 comparison: Pearson r with p-values (Alex comment 14)")
print("=" * 100)
if os.path.exists(SWAP_CSV):
    s = pd.read_csv(SWAP_CSV, low_memory=False)
    for prod, col in [('CCI', 'cci_snow_depth'), ('Cryo-TEMPO', 'ct_snow_depth')]:
        print(f" {prod}")
        print(pearson_block(s['snow_thickness'].values, s[col].values, 'pooled'))
        for sec, g in s.groupby('sector'):
            print(pearson_block(g['snow_thickness'].values, g[col].values, sec))

print("=" * 100)
print("5. Grid-scale statistics (R3_amsr2_gridscale_v7.csv)")
print("=" * 100)
if os.path.exists(GRID_CSV):
    g = pd.read_csv(GRID_CSV)
    cols = [c for c in ['scale', 'region', 'n', 'median_c2i', 'median_amsr2', 'mean_c2i', 'mean_amsr2', 'bias',
                        'pearson_r', 'r2', 'pearson_p', 'spearman_rho'] if c in g.columns]
    print(g[cols].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
else:
    print("   not found:", GRID_CSV)

print("\nDecision rule: implied delta ~0.70 in Section 2 -> case A (subset only, no rerun).")
print("               implied delta ~1.00 in Section 2 -> case B (rerun Step 09 with the C2I snow recomputed at 0.70).")

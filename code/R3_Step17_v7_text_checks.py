#!/usr/bin/env python3
"""
R3_Step17_v7_text_checks.py — numbers needed to close the 26 Sep 2026 proofreading pass
========================================================================================
Read-only; runs in < 1 min on a login node. Prints:

  1. LaTeX rows for Table 4 (tab:subsector_stats) from R3_ExpA_d070_v7.npz.
     The table in manuscript3_V2_clean.tex still carries pre-v7 numbers
     (W. Weddell 2.621 m) that disagree with the text (2.86 m).
  2. Root-mean-square ratio of central-value sea-ice thickness to snow depth per
     stratum, to check the first-order crossover argument proposed for Section 5.2
     (density and penetration terms equal where h_i/h_s ~ (rho_w-rho_s) sigma_delta / (delta sigma_rho)).
  3. Snow-replacement median Delta h_i by sector at delta = 0.40 (is the sign
     positive in every sector, as the text claims?).
  4. Skew-normal minus Gaussian changes for rho_i AND delta by stratum (does the
     rho_i/delta order survive the shape change in every stratum?).
  5. Cryo-TEMPO match count, to fix the "100.0%" match rate in Section 2.3.

Run: source ~/cryo2ice_env/bin/activate && python R3_Step17_v7_text_checks.py
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

pd.set_option('display.width', 200)
LABEL = {'Weddell': 'Weddell', 'Weddell_W': 'Western Weddell', 'Weddell_E': 'Eastern Weddell',
         'Ross': 'Ross', 'MYI_obs': 'Observed MYI', 'FYI_obs': 'Observed FYI'}

# ── 1. Table 4 from Experiment A v7 ──────────────────────────────────────
print("=" * 78)
print("1. Table 4 (tab:subsector_stats) rows from R3_ExpA_d070_v7.npz")
print("   Mean/Median: of per-matchup MC means; sigma: mean per-matchup MC std;")
print("   90% interval: mean of per-matchup 5th and 95th percentiles")
print("=" * 78)
A = np.load(f'{M3_OUTPUT}/R3_ExpA_d070_v7.npz', allow_pickle=True)
sectors, subsectors, icetype = A['sectors'], A['subsectors'], A['icetype']
for r in REGIONS_V6:
    m = get_region_mask_v6(sectors, subsectors, icetype, r)
    n = int(m.sum())
    print(f"{LABEL[r]} &\n{n:,} &\n{n * N_MC:,} &\n{A['hi_mean'][m].mean():.3f} &\n"
          f"{np.median(A['hi_mean'][m]):.3f} &\n{A['hi_std'][m].mean():.3f} &\n"
          f"[{A['hi_p05'][m].mean():.3f}, {A['hi_p95'][m].mean():.3f}] \\\\\n\\hline\n")

# ── 2. thickness-to-snow ratio ───────────────────────────────────────────
print("=" * 78)
print("2. Central-value h_i / h_s (delta 0.70, rho_s 330, mixture-mean rho_i)")
print("=" * 78)
df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_icetype.pkl')
if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df['lon'].values)
eta = (1 + 0.51 * RHO_S_CENTRAL / 1000.0) ** 1.5
hs = (df['is2_fb_wm'].values - df['cs2_rfb'].values) / (DELTA_BASELINE * eta)
hi = compute_ice_thickness(df['is2_fb_wm'].values, hs, RHO_S_CENTRAL,
                           mixture_rho_i_central(df['myi_frac'].values), RHO_W_CENTRAL)
# predicted crossover ratio with realised prior moments (delta 0.696 / 0.184; rho_i ~28-29)
pred = (RHO_W_CENTRAL - RHO_S_CENTRAL) * 0.184 / (0.696 * 28.5)
print(f"   predicted equality at h_i/h_s ~ {pred:.1f}")
for r in ['all'] + REGIONS_V6:
    m = np.ones(len(df), bool) if r == 'all' else get_region_mask_v6(
        df['sector'].values, df['subsector'].values, df['icetype'].values, r)
    rms = np.sqrt(np.mean(hi[m] ** 2) / np.mean(hs[m] ** 2))
    print(f"   {r:10s} n={m.sum():6,d}  mean h_s={hs[m].mean():.3f} m  mean h_i={hi[m].mean():.2f} m  "
          f"rms ratio={rms:5.2f}  median ratio={np.median(hi[m] / hs[m]):5.2f}")

# ── 3. snow replacement at delta = 0.40 ──────────────────────────────────
print("=" * 78)
print("3. Snow replacement, median Delta h_i by sector at delta = 0.40 and 1.00")
print("=" * 78)
hits = glob.glob(f'{M3_OUTPUT}/**/R3_snow_swap_delta_sweep_v7.csv', recursive=True)
if hits:
    s = pd.read_csv(hits[0])
    s = s[np.isclose(s['delta'], 0.40) | np.isclose(s['delta'], 1.00)]
    print(s[['delta', 'product', 'sector', 'n', 'median_hs_c2i', 'median_dh_i']].to_string(index=False))
else:
    print("   R3_snow_swap_delta_sweep_v7.csv not found under", M3_OUTPUT)

# ── 4. shape robustness: rho_i and delta changes, order preserved? ──────
print("=" * 78)
print("4. Skew-normal vs Gaussian: rho_i and delta shares and order by stratum")
print("=" * 78)
p = f'{M3_OUTPUT}/R3_shape_robustness_v7.json'
if os.path.exists(p):
    J = json.load(open(p))['results']
    for reg in J['gaussian']:
        g, s_ = J['gaussian'][reg], J['skewnorm'][reg]
        og = 'rho_i' if g['frac_rho_i'] > g['frac_delta'] else 'delta'
        os_ = 'rho_i' if s_['frac_rho_i'] > s_['frac_delta'] else 'delta'
        print(f"   {reg:10s} rho_i {g['frac_rho_i']:5.1f}->{s_['frac_rho_i']:5.1f}  "
              f"delta {g['frac_delta']:5.1f}->{s_['frac_delta']:5.1f}  lead {og}->{os_}"
              f"{'' if og == os_ else '   *** ORDER CHANGES ***'}")
else:
    print("   ", p, "not found")

# ── 5. Cryo-TEMPO match count ────────────────────────────────────────────
print("=" * 78)
print("5. Snow-swap collocation counts (for the Section 2.3 match rates)")
print("=" * 78)
hits = glob.glob(f'{M3_OUTPUT}/**/R3_SIT_snow_swap_all_winters_v7.csv', recursive=True)
if hits:
    s = pd.read_csv(hits[0], low_memory=False)
    print(f"   rows in file: {len(s):,}")
    for c in s.columns:
        if any(k in c.lower() for k in ('h_s_ct', 'h_s_cci', 'dist')):
            print(f"   {c:30s} non-null: {s[c].notna().sum():,}")
else:
    print("   R3_SIT_snow_swap_all_winters_v7.csv not found")

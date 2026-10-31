#!/usr/bin/env python3
"""
R3_Step16_v7_alt_priors.py — variance decomposition under the alternative priors of Table 1
============================================================================================
Table 1 (and Appendix Table A.1) list four "alternative priors evaluated as sensitivity
cases", but no v7 result for them is reported anywhere in the manuscript. This script
produces those results on exactly the v7 decomposition subsample (5,000 matchups,
seed-42 replay), with the one-at-a-time seeds of Step 02 v7 (300+ip) reused for every
case (common random numbers), so differences between cases come from the priors only.

Cases
  base            primary priors; reproduces Step 02 v7 within simultaneous-draw noise
  rhoi_fyi_900    first-year component 900.0 +/- 35.7, cap 940 (multi-year unchanged)
  rhoi_fyi_800    first-year component 800.0 +/- 50.0, cap 940 (multi-year unchanged)
  delta_058       penetration N(0.58, 0.20) clipped [0.30, 1.00]
  cs2_sigma_0100  CryoSat-2 radar freeboard sigma 0.100 m (ICESat-2 stays 0.030 m)

Shares are fractions of the simultaneous-perturbation variance (not renormalised),
computed as region-mean single-parameter variance / region-mean simultaneous variance,
identical to R3_sweep_engine_v7.shares_by_region.

Also reports the deterministic central-value mean thickness per stratum over all
65,165 matchups for each case (the 800 and 900 cases shift absolute thickness).

Input : {M3_OUTPUT}/R3_input_d070_icetype.pkl, R3_step02_v7_subsample_idx.npy (via engine)
Output: {M3_OUTPUT}/R3_alt_priors_v7.csv, {M3_OUTPUT}/R3_alt_priors_central_hi_v7.csv
Run   : qsub pbs_step16_alt_priors_v7.sh   (or interactively; ~10-30 min)
Author: Xinlong Liu | 26 September 2026
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *
from R3_sweep_engine_v7 import PARAMS, BASE_SEED, load_subsample, shares_by_region

t0 = time.time()

BASE = dict(fyi_mu=RHO_I_FYI_MU, fyi_sigma=RHO_I_FYI_SIGMA,
            delta_mu=DELTA_BASELINE, is2_sigma=FB_SIGMA_RAW, cs2_sigma=FB_SIGMA_RAW)
CASES = {
    'base':           dict(BASE),
    'rhoi_fyi_900':   dict(BASE, fyi_mu=900.0, fyi_sigma=35.7),
    'rhoi_fyi_800':   dict(BASE, fyi_mu=800.0, fyi_sigma=50.0),
    'delta_058':      dict(BASE, delta_mu=0.58),
    'cs2_sigma_0100': dict(BASE, cs2_sigma=0.100),
}


# ── samplers with explicit case parameters (same rng call order as the engine) ──
def rho_i_draw(myi, n_draw, rng, c):
    f = np.clip(np.asarray(myi, float) / 100.0, 0.0, 1.0)
    is_myi = rng.random((n_draw, len(f))) < f[np.newaxis, :]
    mu = np.where(is_myi, RHO_I_MYI_MU, c['fyi_mu'])
    sd = np.where(is_myi, RHO_I_MYI_SIGMA, c['fyi_sigma'])
    return np.minimum(rng.normal(mu, sd), RHO_I_UPPER_CAP)


def rho_i_central(myi, c):
    f = np.clip(np.asarray(myi, float) / 100.0, 0.0, 1.0)
    return f * RHO_I_MYI_MU + (1.0 - f) * c['fyi_mu']


def delta_draw(n, rng, c):
    return np.clip(rng.normal(c['delta_mu'], DELTA_SIGMA, size=n), DELTA_CLIP[0], DELTA_CLIP[1])


def hi_from(is2, cs2, rho_s, rho_i, delta):
    eta = (1 + 0.51 * rho_s / 1000.0) ** 1.5
    hs = (is2 - cs2) / (delta * eta)
    return compute_ice_thickness(is2, hs, rho_s, rho_i, RHO_W_CENTRAL)


def single_param_variance(sub, param, seed, c, n_mc=N_MC):
    rng = np.random.default_rng(seed)
    n = len(sub['idx'])
    mu_i = rho_i_central(sub['myi'], c)
    hi = np.zeros((n_mc, n))
    for t in range(n_mc):
        is2, cs2 = sub['is2'], sub['cs2']
        rho_s, rho_i, delta = RHO_S_CENTRAL, mu_i, c['delta_mu']
        if param == 'fb_is2':
            is2 = is2 + rng.normal(0, c['is2_sigma'], n)
        elif param == 'fb_cs2':
            cs2 = cs2 + rng.normal(0, c['cs2_sigma'], n)
        elif param == 'rho_s':
            rho_s = rng.normal(RHO_S_CENTRAL, RHO_S_SIGMA)
        elif param == 'rho_i':
            rho_i = rho_i_draw(sub['myi'], 1, rng, c)[0]
        elif param == 'delta':
            delta = delta_draw(1, rng, c)[0]
        hi[t] = hi_from(is2, cs2, rho_s, rho_i, delta)
    return np.var(hi, axis=0)


def simultaneous_variance(sub, seed, c, n_mc=N_MC, batch=1000):
    rng = np.random.default_rng(seed)
    n = len(sub['idx'])
    var = np.zeros(n)
    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        nb = i1 - i0
        is2 = rng.normal(sub['is2'][i0:i1], c['is2_sigma'], size=(n_mc, nb))
        cs2 = rng.normal(sub['cs2'][i0:i1], c['cs2_sigma'], size=(n_mc, nb))
        delta = delta_draw(n_mc, rng, c).reshape(-1, 1)
        rho_s = rng.normal(RHO_S_CENTRAL, RHO_S_SIGMA, size=(n_mc, 1))
        rho_i = rho_i_draw(sub['myi'][i0:i1], n_mc, rng, c)
        var[i0:i1] = np.var(hi_from(is2, cs2, rho_s, rho_i, delta), axis=0)
    return var


# ── variance decomposition ───────────────────────────────────────────────
print("=" * 78)
print("R3 Step 16 v7: alternative priors of Table 1 (mixture rho_i prior, v7 subsample)")
print("=" * 78)
sub = load_subsample()

rows = []
for k, (case, c) in enumerate(CASES.items()):
    t = time.time()
    var_p = {p: single_param_variance(sub, p, BASE_SEED[p], c) for p in PARAMS}
    var_sim = simultaneous_variance(sub, 7000 + k, c)
    for r in shares_by_region(sub, var_p, var_sim):
        r['case'] = case
        rows.append(r)
    print(f"  {case:15s} done in {time.time() - t:5.0f}s")

out = pd.DataFrame(rows)
p_csv = f'{M3_OUTPUT}/R3_alt_priors_v7.csv'
out.to_csv(p_csv, index=False)
print(f"\nSaved: {p_csv}")

# ── deterministic central-value thickness, full dataset ──────────────────
df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_icetype.pkl')
if 'subsector' not in df.columns:
    df['subsector'] = assign_subsector_vec(df['lon'].values)
masks = {r: get_region_mask_v6(df['sector'].values, df['subsector'].values,
                               df['icetype'].values, r) for r in REGIONS_V6}
masks = {'all': np.ones(len(df), bool), **masks}
cen = []
for case, c in CASES.items():
    hi = hi_from(df['is2_fb_wm'].values, df['cs2_rfb'].values, RHO_S_CENTRAL,
                 rho_i_central(df['myi_frac'].values, c), c['delta_mu'])
    cen.append({'case': case, **{r: float(np.mean(hi[m])) for r, m in masks.items()}})
cen = pd.DataFrame(cen)
p_cen = f'{M3_OUTPUT}/R3_alt_priors_central_hi_v7.csv'
cen.to_csv(p_cen, index=False)
print(f"Saved: {p_cen}")

# ── compact summary for the manuscript ───────────────────────────────────
strata = ['all', 'Weddell', 'Weddell_W', 'Weddell_E', 'Ross', 'MYI_obs', 'FYI_obs']
print("\nShares (%) of simultaneous variance: rho_i / delta / fb_CS2 / interaction")
print(f"  {'case':15s}" + "".join(f"{s:>22s}" for s in strata))
for case in CASES:
    line = f"  {case:15s}"
    for s in strata:
        r = out[(out.case == case) & (out.region == s)]
        if len(r):
            r = r.iloc[0]
            line += f"  {r.frac_rho_i:4.1f}/{r.frac_delta:4.1f}/{r.frac_fb_cs2:4.1f}/{r.frac_interaction:4.1f}"
        else:
            line += f"{'--':>22s}"
    print(line)

print("\nCentral-value mean sea-ice thickness (m)")
print(cen.set_index('case')[strata].round(2).to_string())

print("\nLeading term per stratum (rho_i or delta)")
for case in CASES:
    lead = []
    for s in strata:
        r = out[(out.case == case) & (out.region == s)]
        if len(r):
            r = r.iloc[0]
            lead.append(f"{s}:{'rho_i' if r.frac_rho_i > r.frac_delta else 'delta'}")
    print(f"  {case:15s} " + "  ".join(lead))

print(f"\nStep 16 v7 complete in {(time.time() - t0) / 60:.1f} min")

#!/usr/bin/env python3
"""
R3_sweep_engine_v7.py — shared machinery for the v7 sensitivity sweeps
=======================================================================
Used by R3_Step03_v7_rho_sweep.py and R3_Step03b_v7_delta_sweep.py.

Conventions (identical to R3_Step02_v7_mixture.py):
  * rho_i prior  : per-matchup two-component mixture, MYI weight = collocated
                   Melsheimer et al. (2023) MYI concentration (Step 13)
  * subsample    : the SAME 5,000 matchups as Step 02 v7 Experiment B, obtained
                   by replaying the seed-42 stream (config.replay_step02_subsample)
  * shares       : single-parameter variance / SIMULTANEOUS variance, not
                   renormalised; the residual is reported as 'interaction'
  * seeds        : the one-at-a-time seeds 300+ip are reused for the baseline
                   configuration so that the rho_i / delta single-parameter
                   variances equal those of Step 02 v7 exactly (self-check)
  * strata       : REGIONS_V6 (four sectors + observed MYI / FYI)
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

PARAMS = ['fb_is2', 'fb_cs2', 'rho_s', 'rho_i', 'delta']
BASE_SEED = {p: 300 + i for i, p in enumerate(PARAMS)}   # Step 02 seeds


def load_subsample(n_sub=5000, log=print):
    """Load the ice-type pickle and return the Step 02 v7 subsample arrays."""
    df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_icetype.pkl')
    if 'subsector' not in df.columns:
        df['subsector'] = assign_subsector_vec(df['lon'].values)
    n_total = len(df)
    idx_path = f'{M3_OUTPUT}/R3_step02_v7_subsample_idx.npy'
    if os.path.exists(idx_path):
        idx = np.load(idx_path)
        log(f"  Subsample: loaded {idx_path}")
    else:
        t = time.time()
        idx = replay_step02_subsample(n_total, n_sub=n_sub)
        np.save(idx_path, idx)
        log(f"  Subsample: replayed Step 02 v7 stream in {time.time() - t:.0f}s, saved {idx_path}")
    sub = dict(
        idx=idx,
        is2=df['is2_fb_wm'].values[idx],
        cs2=df['cs2_rfb'].values[idx],
        sectors=df['sector'].values[idx],
        subsectors=df['subsector'].values[idx],
        icetype=df['icetype'].values[idx],
        myi=df['myi_frac'].values[idx],
    )
    sub['mu_i'] = mixture_rho_i_central(sub['myi'])
    sub['masks'] = {r: get_region_mask_v6(sub['sectors'], sub['subsectors'], sub['icetype'], r)
                    for r in REGIONS_V6}
    log(f"  Loaded {n_total:,} matchups; subsample n={len(idx):,}")
    return sub


def single_param_variance(sub, param, seed, n_mc=N_MC, sigma_scale=1.0, delta_sigma=DELTA_SIGMA):
    """One-at-a-time perturbation of one parameter; per-segment variance."""
    rng = np.random.default_rng(seed)
    n = len(sub['idx'])
    hi = np.zeros((n_mc, n))
    for t in range(n_mc):
        is2, cs2 = sub['is2'], sub['cs2']
        rho_s, rho_i, delta = RHO_S_CENTRAL, sub['mu_i'], DELTA_BASELINE
        if param == 'fb_is2':
            is2 = is2 + rng.normal(0, FB_SIGMA_RAW, n)
        elif param == 'fb_cs2':
            cs2 = cs2 + rng.normal(0, FB_SIGMA_RAW, n)
        elif param == 'rho_s':
            rho_s = rng.normal(RHO_S_CENTRAL, RHO_S_SIGMA)
        elif param == 'rho_i':
            rho_i = sample_rho_i_mixture_scaled(sub['myi'], 1, rng, sigma_scale)[0]
        elif param == 'delta':
            delta = sample_delta_sigma(1, rng, delta_sigma)[0]
        eta = (1 + 0.51 * rho_s / 1000.0) ** 1.5
        hs = (is2 - cs2) / (delta * eta)
        hi[t] = compute_ice_thickness(is2, hs, rho_s, rho_i, RHO_W_CENTRAL)
    return np.var(hi, axis=0)


def simultaneous_variance(sub, seed, n_mc=N_MC, sigma_scale=1.0, delta_sigma=DELTA_SIGMA, batch=1000):
    """Experiment-A-style joint perturbation on the subsample; per-segment variance."""
    rng = np.random.default_rng(seed)
    n = len(sub['idx'])
    var = np.zeros(n)
    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        nb = i1 - i0
        is2 = rng.normal(sub['is2'][i0:i1], FB_SIGMA_RAW, size=(n_mc, nb))
        cs2 = rng.normal(sub['cs2'][i0:i1], FB_SIGMA_RAW, size=(n_mc, nb))
        delta = sample_delta_sigma(n_mc, rng, delta_sigma).reshape(-1, 1)
        rho_s = rng.normal(RHO_S_CENTRAL, RHO_S_SIGMA, size=(n_mc, 1))
        rho_i = sample_rho_i_mixture_scaled(sub['myi'][i0:i1], n_mc, rng, sigma_scale)
        eta = (1 + 0.51 * rho_s / 1000.0) ** 1.5
        hs = (is2 - cs2) / (delta * eta)
        hi = compute_ice_thickness(is2, hs, rho_s, rho_i, RHO_W_CENTRAL)
        var[i0:i1] = np.var(hi, axis=0)
    return var


def shares_by_region(sub, var_by_param, var_sim):
    """Rows of un-renormalised shares (%) for every stratum in REGIONS_V6 + 'all'."""
    rows = []
    for region in ['all'] + list(REGIONS_V6):
        m = np.ones(len(sub['idx']), bool) if region == 'all' else sub['masks'][region]
        if m.sum() < 10:
            continue
        vs = np.mean(var_sim[m])
        row = {'region': region, 'n': int(m.sum())}
        tot = 0.0
        for p in PARAMS:
            row[f'frac_{p}'] = float(np.mean(var_by_param[p][m]) / vs * 100)
            tot += row[f'frac_{p}']
        row['frac_fb_combined'] = row['frac_fb_is2'] + row['frac_fb_cs2']
        row['frac_interaction'] = 100.0 - tot
        rows.append(row)
    return rows


def print_region_table(df_rows, sweep_col, label):
    for region in ['all'] + list(REGIONS_V6):
        sub = df_rows[df_rows['region'] == region].sort_values(sweep_col)
        if len(sub) == 0:
            continue
        print(f"\n  {REGION_LABELS.get(region, region)} (n={sub.iloc[0]['n']:,}):")
        print(f"  {label:>8s}  {'fb_IS2':>6s}  {'fb_CS2':>6s}  {'rho_s':>6s}  "
              f"{'rho_i':>6s}  {'delta':>6s}  {'inter':>6s}  {'d/rho_i':>7s}")
        print("  " + "-" * 66)
        for _, r in sub.iterrows():
            print(f"  {r[sweep_col]:8.3g}  {r['frac_fb_is2']:5.1f}%  {r['frac_fb_cs2']:5.1f}%  "
                  f"{r['frac_rho_s']:5.1f}%  {r['frac_rho_i']:5.1f}%  {r['frac_delta']:5.1f}%  "
                  f"{r['frac_interaction']:5.1f}%  {r['frac_delta'] / r['frac_rho_i']:6.2f}")


def check_against_step02(df_rows, sweep_col, baseline_value, tol=0.6):
    """At the baseline configuration the shares should reproduce Step 02 v7."""
    import json
    p = f'{M3_OUTPUT}/R3_ExpB_d070_v7.json'
    if not os.path.exists(p):
        print(f"\n  (no {p}; skipping Step 02 v7 cross-check)")
        return
    ref = json.load(open(p))
    base = df_rows[np.isclose(df_rows[sweep_col], baseline_value)]
    print(f"\n  Cross-check at {sweep_col} = {baseline_value} against Step 02 v7 "
          f"(differences arise only from the simultaneous-variance draw; expect < {tol} pt):")
    worst = 0.0
    for _, r in base.iterrows():
        reg = r['region']
        for p_ in PARAMS:
            d = r[f'frac_{p_}'] - ref[p_][reg]
            worst = max(worst, abs(d))
    print(f"  max |sweep - Step02| over all strata and parameters: {worst:.2f} pt "
          f"{'OK' if worst < tol else '*** CHECK ***'}")

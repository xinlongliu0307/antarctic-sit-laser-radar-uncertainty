#!/usr/bin/env python3
"""
R3_Step15_state_dependent_delta.py — Run 4: penetration fraction conditioned on
snow and ice state (Reviewer 3 major comment 4, RSE-D-26-03119)
==============================================================================
The primary analysis samples delta ~ N(0.70, 0.20) independently of the snow
and ice state. This script repeats the variance decomposition with the MEAN of
the delta prior made a function of the matchup's state, holding its spread at
0.20 unless stated, to answer "what happens if penetration is allowed to vary
with snow depth, snow density or ice type? Would the ranking remain the same?"

The state enters only through the prior mean, mu_delta(x); the draw is still
delta ~ N(mu_delta(x), sigma_delta) clipped to [0.30, 1.00], and snow depth is
re-derived from the drawn delta exactly as in Step 02, so the algebraic
delta-h_s coupling is preserved. The snow-depth predictor is the central-value
retrieval h_s(delta = 0.70), which avoids circularity.

Scenarios (all relative to the 0.70 baseline; signs follow the physical
expectation that deeper, denser or flooded snow places the scattering horizon
higher in the pack, i.e. lower delta, and dry multi-year-ice snow lower, i.e.
higher delta):
  S0  baseline                mu = 0.70                                  (= Step 02 v7)
  S1  ice type                mu = 0.70 - 0.05 + 0.10 f          (FYI 0.65, MYI 0.75)
  S2  snow depth              mu = 0.70 - 0.10 * (h_s - h_s,med) / IQR(h_s)
  S3  snow density            mu = 0.70 - 0.05 * (rho_s - 330) / 70   (rho_s drawn jointly)
  S4  combined                S1 + S2 + S3
  S4r combined, reversed      signs of S1-S3 reversed (tests direction dependence)
  S5  heteroscedastic         mu as S4; sigma_delta = 0.15 over MYI, 0.25 over FYI
Amplitudes are chosen so that mu_delta spans roughly 0.60-0.80 across the
dataset (the range of empirical delta* in Liu et al. 2026c and Arctic studies);
they are assumptions, and the script reports the realised mu_delta range.

Subsample, seeds and share convention identical to Step 02 v7; S0 reproduces it.

Input : {M3_OUTPUT}/R3_input_d070_icetype.pkl, R3_ExpB_d070_v7.json
Output: {M3_OUTPUT}/R3_state_dependent_delta_v7.csv  (shares by scenario and stratum)
        {M3_OUTPUT}/R3_state_dependent_delta_v7.json
        {M3_FIGURES}/R3_Fig_state_dependent_delta.{png,pdf}
Run   : source ~/cryo2ice_env/bin/activate && python R3_Step15_state_dependent_delta.py
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *
from R3_sweep_engine_v7 import PARAMS, BASE_SEED, load_subsample, shares_by_region

t0 = time.time()
A_TYPE, A_HS, A_RS = 0.10, 0.10, 0.05          # amplitudes (see docstring)

print("=" * 72)
print("R3 Step 15 (Run 4): state-dependent penetration-fraction prior")
print("=" * 72)
sub = load_subsample()
n = len(sub['idx'])
f = np.clip(sub['myi'] / 100.0, 0.0, 1.0)
hs_c = (sub['is2'] - sub['cs2']) / (DELTA_BASELINE * ETA_S)         # central-value snow depth
hs_med = np.median(hs_c)
hs_iqr = np.subtract(*np.percentile(hs_c, [75, 25]))
print(f"  central h_s: median {hs_med:.3f} m, IQR {hs_iqr:.3f} m; MYI weight mean {f.mean():.3f}")


def mu_delta(scn, rho_s):
    """Per-matchup prior mean of delta for scenario scn; rho_s may be scalar or array."""
    sgn = -1.0 if scn == 'S4r' else 1.0
    mu = np.full(n, DELTA_BASELINE)
    if scn in ('S1', 'S4', 'S4r', 'S5'):
        mu = mu + sgn * (-A_TYPE / 2 + A_TYPE * f)
    if scn in ('S2', 'S4', 'S4r', 'S5'):
        mu = mu - sgn * A_HS * (hs_c - hs_med) / hs_iqr
    if scn in ('S3', 'S4', 'S4r', 'S5'):
        mu = mu - sgn * A_RS * (np.asarray(rho_s) - RHO_S_CENTRAL) / RHO_S_SIGMA
    return np.clip(mu, DELTA_CLIP[0] + 0.05, DELTA_CLIP[1] - 0.05)


def sigma_delta(scn):
    if scn == 'S5':
        return np.where(f >= 0.5, 0.15, 0.25)
    return np.full(n, DELTA_SIGMA)


def draw_delta(scn, rng, rho_s, size_first=None):
    """Draw delta with state-dependent mean; rho_s scalar (one-at-a-time) or (N,1)/(N,n) array."""
    mu = mu_delta(scn, rho_s)
    sg = sigma_delta(scn)
    if size_first is None:
        return np.clip(rng.normal(mu, sg), *DELTA_CLIP)                        # (n,)
    return np.clip(rng.normal(mu[np.newaxis, :], sg[np.newaxis, :], size=(size_first, n)), *DELTA_CLIP)


def one_at_a_time(scn, param, seed):
    rng = np.random.default_rng(seed)
    hi = np.zeros((N_MC, n))
    for t in range(N_MC):
        is2, cs2 = sub['is2'], sub['cs2']
        rho_s, rho_i = RHO_S_CENTRAL, sub['mu_i']
        delta = mu_delta(scn, RHO_S_CENTRAL)                                   # unperturbed: prior mean
        if param == 'fb_is2':
            is2 = is2 + rng.normal(0, FB_SIGMA_RAW, n)
        elif param == 'fb_cs2':
            cs2 = cs2 + rng.normal(0, FB_SIGMA_RAW, n)
        elif param == 'rho_s':
            rho_s = rng.normal(RHO_S_CENTRAL, RHO_S_SIGMA)
            delta = mu_delta(scn, rho_s)          # S3/S4: the prior mean follows rho_s; no delta noise here
        elif param == 'rho_i':
            rho_i = sample_rho_i_mixture(sub['myi'], 1, rng)[0]
        elif param == 'delta':
            delta = draw_delta(scn, rng, RHO_S_CENTRAL)
        eta = (1 + 0.51 * rho_s / 1000.0) ** 1.5
        hs = (is2 - cs2) / (delta * eta)
        hi[t] = compute_ice_thickness(is2, hs, rho_s, rho_i, RHO_W_CENTRAL)
    return np.var(hi, axis=0)


def simultaneous(scn, seed, batch=1000):
    rng = np.random.default_rng(seed)
    var = np.zeros(n)
    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        nb = i1 - i0
        is2 = rng.normal(sub['is2'][i0:i1], FB_SIGMA_RAW, size=(N_MC, nb))
        cs2 = rng.normal(sub['cs2'][i0:i1], FB_SIGMA_RAW, size=(N_MC, nb))
        rho_s = rng.normal(RHO_S_CENTRAL, RHO_S_SIGMA, size=(N_MC, 1))
        mu_full = mu_delta(scn, rho_s)[:, i0:i1] if scn in ('S3', 'S4', 'S4r', 'S5') else \
            np.broadcast_to(mu_delta(scn, RHO_S_CENTRAL)[i0:i1], (N_MC, nb))
        sg = sigma_delta(scn)[i0:i1]
        delta = np.clip(rng.normal(mu_full, sg[np.newaxis, :]), *DELTA_CLIP)
        rho_i = sample_rho_i_mixture(sub['myi'][i0:i1], N_MC, rng)
        eta = (1 + 0.51 * rho_s / 1000.0) ** 1.5
        hs = (is2 - cs2) / (delta * eta)
        var[i0:i1] = np.var(compute_ice_thickness(is2, hs, rho_s, rho_i, RHO_W_CENTRAL), axis=0)
    return var


SCENARIOS = ['S0', 'S1', 'S2', 'S3', 'S4', 'S4r', 'S5']
LABELS = {'S0': 'baseline (state-independent)', 'S1': 'ice type', 'S2': 'snow depth', 'S3': 'snow density',
          'S4': 'combined', 'S4r': 'combined, reversed signs', 'S5': 'combined + heteroscedastic'}
rows, meta = [], {}
for k, scn in enumerate(SCENARIOS):
    t = time.time()
    mu = mu_delta(scn, RHO_S_CENTRAL)
    seeds = {p: (BASE_SEED[p] if scn == 'S0' else 9000 + 10 * k + i) for i, p in enumerate(PARAMS)}
    var_by_param = {p: one_at_a_time(scn, p, seeds[p]) for p in PARAMS}
    var_sim = simultaneous(scn, 5000 if scn == 'S0' else 9500 + k)
    for r in shares_by_region(sub, var_by_param, var_sim):
        r.update(scenario=scn, label=LABELS[scn])
        rows.append(r)
    meta[scn] = dict(label=LABELS[scn], mu_delta_mean=float(mu.mean()), mu_delta_p05=float(np.percentile(mu, 5)),
                     mu_delta_p95=float(np.percentile(mu, 95)), sigma_delta_mean=float(sigma_delta(scn).mean()))
    a = [r for r in rows if r['scenario'] == scn and r['region'] == 'all'][0]
    print(f"  {scn:4s} {LABELS[scn]:28s} mu_delta [{meta[scn]['mu_delta_p05']:.2f}, {meta[scn]['mu_delta_p95']:.2f}]  "
          f"rho_i={a['frac_rho_i']:5.1f}%  delta={a['frac_delta']:5.1f}%  rho_s={a['frac_rho_s']:4.1f}%  "
          f"inter={a['frac_interaction']:4.1f}%  ({time.time() - t:.0f}s)")

out = pd.DataFrame(rows)
out.to_csv(f'{M3_OUTPUT}/R3_state_dependent_delta_v7.csv', index=False, float_format='%.3f')
with open(f'{M3_OUTPUT}/R3_state_dependent_delta_v7.json', 'w') as fh:
    json.dump({'amplitudes': dict(ice_type=A_TYPE, snow_depth_per_IQR=A_HS, snow_density_per_sigma=A_RS),
               'scenarios': meta, 'n_sub': int(n), 'n_mc': N_MC}, fh, indent=2)

print("\n  rho_i / delta shares (%) by scenario and stratum:")
hdr = f"  {'scenario':5s} " + " ".join(f"{r:>12s}" for r in ['all'] + list(REGIONS_V6))
print(hdr)
for scn in SCENARIOS:
    s = out[out.scenario == scn].set_index('region')
    print(f"  {scn:5s} " + " ".join(f"{s.loc[r, 'frac_rho_i']:5.1f}/{s.loc[r, 'frac_delta']:5.1f}" for r in ['all'] + list(REGIONS_V6)))
print("\n  Ordering (rho_i leads = R, delta leads = D):")
for scn in SCENARIOS:
    s = out[out.scenario == scn].set_index('region')
    print(f"  {scn:5s} " + " ".join(f"{'R' if s.loc[r, 'frac_rho_i'] >= s.loc[r, 'frac_delta'] else 'D':>12s}" for r in ['all'] + list(REGIONS_V6)))

# ── Figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
xs = np.arange(len(SCENARIOS))
for ax, reg, ttl in zip(axes, ['all', 'Weddell_W'], ['(a) All matchups', '(b) Western Weddell Sea']):
    s = out[out.region == reg].set_index('scenario').reindex(SCENARIOS)
    for p, c in [('rho_i', '#009E73'), ('delta', '#0072B2'), ('rho_s', '#F0E442'), ('fb_cs2', '#E69F00'), ('interaction', '#7F7F7F')]:
        ax.plot(xs, s[f'frac_{p}'], '-o', color=c, lw=1.6, ms=5,
                label={'rho_i': '$\\rho_i$', 'delta': '$\\delta$', 'rho_s': '$\\rho_s$', 'fb_cs2': '$h_f^{\\mathrm{CS2}}$', 'interaction': 'interaction'}[p])
        for x, v in zip(xs, s[f'frac_{p}']):
            if p in ('rho_i', 'delta'):
                ax.text(x, v + 1, f'{v:.1f}', ha='center', fontsize=7, color=c)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{k}\n{LABELS[k]}' for k in SCENARIOS], fontsize=7)
    ax.set_ylabel('Fraction of simultaneous variance (%)')
    ax.set_title(ttl, fontsize=10)
    ax.set_ylim(0, 60)
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=8, frameon=False, ncol=5, loc='upper center')
fig.suptitle('Variance budget with the penetration-fraction prior conditioned on ice type, snow depth and snow density',
             fontsize=10)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'{M3_FIGURES}/R3_Fig_state_dependent_delta.{ext}')
print(f"\nSaved: R3_state_dependent_delta_v7.csv/json, R3_Fig_state_dependent_delta.png/pdf")
print(f"Step 15 complete in {(time.time() - t0) / 60:.1f} min")

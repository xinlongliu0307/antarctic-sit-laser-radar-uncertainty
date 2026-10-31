#!/usr/bin/env python3
"""
R3_Step02b_v7_shape_robustness.py — rho_i distribution-shape robustness, mixture prior
======================================================================================
Derived from R3_Step02b_shape_robustness.py. Changes:
  * the shape test is applied to the per-matchup MIXTURE prior: BOTH components
    (FYI 916.7/35.7, MYI 882.0/23.0) are replaced by skew-normal (alpha = -2)
    distributions with location and scale adjusted so that each component's
    realised mean and standard deviation match its Gaussian counterpart; the
    940 kg/m^3 cap and the MYI weight f are unchanged
  * shares are fractions of the SIMULTANEOUS variance (un-renormalised), on the
    same 5,000-matchup subsample and with the same seeds as Step 02 v7, so the
    Gaussian row reproduces Table 3 and the residual interaction is carried
  * strata: REGIONS_V6 (four sectors + observed MYI / FYI)
  * full N_MC (10,000) rather than the 5,000 of the original

Input : {M3_OUTPUT}/R3_input_d070_icetype.pkl, R3_ExpB_d070_v7.json (cross-check)
Output: {M3_OUTPUT}/R3_shape_robustness_v7.json, {M3_FIGURES}/R3_Fig_shape_robustness.{png,pdf}
Run   : source ~/cryo2ice_env/bin/activate && python R3_Step02b_v7_shape_robustness.py
"""
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *
from R3_sweep_engine_v7 import (PARAMS, BASE_SEED, load_subsample, single_param_variance,
                                simultaneous_variance, shares_by_region)

t0 = time.time()
SHAPES = ['gaussian', 'skewnorm']
SHAPE_LABELS = {'gaussian': 'Gaussian components', 'skewnorm': 'Skew-normal components ($\\alpha=-2$)'}
ALPHA = -2.0


def skew_loc_scale(mu, sigma, alpha=ALPHA):
    """Skew-normal loc/scale reproducing the target mean and standard deviation."""
    d = alpha / np.sqrt(1 + alpha ** 2)
    scale = sigma / np.sqrt(1 - 2 * d ** 2 / np.pi)
    loc = mu - scale * np.sqrt(2 / np.pi) * d
    return loc, scale


def sample_rho_i_mixture_shape(myi_frac, n_per_seg, rng, shape):
    """Mixture prior with Gaussian or skew-normal components; same MYI weight f."""
    f = np.clip(np.asarray(myi_frac, dtype=float) / 100.0, 0.0, 1.0)
    is_myi = rng.random((n_per_seg, len(f))) < f[np.newaxis, :]
    if shape == 'gaussian':
        mu = np.where(is_myi, RHO_I_MYI_MU, RHO_I_FYI_MU)
        sg = np.where(is_myi, RHO_I_MYI_SIGMA, RHO_I_FYI_SIGMA)
        s = rng.normal(mu, sg)
    else:
        lf, sf = skew_loc_scale(RHO_I_FYI_MU, RHO_I_FYI_SIGMA)
        lm, sm = skew_loc_scale(RHO_I_MYI_MU, RHO_I_MYI_SIGMA)
        z = skewnorm.rvs(ALPHA, size=is_myi.shape, random_state=rng.integers(2 ** 31))
        s = np.where(is_myi, lm + sm * z, lf + sf * z)
    return np.minimum(s, RHO_I_UPPER_CAP)


print("=" * 72)
print("R3 Step 02b v7: rho_i shape robustness for the mixture prior")
print(f"  shapes: {SHAPES}; N_MC = {N_MC:,}; strata: {REGIONS_V6}")
print("=" * 72)

rng_t = np.random.default_rng(0)
print("\nComponent check (200,000 draws, before cap):")
for shape in SHAPES:
    for f_, name in [(0.0, 'FYI'), (100.0, 'MYI')]:
        s = sample_rho_i_mixture_shape(np.array([f_]), 200000, rng_t, shape)[:, 0]
        print(f"  {shape:9s} {name}: mean={s.mean():7.1f}  std={s.std():5.1f}  "
              f"skew={((s - s.mean()) ** 3).mean() / s.std() ** 3:+.2f}  at cap={100 * np.mean(s >= RHO_I_UPPER_CAP - 1e-6):.1f}%")

sub = load_subsample()

# Fixed perturbations are shape-independent; compute once with Step 02 seeds
var_fixed = {p: single_param_variance(sub, p, BASE_SEED[p]) for p in ['fb_is2', 'fb_cs2', 'rho_s', 'delta']}

results, rows_out = {}, []
for k, shape in enumerate(SHAPES):
    t = time.time()
    # rho_i one-at-a-time with the shape-specific sampler (Step 02 seed for Gaussian)
    rng = np.random.default_rng(BASE_SEED['rho_i'] if shape == 'gaussian' else 700 + k)
    n = len(sub['idx'])
    hi = np.zeros((N_MC, n))
    for tr in range(N_MC):
        rho_i = sample_rho_i_mixture_shape(sub['myi'], 1, rng, shape)[0]
        hs = (sub['is2'] - sub['cs2']) / (DELTA_BASELINE * ETA_S)
        hi[tr] = compute_ice_thickness(sub['is2'], hs, RHO_S_CENTRAL, rho_i, RHO_W_CENTRAL)
    var_by_param = dict(var_fixed, rho_i=np.var(hi, axis=0))

    # simultaneous variance with the same shape
    rng_s = np.random.default_rng(5000 if shape == 'gaussian' else 800 + k)
    var_sim = np.zeros(n)
    for i0 in range(0, n, 1000):
        i1 = min(i0 + 1000, n)
        nb = i1 - i0
        is2 = rng_s.normal(sub['is2'][i0:i1], FB_SIGMA_RAW, size=(N_MC, nb))
        cs2 = rng_s.normal(sub['cs2'][i0:i1], FB_SIGMA_RAW, size=(N_MC, nb))
        delta = sample_delta(N_MC, rng_s).reshape(-1, 1)
        rho_s = rng_s.normal(RHO_S_CENTRAL, RHO_S_SIGMA, size=(N_MC, 1))
        rho_i = sample_rho_i_mixture_shape(sub['myi'][i0:i1], N_MC, rng_s, shape)
        eta = (1 + 0.51 * rho_s / 1000.0) ** 1.5
        hs = (is2 - cs2) / (delta * eta)
        var_sim[i0:i1] = np.var(compute_ice_thickness(is2, hs, rho_s, rho_i, RHO_W_CENTRAL), axis=0)

    shape_rows = shares_by_region(sub, var_by_param, var_sim)
    results[shape] = {r['region']: r for r in shape_rows}
    print(f"\n  {SHAPE_LABELS[shape]} ({time.time() - t:.0f}s)")
    print(f"  {'stratum':10s}  {'fb_IS2':>6s}  {'fb_CS2':>6s}  {'rho_s':>6s}  {'rho_i':>6s}  {'delta':>6s}  {'inter':>6s}")
    for r in shape_rows:
        print(f"  {r['region']:10s}  {r['frac_fb_is2']:5.1f}%  {r['frac_fb_cs2']:5.1f}%  {r['frac_rho_s']:5.1f}%  "
              f"{r['frac_rho_i']:5.1f}%  {r['frac_delta']:5.1f}%  {r['frac_interaction']:5.1f}%")

print("\n  Skew-normal minus Gaussian (percentage points):")
print(f"  {'stratum':10s}  " + "  ".join(f"{p:>7s}" for p in PARAMS + ['interaction']))
maxd = 0.0
for reg in ['all'] + list(REGIONS_V6):
    d = [results['skewnorm'][reg][f'frac_{p}'] - results['gaussian'][reg][f'frac_{p}'] for p in PARAMS + ['interaction']]
    maxd = max(maxd, max(abs(x) for x in d))
    print(f"  {reg:10s}  " + "  ".join(f"{x:+7.2f}" for x in d))
print(f"  Largest absolute change in any share: {maxd:.2f} points")

with open(f'{M3_OUTPUT}/R3_shape_robustness_v7.json', 'w') as f:
    json.dump({'alpha': ALPHA, 'n_mc': N_MC, 'results': results, 'max_abs_change_pt': maxd,
               'note': 'Both mixture components skew-normal with matched mean/std; shares un-renormalised.'}, f, indent=2)
print(f"  Saved: {M3_OUTPUT}/R3_shape_robustness_v7.json")

# ── Figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
shape_col = {'gaussian': '#0072B2', 'skewnorm': '#882255'}
plab = {'fb_is2': '$h_f^{\\mathrm{IS2}}$', 'fb_cs2': '$h_f^{\\mathrm{CS2}}$', 'rho_s': '$\\rho_s$',
        'rho_i': '$\\rho_i$', 'delta': '$\\delta$', 'interaction': 'inter.'}
cats = PARAMS + ['interaction']
ax = axes[0]
x = np.arange(len(cats))
for i, shape in enumerate(SHAPES):
    v = [results[shape]['all'][f'frac_{p}'] for p in cats]
    b = ax.bar(x + (i - 0.5) * 0.36, v, 0.36, label=SHAPE_LABELS[shape], color=shape_col[shape], alpha=0.85, edgecolor='white')
    for bb, vv in zip(b, v):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.5, f'{vv:.1f}', ha='center', va='bottom', fontsize=8, color=shape_col[shape])
ax.set_xticks(x)
ax.set_xticklabels([plab[p] for p in cats], fontsize=11)
ax.set_ylabel('Fraction of simultaneous variance (%)')
ax.set_title('(a) All matchups', fontsize=11)
ax.set_ylim(0, 55)
ax.legend(fontsize=9, loc='upper left', frameon=False)
ax.grid(axis='y', alpha=0.2)

ax = axes[1]
xr = np.arange(len(REGIONS_V6))
for i, shape in enumerate(SHAPES):
    v = [results[shape][r]['frac_rho_i'] for r in REGIONS_V6]
    b = ax.bar(xr + (i - 0.5) * 0.36, v, 0.36, label=SHAPE_LABELS[shape], color=shape_col[shape], alpha=0.85, edgecolor='white')
    for bb, vv in zip(b, v):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.5, f'{vv:.1f}', ha='center', va='bottom', fontsize=8, color=shape_col[shape])
ax.set_xticks(xr)
ax.set_xticklabels([REGION_LABELS[r].replace(' (all sectors)', '') for r in REGIONS_V6], fontsize=8.5, rotation=15)
ax.set_ylabel('$\\rho_i$ share of simultaneous variance (%)')
ax.set_title('(b) $\\rho_i$ contribution by stratum', fontsize=11)
ax.set_ylim(0, 60)
ax.grid(axis='y', alpha=0.2)

# inset: the four component PDFs
ins = axes[0].inset_axes([0.55, 0.55, 0.42, 0.40])
xx = np.linspace(780, 960, 400)
from scipy.stats import norm
for mu, sg, lab, c in [(RHO_I_FYI_MU, RHO_I_FYI_SIGMA, 'FYI', '#44AA99'), (RHO_I_MYI_MU, RHO_I_MYI_SIGMA, 'MYI', '#882255')]:
    lo, sc = skew_loc_scale(mu, sg)
    ins.plot(xx, norm.pdf(xx, mu, sg), '-', color=c, lw=1.2, label=f'{lab} Gaussian')
    ins.plot(xx, skewnorm.pdf(xx, ALPHA, lo, sc), '--', color=c, lw=1.2, label=f'{lab} skew-normal')
ins.axvline(RHO_I_UPPER_CAP, color='k', lw=0.6, ls=':')
ins.set_xlabel('$\\rho_i$ (kg m$^{-3}$)', fontsize=7)
ins.set_yticks([])
ins.tick_params(labelsize=7)
ins.legend(fontsize=5.5, frameon=False)

fig.suptitle('Robustness of the variance decomposition to the shape of the $\\rho_i$ mixture components '
             '(940 kg m$^{-3}$ cap and MYI weight unchanged)', fontsize=10)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'{M3_FIGURES}/R3_Fig_shape_robustness.{ext}')
print(f"  Saved: R3_Fig_shape_robustness.png/pdf")
print(f"\nStep 02b v7 complete in {(time.time() - t0) / 60:.1f} min")

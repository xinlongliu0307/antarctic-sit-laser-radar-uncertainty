#!/usr/bin/env python3
"""
R3_Step03_variance_decomposition_ExpB.py
=========================================
Experiment B: One-at-a-time perturbation for variance decomposition.

Column names (verified 01 Apr 2026):
  cs2_rfb, is2_fb_wm, snow_thickness, snow_thickness_unc, sector (derived)

Environment: ~/cryo2ice_env/ on NCI Gadi (Python 3.9.2)
Author: Xinlong Liu | 01 Apr 2026
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
# 0. Configuration
# ────────────────────────────────────────────────────────────────
M3_BASE = '/g/data/gv90/xl1657/phd/M3_workspace'
OUT_DIR = os.path.join(M3_BASE, 'output')
FIG_DIR = os.path.join(M3_BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

N_SUB = 2000;  N_MC = 5000;  SEED = 42

COL_CS2_FB = 'cs2_rfb'
COL_HS     = 'snow_thickness'
COL_HS_UNC = 'snow_thickness_unc'
COL_SECTOR = 'sector'

PARAM_NAMES = [
    r'$f_b^{\mathrm{radar}}$ (CS2)', r'$h_s$ (snow depth)',
    r'$\rho_s$ (snow density)', r'$\rho_i$ (ice density)',
    r'$\rho_w$ (water density)', r'$\delta$ (penetration)',
]
PARAM_SHORT = ['fb_radar', 'h_s', 'rho_s', 'rho_i', 'rho_w', 'delta']
N_PARAMS = 6

RHO_S_MEAN, RHO_S_SIGMA = 330.0, 70.0
RHO_S_LO, RHO_S_HI = 200.0, 450.0
RHO_I_MEAN, RHO_I_SIGMA = 915.0, 25.0
RHO_W_MEAN, RHO_W_SIGMA = 1024.0, 0.5
DELTA_LO, DELTA_HI = 0.50, 1.00
DELTA_CENTRAL = 1.0
FB_SIGMA = 0.027

t0 = time.time()


def eta_s(rho_s):
    return (1.0 + 0.51 * rho_s / 1000.0) ** 1.5


def ice_thickness(fb_radar, h_s, rho_s, rho_i, rho_w, delta):
    eta = eta_s(rho_s)
    fb_ice = fb_radar + (1.0 - delta) * h_s / eta
    return (rho_w * fb_ice + rho_s * h_s) / (rho_w - rho_i)


# ────────────────────────────────────────────────────────────────
# 1. Load verified data
# ────────────────────────────────────────────────────────────────
print('=' * 65)
print('R3 Step 03: Experiment B — Variance Decomposition')
print('=' * 65)

d = pd.read_pickle(os.path.join(OUT_DIR, 'R3_input_verified.pkl'))
print(f'Loaded {len(d):,} matchups\n')

# ────────────────────────────────────────────────────────────────
# 2. Subsample
# ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
n_use = min(N_SUB, len(d))
idx = rng.choice(len(d), size=n_use, replace=False)
ds = d.iloc[idx].reset_index(drop=True)

fb_radar = ds[COL_CS2_FB].values.astype(np.float64)
hs_obs   = ds[COL_HS].values.astype(np.float64)
hs_unc   = ds[COL_HS_UNC].values.astype(np.float64)
hs_unc   = np.where(np.isnan(hs_unc) | (hs_unc <= 0), 0.03, hs_unc)
sector   = ds[COL_SECTOR].values

print(f'Subset: {n_use} matchups × {N_MC:,} MC × {N_PARAMS} params')
print(f'Total evaluations: {n_use * N_MC * N_PARAMS:,.0f}\n')

# ────────────────────────────────────────────────────────────────
# 3. One-at-a-time perturbation
# ────────────────────────────────────────────────────────────────
print('Running one-at-a-time perturbation...')
t_mc = time.time()

var_pp = np.full((n_use, N_PARAMS), np.nan)

for k in range(n_use):
    if (k + 1) % 500 == 0:
        print(f'  {k+1}/{n_use}  ({100*(k+1)/n_use:.0f}%)')

    for p in range(N_PARAMS):
        fb_k  = np.full(N_MC, fb_radar[k])
        hs_k  = np.full(N_MC, hs_obs[k])
        rs_k  = np.full(N_MC, RHO_S_MEAN)
        ri_k  = np.full(N_MC, RHO_I_MEAN)
        rw_k  = np.full(N_MC, RHO_W_MEAN)
        del_k = np.full(N_MC, DELTA_CENTRAL)

        if   p == 0: fb_k  = rng.normal(fb_radar[k], FB_SIGMA, N_MC)
        elif p == 1: hs_k  = rng.normal(hs_obs[k], hs_unc[k], N_MC)
        elif p == 2: rs_k  = np.clip(rng.normal(RHO_S_MEAN, RHO_S_SIGMA, N_MC), RHO_S_LO, RHO_S_HI)
        elif p == 3: ri_k  = rng.normal(RHO_I_MEAN, RHO_I_SIGMA, N_MC)
        elif p == 4: rw_k  = rng.normal(RHO_W_MEAN, RHO_W_SIGMA, N_MC)
        elif p == 5: del_k = rng.uniform(DELTA_LO, DELTA_HI, N_MC)

        var_pp[k, p] = np.nanvar(ice_thickness(fb_k, hs_k, rs_k, ri_k, rw_k, del_k))

print(f'Perturbation complete in {time.time() - t_mc:.1f} s\n')

# ────────────────────────────────────────────────────────────────
# 4. Compute fractional contributions
# ────────────────────────────────────────────────────────────────
mv_all = np.nanmean(var_pp, axis=0)
frac_all = 100 * mv_all / mv_all.sum()

print('=' * 55)
print('EXPERIMENT B: Variance Decomposition — All Matchups')
print('=' * 55)
for p in range(N_PARAMS):
    bar = '█' * int(frac_all[p] / 2)
    print(f'  {PARAM_SHORT[p]:>10s}: {frac_all[p]:5.1f}%  {bar}')

w_mask = (sector == 'Weddell')
r_mask = (sector == 'Ross')

frac_w = frac_all.copy()
frac_r = frac_all.copy()

for name, mask in [('Weddell', w_mask), ('Ross', r_mask)]:
    n = mask.sum()
    if n == 0:
        print(f'\n{name}: no matchups'); continue
    mv = np.nanmean(var_pp[mask], axis=0)
    frac = 100 * mv / mv.sum()
    if name == 'Weddell': frac_w = frac
    else: frac_r = frac
    print(f'\n{name} (n={n}):')
    for p in range(N_PARAMS):
        bar = '█' * int(frac[p] / 2)
        print(f'  {PARAM_SHORT[p]:>10s}: {frac[p]:5.1f}%  {bar}')

# ────────────────────────────────────────────────────────────────
# 5. Headline figure: grouped bar chart
# ────────────────────────────────────────────────────────────────
print('\nGenerating headline figure...')

fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
x = np.arange(N_PARAMS); width = 0.32

bars_w = ax.bar(x - width/2, frac_w, width, color='#C0392B', alpha=0.85,
                label='Weddell Sea', edgecolor='white', linewidth=0.5)
bars_r = ax.bar(x + width/2, frac_r, width, color='#2980B9', alpha=0.85,
                label='Ross Sea', edgecolor='white', linewidth=0.5)

for bars in [bars_w, bars_r]:
    for bar in bars:
        h = bar.get_height()
        if h > 1.5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(PARAM_NAMES, fontsize=10)
ax.set_ylabel('Fractional Contribution to $h_i$ Variance (%)', fontsize=12)
ax.set_title('Ice Thickness Uncertainty Budget: Variance Decomposition\n'
             f'(one-at-a-time, n = {n_use:,}, N = {N_MC:,})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(0, max(frac_w.max(), frac_r.max()) * 1.18)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(FIG_DIR, 'R3_ExpB_variance_decomposition.png'),
            dpi=250, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'R3_ExpB_variance_decomposition.pdf'),
            bbox_inches='tight')
print(f'  Saved: {FIG_DIR}/R3_ExpB_variance_decomposition.png')
plt.close(fig)

# ────────────────────────────────────────────────────────────────
# 6. Penetration sensitivity curve
# ────────────────────────────────────────────────────────────────
print('Generating penetration sensitivity figure...')

delta_range = np.linspace(0.40, 1.00, 61)
fig2, ax2 = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

for name, mask, c in [('Weddell', w_mask, '#C0392B'), ('Ross', r_mask, '#2980B9')]:
    if not mask.any(): continue
    fb_m, hs_m = np.nanmean(fb_radar[mask]), np.nanmean(hs_obs[mask])
    hi_c = ice_thickness(fb_m, hs_m, RHO_S_MEAN, RHO_I_MEAN, RHO_W_MEAN, delta_range)
    ax2.plot(delta_range, hi_c, c=c, lw=2.5, label=name)
    hi_lo = ice_thickness(fb_m, hs_m, RHO_S_MEAN+RHO_S_SIGMA, RHO_I_MEAN, RHO_W_MEAN, delta_range)
    hi_hi = ice_thickness(fb_m, hs_m, RHO_S_MEAN-RHO_S_SIGMA, RHO_I_MEAN, RHO_W_MEAN, delta_range)
    ax2.fill_between(delta_range, hi_lo, hi_hi, color=c, alpha=0.12)

ax2.axvline(1.0, color='gray', lw=1, ls=':', label=r'$\delta=1$ (full penetration)')
ax2.axvline(0.55, color='#27AE60', lw=1.5, ls='--', label=r'$\delta^*\approx 0.55$ (M2)')
ax2.set_xlabel(r'Penetration fraction $\delta$', fontsize=12)
ax2.set_ylabel('$h_i$ (m)', fontsize=12)
ax2.set_title('Sensitivity of $h_i$ to penetration assumption',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left'); ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

fig2.savefig(os.path.join(FIG_DIR, 'R3_penetration_sensitivity.png'),
             dpi=250, bbox_inches='tight')
fig2.savefig(os.path.join(FIG_DIR, 'R3_penetration_sensitivity.pdf'),
             bbox_inches='tight')
print(f'  Saved: {FIG_DIR}/R3_penetration_sensitivity.png')
plt.close(fig2)

# ────────────────────────────────────────────────────────────────
# 7. Save results
# ────────────────────────────────────────────────────────────────
np.savez(os.path.join(OUT_DIR, 'R3_ExpB_results.npz'),
         var_per_param=var_pp, frac_all=frac_all,
         frac_weddell=frac_w, frac_ross=frac_r,
         param_names=PARAM_SHORT, param_labels=PARAM_NAMES,
         N_sub=n_use, N_mc=N_MC, sector=sector)

pd.DataFrame({
    'Parameter': PARAM_SHORT, 'All_pct': frac_all,
    'Weddell_pct': frac_w, 'Ross_pct': frac_r,
}).to_csv(os.path.join(OUT_DIR, 'R3_ExpB_summary.csv'),
         index=False, float_format='%.1f')

print(f'  Saved: {OUT_DIR}/R3_ExpB_results.npz')
print(f'  Saved: {OUT_DIR}/R3_ExpB_summary.csv')

elapsed = time.time() - t0
print(f'\nStep 03 complete in {elapsed:.1f} seconds.')

print('\n' + '=' * 55)
print('KEY TAKEAWAY FOR SUPERVISORY MEETING:')
print('=' * 55)
dominant_idx = np.argmax(frac_all)
second_idx   = np.argsort(frac_all)[-2]
print(f'  Dominant source: {PARAM_SHORT[dominant_idx]} '
      f'({frac_all[dominant_idx]:.0f}%)')
print(f'  Second:          {PARAM_SHORT[second_idx]} '
      f'({frac_all[second_idx]:.0f}%)')
print(f'  rho_w contributes: {frac_all[4]:.1f}% (negligible, as expected)')

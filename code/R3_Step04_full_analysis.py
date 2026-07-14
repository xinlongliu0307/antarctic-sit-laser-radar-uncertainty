#!/usr/bin/env python3
"""
R3_Step04_full_analysis.py
==========================
Comprehensive R3 analysis for supervisory meeting and Easter work block.

Part 1: Load data + apply quality filter (cs2_rfb in [-0.5, 1.0] m)
Part 2: σ_ρi sensitivity sweep  → crossover line plot (KEY FIGURE)
Part 3: Full-dataset Experiment A (quality-filtered)  → h_i distributions
Part 4: Full-dataset Experiment B (quality-filtered)  → variance decomposition
Part 5: Updated penetration sensitivity with δ* = 0.55/0.58
Part 6: Summary table for all results

Reads from:  /g/data/gv90/xl1657/phd/M3_workspace/output/R3_input_verified.pkl
Writes to:   /g/data/gv90/xl1657/phd/M3_workspace/output/ and figures/

Environment: ~/cryo2ice_env/ on NCI Gadi (Python 3.9.2)
Author: Xinlong Liu | 02 Apr 2026
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
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42

# Verified column names
COL_CS2_FB = 'cs2_rfb'
COL_IS2_FB = 'is2_fb_wm'
COL_HS     = 'snow_thickness'
COL_HS_UNC = 'snow_thickness_unc'
COL_SECTOR = 'sector'

# Quality filter bounds
FB_LO, FB_HI = -0.5, 1.0  # CS2 radar freeboard [m]

# Parameter central values
RHO_S_MEAN, RHO_S_SIGMA = 330.0, 70.0
RHO_S_LO, RHO_S_HI = 200.0, 450.0
RHO_I_MEAN = 915.0            # σ varies in sweep
RHO_W_MEAN, RHO_W_SIGMA = 1024.0, 0.5
DELTA_LO, DELTA_HI = 0.50, 1.00
DELTA_CENTRAL = 1.0
FB_SIGMA = 0.027

# M2 empirically optimal penetration factors
DELTA_STAR_MEDIAN = 0.55
DELTA_STAR_MEAN   = 0.58

# σ_ρi sweep values
RHO_I_SIGMAS = [10, 15, 20, 25, 30, 35]

# Monte Carlo settings
N_MC_SWEEP = 5000     # iterations for sensitivity sweep
N_SUB_SWEEP = 5000    # matchups for sweep (subset for speed)
N_MC_FULL_A = 10000   # iterations for full Experiment A
N_MC_FULL_B = 5000    # iterations for full Experiment B

PARAM_NAMES = [
    r'$f_b^{\mathrm{radar}}$ (CS2)', r'$h_s$ (snow depth)',
    r'$\rho_s$ (snow density)', r'$\rho_i$ (ice density)',
    r'$\rho_w$ (water density)', r'$\delta$ (penetration)',
]
PARAM_SHORT = ['fb_radar', 'h_s', 'rho_s', 'rho_i', 'rho_w', 'delta']
N_PARAMS = 6

# Colours
C_WEDDELL = '#C0392B'
C_ROSS    = '#2980B9'
C_ALL     = '#2C3E50'

t_total = time.time()


def eta_s(rho_s):
    return (1.0 + 0.51 * rho_s / 1000.0) ** 1.5


def ice_thickness(fb_radar, h_s, rho_s, rho_i, rho_w, delta):
    eta = eta_s(rho_s)
    fb_ice = fb_radar + (1.0 - delta) * h_s / eta
    return (rho_w * fb_ice + rho_s * h_s) / (rho_w - rho_i)


# ================================================================
# PART 1: Load data + quality filter
# ================================================================
print('=' * 65)
print('R3 Step 04: Full Analysis (quality-filtered)')
print('=' * 65)

d_raw = pd.read_pickle(os.path.join(OUT_DIR, 'R3_input_verified.pkl'))
n_raw = len(d_raw)

# Apply quality filter on CS2 radar freeboard
qf_mask = d_raw[COL_CS2_FB].between(FB_LO, FB_HI)
d = d_raw[qf_mask].copy().reset_index(drop=True)
n_filtered = len(d)
n_removed = n_raw - n_filtered

print(f'\nQuality filter: {COL_CS2_FB} in [{FB_LO}, {FB_HI}] m')
print(f'  Before: {n_raw:,}')
print(f'  After:  {n_filtered:,}')
print(f'  Removed: {n_removed:,} ({100*n_removed/n_raw:.1f}%)')

# Sector counts
n_w = (d[COL_SECTOR] == 'Weddell').sum()
n_r = (d[COL_SECTOR] == 'Ross').sum()
print(f'  Weddell: {n_w:,}')
print(f'  Ross:    {n_r:,}')

# Quick stats after filter
hs = d[COL_HS]
print(f'\nPost-filter h_s: mean={hs.mean():.4f}, median={hs.median():.4f}, '
      f'neg={100*(hs<0).sum()/len(hs):.1f}%')
fb = d[COL_CS2_FB]
print(f'Post-filter fb:  mean={fb.mean():.4f}, '
      f'range=[{fb.min():.3f}, {fb.max():.3f}]')

# Extract arrays for MC
fb_all   = d[COL_CS2_FB].values.astype(np.float64)
hs_all   = d[COL_HS].values.astype(np.float64)
hsunc_all = d[COL_HS_UNC].values.astype(np.float64)
hsunc_all = np.where(np.isnan(hsunc_all) | (hsunc_all <= 0), 0.03, hsunc_all)
sector_all = d[COL_SECTOR].values
w_all = (sector_all == 'Weddell')
r_all = (sector_all == 'Ross')


# ================================================================
# PART 2: σ_ρi Sensitivity Sweep
# ================================================================
print('\n' + '=' * 65)
print('PART 2: sigma_rho_i Sensitivity Sweep')
print('=' * 65)

rng = np.random.default_rng(SEED)
n_sweep = min(N_SUB_SWEEP, n_filtered)
idx_sweep = rng.choice(n_filtered, size=n_sweep, replace=False)

fb_sw   = fb_all[idx_sweep]
hs_sw   = hs_all[idx_sweep]
hsu_sw  = hsunc_all[idx_sweep]
sec_sw  = sector_all[idx_sweep]
w_sw    = (sec_sw == 'Weddell')
r_sw    = (sec_sw == 'Ross')

print(f'Sweep subset: {n_sweep} matchups × {N_MC_SWEEP:,} MC '
      f'× {N_PARAMS} params × {len(RHO_I_SIGMAS)} σ values')
print(f'Total evaluations: {n_sweep * N_MC_SWEEP * N_PARAMS * len(RHO_I_SIGMAS):,.0f}\n')

# Storage: [n_sigma, N_PARAMS] for all, Weddell, Ross
frac_sweep_all = np.zeros((len(RHO_I_SIGMAS), N_PARAMS))
frac_sweep_w   = np.zeros((len(RHO_I_SIGMAS), N_PARAMS))
frac_sweep_r   = np.zeros((len(RHO_I_SIGMAS), N_PARAMS))

for si, sigma_ri in enumerate(RHO_I_SIGMAS):
    t_s = time.time()
    var_pp = np.full((n_sweep, N_PARAMS), np.nan)

    for k in range(n_sweep):
        for p in range(N_PARAMS):
            fb_k  = np.full(N_MC_SWEEP, fb_sw[k])
            hs_k  = np.full(N_MC_SWEEP, hs_sw[k])
            rs_k  = np.full(N_MC_SWEEP, RHO_S_MEAN)
            ri_k  = np.full(N_MC_SWEEP, RHO_I_MEAN)
            rw_k  = np.full(N_MC_SWEEP, RHO_W_MEAN)
            del_k = np.full(N_MC_SWEEP, DELTA_CENTRAL)

            if   p == 0: fb_k  = rng.normal(fb_sw[k], FB_SIGMA, N_MC_SWEEP)
            elif p == 1: hs_k  = rng.normal(hs_sw[k], hsu_sw[k], N_MC_SWEEP)
            elif p == 2: rs_k  = np.clip(rng.normal(RHO_S_MEAN, RHO_S_SIGMA, N_MC_SWEEP), RHO_S_LO, RHO_S_HI)
            elif p == 3: ri_k  = rng.normal(RHO_I_MEAN, sigma_ri, N_MC_SWEEP)
            elif p == 4: rw_k  = rng.normal(RHO_W_MEAN, RHO_W_SIGMA, N_MC_SWEEP)
            elif p == 5: del_k = rng.uniform(DELTA_LO, DELTA_HI, N_MC_SWEEP)

            var_pp[k, p] = np.nanvar(ice_thickness(fb_k, hs_k, rs_k, ri_k, rw_k, del_k))

    # Compute fractions
    mv = np.nanmean(var_pp, axis=0)
    frac_sweep_all[si] = 100 * mv / mv.sum()

    if w_sw.any():
        mv_w = np.nanmean(var_pp[w_sw], axis=0)
        frac_sweep_w[si] = 100 * mv_w / mv_w.sum()

    if r_sw.any():
        mv_r = np.nanmean(var_pp[r_sw], axis=0)
        frac_sweep_r[si] = 100 * mv_r / mv_r.sum()

    elapsed_s = time.time() - t_s
    print(f'  σ_ρi = {sigma_ri:2d} kg/m³: '
          f'ρ_i = {frac_sweep_all[si, 3]:5.1f}%, '
          f'fb = {frac_sweep_all[si, 0]:5.1f}%, '
          f'δ = {frac_sweep_all[si, 5]:5.1f}%  '
          f'({elapsed_s:.1f}s)')

# --- KEY FIGURE: σ_ρi sensitivity line plot ---
print('\nGenerating sensitivity sweep figure...')

param_colors = ['#E74C3C', '#3498DB', '#F39C12', '#2ECC71', '#9B59B6', '#1ABC9C']
param_markers = ['o', 's', '^', 'D', 'v', 'P']

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

for ax, frac_data, title in [
    (axes[0], frac_sweep_all, 'All matchups'),
    (axes[1], frac_sweep_w,   'Weddell Sea'),
    (axes[2], frac_sweep_r,   'Ross Sea'),
]:
    for p in range(N_PARAMS):
        ax.plot(RHO_I_SIGMAS, frac_data[:, p],
                color=param_colors[p], marker=param_markers[p],
                markersize=7, lw=2, label=PARAM_SHORT[p])

    ax.set_xlabel(r'$\sigma_{\rho_i}$ (kg m$^{-3}$)', fontsize=12)
    ax.set_ylabel('Fractional variance contribution (%)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(-2, 102)
    ax.set_xticks(RHO_I_SIGMAS)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[2].legend(fontsize=9, loc='center right',
               bbox_to_anchor=(1.0, 0.5))

fig.suptitle(r'Sensitivity of $h_i$ Variance Decomposition to Ice Density Uncertainty'
             f'\n(quality-filtered, n = {n_sweep:,}, N = {N_MC_SWEEP:,})',
             fontsize=13, fontweight='bold', y=1.02)

fig.savefig(os.path.join(FIG_DIR, 'R3_rho_i_sensitivity_sweep.png'),
            dpi=250, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'R3_rho_i_sensitivity_sweep.pdf'),
            bbox_inches='tight')
print(f'  Saved: {FIG_DIR}/R3_rho_i_sensitivity_sweep.png')
plt.close(fig)

# Save sweep data
sweep_results = {
    'sigma_rho_i': RHO_I_SIGMAS,
    'frac_all': frac_sweep_all,
    'frac_weddell': frac_sweep_w,
    'frac_ross': frac_sweep_r,
    'param_names': PARAM_SHORT,
    'n_sweep': n_sweep,
    'N_mc': N_MC_SWEEP,
}
np.savez(os.path.join(OUT_DIR, 'R3_rho_i_sweep_results.npz'), **sweep_results)

# CSV summary
rows = []
for si, sig in enumerate(RHO_I_SIGMAS):
    for p in range(N_PARAMS):
        rows.append({
            'sigma_rho_i': sig,
            'parameter': PARAM_SHORT[p],
            'all_pct': frac_sweep_all[si, p],
            'weddell_pct': frac_sweep_w[si, p],
            'ross_pct': frac_sweep_r[si, p],
        })
pd.DataFrame(rows).to_csv(
    os.path.join(OUT_DIR, 'R3_rho_i_sweep_summary.csv'),
    index=False, float_format='%.2f')
print(f'  Saved: {OUT_DIR}/R3_rho_i_sweep_summary.csv')


# ================================================================
# PART 3: Full-Dataset Experiment A (quality-filtered)
# ================================================================
print('\n' + '=' * 65)
print(f'PART 3: Experiment A — Full Dataset ({n_filtered:,} matchups)')
print('=' * 65)

t_A = time.time()
hi_mean_f = np.full(n_filtered, np.nan)
hi_med_f  = np.full(n_filtered, np.nan)
hi_std_f  = np.full(n_filtered, np.nan)
hi_p05_f  = np.full(n_filtered, np.nan)
hi_p95_f  = np.full(n_filtered, np.nan)

rng_A = np.random.default_rng(SEED)

for k in range(n_filtered):
    if (k + 1) % 10000 == 0:
        elapsed_k = time.time() - t_A
        eta_min = elapsed_k / (k + 1) * (n_filtered - k - 1) / 60
        print(f'  {k+1:,}/{n_filtered:,}  '
              f'({100*(k+1)/n_filtered:.0f}%, ETA {eta_min:.1f} min)')

    rho_s_d = np.clip(rng_A.normal(RHO_S_MEAN, RHO_S_SIGMA, N_MC_FULL_A),
                      RHO_S_LO, RHO_S_HI)
    rho_i_d = rng_A.normal(RHO_I_MEAN, 25.0, N_MC_FULL_A)  # σ=25 default
    rho_w_d = rng_A.normal(RHO_W_MEAN, RHO_W_SIGMA, N_MC_FULL_A)
    delta_d = rng_A.uniform(DELTA_LO, DELTA_HI, N_MC_FULL_A)
    fb_d    = rng_A.normal(fb_all[k], FB_SIGMA, N_MC_FULL_A)
    hs_d    = rng_A.normal(hs_all[k], hsunc_all[k], N_MC_FULL_A)

    hi_d = ice_thickness(fb_d, hs_d, rho_s_d, rho_i_d, rho_w_d, delta_d)

    hi_mean_f[k] = np.nanmean(hi_d)
    hi_med_f[k]  = np.nanmedian(hi_d)
    hi_std_f[k]  = np.nanstd(hi_d)
    hi_p05_f[k]  = np.nanpercentile(hi_d, 5)
    hi_p95_f[k]  = np.nanpercentile(hi_d, 95)

elapsed_A = time.time() - t_A
print(f'\nExperiment A (full) complete in {elapsed_A:.1f} s')

print(f'\n--- Experiment A Results (quality-filtered, n={n_filtered:,}) ---')
print(f'All:')
print(f'  Mean h_i:       {np.nanmean(hi_mean_f):.3f} m')
print(f'  Mean sigma:     {np.nanmean(hi_std_f):.3f} m')
print(f'  Mean 90%% CI:   {np.nanmean(hi_p95_f - hi_p05_f):.3f} m')

for name, mask in [('Weddell', w_all), ('Ross', r_all)]:
    n = mask.sum()
    if n == 0: continue
    print(f'{name} (n={n:,}):')
    print(f'  h_i = {np.nanmean(hi_mean_f[mask]):.3f} +/- '
          f'{np.nanmean(hi_std_f[mask]):.3f} m')
    print(f'  90%% CI: [{np.nanmean(hi_p05_f[mask]):.3f}, '
          f'{np.nanmean(hi_p95_f[mask]):.3f}] m')

# Save full Experiment A
expA_full = pd.DataFrame({
    'hi_mean': hi_mean_f, 'hi_median': hi_med_f, 'hi_std': hi_std_f,
    'hi_p05': hi_p05_f, 'hi_p95': hi_p95_f,
    'sector': sector_all, 'hs_obs': hs_all, 'fb_radar': fb_all,
})
expA_full.to_pickle(os.path.join(OUT_DIR, 'R3_ExpA_full_QF_results.pkl'))
print(f'  Saved: {OUT_DIR}/R3_ExpA_full_QF_results.pkl')

# Experiment A figure (quality-filtered, full dataset)
print('Generating Experiment A figure (quality-filtered)...')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

ax = axes[0]
valid = hi_mean_f[~np.isnan(hi_mean_f)]
ax.hist(valid, bins=80, color='#3B6EA5', edgecolor='none', alpha=0.85,
        range=(-2, 8))
ax.axvline(np.nanmean(valid), color=C_WEDDELL, lw=2, label='Mean')
ax.axvline(np.nanpercentile(valid, 5), color=C_WEDDELL, lw=1, ls='--',
           label='5th/95th')
ax.axvline(np.nanpercentile(valid, 95), color=C_WEDDELL, lw=1, ls='--')
ax.set_xlabel('$h_i$ (m)'); ax.set_ylabel('Count')
ax.set_title(f'(a) $h_i$ distribution (n={n_filtered:,}, QF)')
ax.legend(fontsize=9)

ax = axes[1]
# Subsample for scatter to avoid overplotting
n_sc = min(5000, n_filtered)
sc_idx = rng_A.choice(n_filtered, n_sc, replace=False)
for name, mask, c in [('Weddell', w_all, C_WEDDELL), ('Ross', r_all, C_ROSS)]:
    m = mask[sc_idx]
    ax.scatter(hi_mean_f[sc_idx][m], hi_std_f[sc_idx][m],
               s=4, c=c, alpha=0.3, label=name, rasterized=True)
ax.set_xlabel('Mean $h_i$ (m)'); ax.set_ylabel(r'$\sigma_{h_i}$ (m)')
ax.set_title('(b) Uncertainty vs thickness')
ax.set_xlim(-3, 8); ax.set_ylim(0, 8)
ax.legend(fontsize=9)

ax = axes[2]
ci_f = hi_p95_f - hi_p05_f
ax.hist(ci_f[~np.isnan(ci_f)], bins=80, color='#27AE60', edgecolor='none',
        alpha=0.85, range=(0, 8))
ax.axvline(np.nanmean(ci_f), color='k', lw=2,
           label=f'Mean = {np.nanmean(ci_f):.2f} m')
ax.set_xlabel('90% CI Width (m)'); ax.set_ylabel('Count')
ax.set_title('(c) CI width distribution'); ax.legend(fontsize=9)

fig.savefig(os.path.join(FIG_DIR, 'R3_ExpA_full_QF.png'),
            dpi=200, bbox_inches='tight')
print(f'  Saved: {FIG_DIR}/R3_ExpA_full_QF.png')
plt.close(fig)


# ================================================================
# PART 4: Full-Dataset Experiment B (quality-filtered)
# ================================================================
print('\n' + '=' * 65)
print(f'PART 4: Experiment B — Full Dataset ({n_filtered:,} matchups)')
print('=' * 65)

t_B = time.time()
var_pp_full = np.full((n_filtered, N_PARAMS), np.nan)
rng_B = np.random.default_rng(SEED)

for k in range(n_filtered):
    if (k + 1) % 10000 == 0:
        elapsed_k = time.time() - t_B
        eta_min = elapsed_k / (k + 1) * (n_filtered - k - 1) / 60
        print(f'  {k+1:,}/{n_filtered:,}  '
              f'({100*(k+1)/n_filtered:.0f}%, ETA {eta_min:.1f} min)')

    for p in range(N_PARAMS):
        fb_k  = np.full(N_MC_FULL_B, fb_all[k])
        hs_k  = np.full(N_MC_FULL_B, hs_all[k])
        rs_k  = np.full(N_MC_FULL_B, RHO_S_MEAN)
        ri_k  = np.full(N_MC_FULL_B, RHO_I_MEAN)
        rw_k  = np.full(N_MC_FULL_B, RHO_W_MEAN)
        del_k = np.full(N_MC_FULL_B, DELTA_CENTRAL)

        if   p == 0: fb_k  = rng_B.normal(fb_all[k], FB_SIGMA, N_MC_FULL_B)
        elif p == 1: hs_k  = rng_B.normal(hs_all[k], hsunc_all[k], N_MC_FULL_B)
        elif p == 2: rs_k  = np.clip(rng_B.normal(RHO_S_MEAN, RHO_S_SIGMA, N_MC_FULL_B), RHO_S_LO, RHO_S_HI)
        elif p == 3: ri_k  = rng_B.normal(RHO_I_MEAN, 25.0, N_MC_FULL_B)
        elif p == 4: rw_k  = rng_B.normal(RHO_W_MEAN, RHO_W_SIGMA, N_MC_FULL_B)
        elif p == 5: del_k = rng_B.uniform(DELTA_LO, DELTA_HI, N_MC_FULL_B)

        var_pp_full[k, p] = np.nanvar(
            ice_thickness(fb_k, hs_k, rs_k, ri_k, rw_k, del_k))

elapsed_B = time.time() - t_B
print(f'\nExperiment B (full) complete in {elapsed_B:.1f} s')

mv_f = np.nanmean(var_pp_full, axis=0)
frac_f = 100 * mv_f / mv_f.sum()

print(f'\n--- Experiment B Results (quality-filtered, n={n_filtered:,}) ---')
print(f'{"Parameter":>12s}  {"All":>6s}  {"Weddell":>7s}  {"Ross":>5s}')
print('-' * 38)

frac_fw = frac_f.copy()
frac_fr = frac_f.copy()
if w_all.any():
    mv_w = np.nanmean(var_pp_full[w_all], axis=0)
    frac_fw = 100 * mv_w / mv_w.sum()
if r_all.any():
    mv_r = np.nanmean(var_pp_full[r_all], axis=0)
    frac_fr = 100 * mv_r / mv_r.sum()

for p in range(N_PARAMS):
    print(f'{PARAM_SHORT[p]:>12s}  {frac_f[p]:5.1f}%  {frac_fw[p]:6.1f}%  {frac_fr[p]:4.1f}%')

# Save full Experiment B
np.savez(os.path.join(OUT_DIR, 'R3_ExpB_full_QF_results.npz'),
         var_per_param=var_pp_full, frac_all=frac_f,
         frac_weddell=frac_fw, frac_ross=frac_fr,
         param_names=PARAM_SHORT, N_mc=N_MC_FULL_B, sector=sector_all)

pd.DataFrame({
    'Parameter': PARAM_SHORT, 'All_pct': frac_f,
    'Weddell_pct': frac_fw, 'Ross_pct': frac_fr,
}).to_csv(os.path.join(OUT_DIR, 'R3_ExpB_full_QF_summary.csv'),
         index=False, float_format='%.1f')

# Full-dataset variance decomposition bar chart
print('Generating full-dataset variance decomposition figure...')
fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
x = np.arange(N_PARAMS); w = 0.32

ax.bar(x - w/2, frac_fw, w, color=C_WEDDELL, alpha=0.85,
       label='Weddell Sea', edgecolor='white', lw=0.5)
ax.bar(x + w/2, frac_fr, w, color=C_ROSS, alpha=0.85,
       label='Ross Sea', edgecolor='white', lw=0.5)

for xp in range(N_PARAMS):
    for val, dx in [(frac_fw[xp], -w/2), (frac_fr[xp], w/2)]:
        if val > 1.5:
            ax.text(xp + dx, val + 0.8, f'{val:.0f}%',
                    ha='center', va='bottom', fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(PARAM_NAMES, fontsize=10)
ax.set_ylabel('Fractional Contribution to $h_i$ Variance (%)', fontsize=12)
ax.set_title('Ice Thickness Uncertainty Budget (Quality-Filtered)\n'
             f'(full dataset, n = {n_filtered:,}, '
             r'$\sigma_{\rho_i}$ = 25 kg m$^{-3}$)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(0, max(frac_fw.max(), frac_fr.max()) * 1.18)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(FIG_DIR, 'R3_ExpB_full_QF_variance.png'),
            dpi=250, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'R3_ExpB_full_QF_variance.pdf'),
            bbox_inches='tight')
print(f'  Saved: {FIG_DIR}/R3_ExpB_full_QF_variance.png')
plt.close(fig)


# ================================================================
# PART 5: Penetration Sensitivity (corrected δ* values)
# ================================================================
print('\n' + '=' * 65)
print('PART 5: Penetration Sensitivity (corrected delta*)')
print('=' * 65)

delta_range = np.linspace(0.40, 1.00, 61)
fig_pen, ax_pen = plt.subplots(figsize=(7.5, 5), constrained_layout=True)

for name, mask, c in [('Weddell', w_all, C_WEDDELL),
                       ('Ross', r_all, C_ROSS)]:
    if not mask.any(): continue
    fb_m = np.nanmean(fb_all[mask])
    hs_m = np.nanmean(hs_all[mask])
    hi_c = ice_thickness(fb_m, hs_m, RHO_S_MEAN, RHO_I_MEAN,
                         RHO_W_MEAN, delta_range)
    ax_pen.plot(delta_range, hi_c, c=c, lw=2.5, label=name)
    hi_lo = ice_thickness(fb_m, hs_m, RHO_S_MEAN + RHO_S_SIGMA,
                          RHO_I_MEAN, RHO_W_MEAN, delta_range)
    hi_hi = ice_thickness(fb_m, hs_m, RHO_S_MEAN - RHO_S_SIGMA,
                          RHO_I_MEAN, RHO_W_MEAN, delta_range)
    ax_pen.fill_between(delta_range, hi_lo, hi_hi, color=c, alpha=0.12)

ax_pen.axvline(1.0, color='gray', lw=1, ls=':',
               label=r'$\delta = 1$ (full penetration)')
ax_pen.axvline(DELTA_STAR_MEDIAN, color='#27AE60', lw=1.5, ls='--',
               label=fr'$\delta^*_{{median}} \approx {DELTA_STAR_MEDIAN}$ (M2)')
ax_pen.axvline(DELTA_STAR_MEAN, color='#F39C12', lw=1.5, ls='-.',
               label=fr'$\delta^*_{{mean}} \approx {DELTA_STAR_MEAN}$ (M2)')

ax_pen.set_xlabel(r'Penetration fraction $\delta$', fontsize=12)
ax_pen.set_ylabel('$h_i$ (m)', fontsize=12)
ax_pen.set_title('Sensitivity of $h_i$ to Ku-band Penetration Assumption\n'
                 '(quality-filtered, sector means)',
                 fontsize=12, fontweight='bold')
ax_pen.legend(fontsize=9, loc='upper left')
ax_pen.grid(alpha=0.3)
ax_pen.spines['top'].set_visible(False)
ax_pen.spines['right'].set_visible(False)

fig_pen.savefig(os.path.join(FIG_DIR, 'R3_penetration_sensitivity_QF.png'),
                dpi=250, bbox_inches='tight')
fig_pen.savefig(os.path.join(FIG_DIR, 'R3_penetration_sensitivity_QF.pdf'),
                bbox_inches='tight')
print(f'  Saved: {FIG_DIR}/R3_penetration_sensitivity_QF.png')
plt.close(fig_pen)


# ================================================================
# PART 6: Summary
# ================================================================
elapsed_total = time.time() - t_total

print('\n' + '=' * 65)
print('R3 STEP 04 COMPLETE — SUMMARY')
print('=' * 65)
print(f'\nTotal runtime: {elapsed_total:.0f} seconds ({elapsed_total/60:.1f} min)')
print(f'\nQuality filter: {COL_CS2_FB} in [{FB_LO}, {FB_HI}] m')
print(f'  Removed {n_removed:,} / {n_raw:,} = {100*n_removed/n_raw:.1f}%')
print(f'  Retained: {n_filtered:,} matchups')
print(f'\nKey results:')
print(f'  Exp A — Mean h_i: {np.nanmean(hi_mean_f):.3f} m '
      f'(Weddell {np.nanmean(hi_mean_f[w_all]):.3f}, '
      f'Ross {np.nanmean(hi_mean_f[r_all]):.3f})')
print(f'  Exp A — Mean sigma: {np.nanmean(hi_std_f):.3f} m')
print(f'  Exp B — Dominant: {PARAM_SHORT[np.argmax(frac_f)]} '
      f'({frac_f.max():.0f}%)')

print(f'\nFiles saved:')
for d_name in ['output', 'figures']:
    path = os.path.join(M3_BASE, d_name)
    files = sorted(os.listdir(path))
    print(f'\n  {path}/')
    for f in files:
        size = os.path.getsize(os.path.join(path, f))
        print(f'    {f}  ({size/1024:.0f} KB)')

print(f'\nδ* values used: median = {DELTA_STAR_MEDIAN}, mean = {DELTA_STAR_MEAN}')
print('\nDone.')

#!/usr/bin/env python3
"""
R3_Step10_snow_swap_figures_v9.py
==================================
*** v9 CHANGES (September 2026, RSE-D-26-03119 resubmission): ***
  - Reads the Step 09 v7 output (per-matchup mixture expected rho_i from the
    collocated Melsheimer MYI concentration; sector rule retired)
  - Figure C2_02 gains a third panel (c): median Delta h_i as a function of
    the penetration fraction used to produce the C2I snow (delta 0.40-1.00),
    per product and sector, from R3_snow_swap_delta_sweep_v7.csv
    (Reviewer 3 major comment 5)
  - Parameter annotation states the mixture prior instead of the sector rule
  - Output directory figures_v7

*** v8 CHANGES (16 June 2026, Alex Figure Comments 1 & 2): ***
  - Figure C2_01 (snow scatter): hist2d now uses LogNorm so the
    low-density tails are visible alongside the high-density core
  - Figure C2_01: a single shared colour bar is added on the right
    side of the figure with the explicit label
    "Point density per bin (log scale)"
  - Figure C2_01: panel letters (a)..(f) are added in the upper-left
    of each subplot at fontsize=15, bold, non-italic, with a white
    rounded background so they are visible against the density plot
  - Figure C2_01: the per-panel statistics annotation box has been
    moved from the upper-LEFT to the lower-RIGHT corner so it does
    not occlude the densest part of the snow-depth distribution
    near the top of the y-axis (around y = 0.45 to 0.7 m)
  - Figure C2_01: panel-letter, axis-label, tick, and stats fonts
    have all been increased so the figure text is no longer small
    relative to the caption text

v7 features preserved:
  - Reference baseline AMSR2; CRYO2ICE as replacement (Option One)
  - Sign convention Delta h_i = h_i_Imp - h_i_Base
  - Output directory figures_v6 (unchanged)
  - Figure C2_02 violin plots and Figure C2_03 heatmap unchanged

Run on Gadi:
  source ~/cryo2ice_env/bin/activate
  python R3_Step10_snow_swap_figures_v9.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import (
    RHO_S_CENTRAL, RHO_W_CENTRAL, DELTA_BASELINE,
    RHO_I_FYI_MU, RHO_I_MYI_MU, RHO_I_UPPER_CAP,
)

RHO_I_FYI = RHO_I_FYI_MU   # mixture component means (Alexandrov et al. 2010)
RHO_I_MYI = RHO_I_MYI_MU
RHO_I_CAP = RHO_I_UPPER_CAP

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
IN_CSV    = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation/R3_SIT_snow_swap_all_winters_v7.csv'
SWEEP_CSV = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation/R3_snow_swap_delta_sweep_v7.csv'
OUT_DIR   = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation/figures_v7'

C2I_SNOW = 'snow_thickness'
CCI_SNOW = 'cci_snow_depth'
CT_SNOW  = 'ct_snow_depth'
SECTOR   = 'sector'

COL = {
    'Western_Weddell': '#56B4E9',
    'Eastern_Weddell': '#009E73',
    'Ross':            '#D55E00',
}
SECTOR_LABELS = {
    'Western_Weddell': 'Western Weddell',
    'Eastern_Weddell': 'Eastern Weddell',
    'Ross':            'Ross',
}
SECTORS = ['Western_Weddell', 'Eastern_Weddell', 'Ross']
YEARS   = [2022, 2023, 2024, 2025]

# Parameter annotation string (for caption area, not title)
# *** v6: Regional Alexandrov parameterisation per Alex/Stuart 28-29 April 2026 ***
PARAM_STR = (f'$\\rho_i$ per matchup = mixture expected value, '
             f'$f\\,${RHO_I_MYI:.1f} + (1$-f$)$\\,${RHO_I_FYI:.1f} kg m$^{{-3}}$ '
             f'($f$ = collocated MYI concentration), cap $\\leq${RHO_I_CAP:.0f}; '
             f'$\\rho_s$ = {RHO_S_CENTRAL:.0f} kg m$^{{-3}}$, '
             f'$\\rho_w$ = {RHO_W_CENTRAL:.0f} kg m$^{{-3}}$, '
             f'$\\delta$ = {DELTA_BASELINE}')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

os.makedirs(OUT_DIR, exist_ok=True)

# ===========================================================================
#  LOAD DATA
# ===========================================================================
print('Loading data...')
df = pd.read_csv(IN_CSV)
df['year'] = (df['year_month'] // 100).astype(int)

df['Delta_h_i_CCI'] = np.where(
    df['h_i_Imp_CCI'].notna() & df['h_i_Base_CCI'].notna(),
    df['h_i_Imp_CCI'] - df['h_i_Base_CCI'], np.nan)
df['Delta_h_i_CT'] = np.where(
    df['h_i_Imp_CT'].notna() & df['h_i_Base_CT'].notna(),
    df['h_i_Imp_CT'] - df['h_i_Base_CT'], np.nan)

print(f'  Loaded {len(df):,} rows')
print(f'  Option One framing: AMSR2 baseline -> CRYO2ICE replacement')
print(f'  Delta h_i = h_i_Imp - h_i_Base  (positive = CRYO2ICE recovery)')
print(f'  Parameters: rho_i mixture (FYI {RHO_I_FYI}, MYI {RHO_I_MYI}, cap {RHO_I_CAP}), '
      f'mean assigned {df["rho_i_assigned"].mean():.1f}; rho_s={RHO_S_CENTRAL}, '
      f'rho_w={RHO_W_CENTRAL}, delta={DELTA_BASELINE}')
df_sweep = pd.read_csv(SWEEP_CSV) if os.path.exists(SWEEP_CSV) else None
print(f'  delta sweep: {"loaded" if df_sweep is not None else "NOT FOUND (panel c skipped)"}')


# ===========================================================================
#  FIGURE 1: C2I vs CLIMATOLOGICAL SNOW DEPTH SCATTER BY SECTOR
#  (No changes from v3 — snow depths are independent of density params)
# ===========================================================================
print('\nGenerating Figure 1: Snow depth scatter plots (v8: Alex Comments 1, 2)...')

# v8 changes addressing Alex Comments 1 and 2:
#   - Log-scale colour normalisation (LogNorm) so low-density tails are visible
#   - Shared colour bar on the right with explicit "Point density (log scale)" label
#   - Panel letters (a)..(f) in upper-left, large, non-italic, bold
#   - Stats annotation moved to LOWER-RIGHT corner to avoid occluding upper data
#   - Increased aspect ratio so colour bar fits without crowding
from matplotlib.colors import LogNorm

fig1, axes1 = plt.subplots(2, 3, figsize=(12.5, 7.8), sharey='row', sharex='row')

panel_letters = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]
density_im = None  # capture last hist2d image for shared colorbar

for row, (product, prod_col, period_label) in enumerate([
    ('CCI v4.0', CCI_SNOW, '2022–2023'),
    ('Cryo-TEMPO', CT_SNOW, '2022–2025'),
]):
    for col_idx, sector in enumerate(SECTORS):
        ax = axes1[row, col_idx]
        mask = (df[SECTOR] == sector) & df[prod_col].notna() & df[C2I_SNOW].notna()
        x = df.loc[mask, C2I_SNOW].values
        y = df.loc[mask, prod_col].values

        plot_mask = (x > -0.5) & (x < 1.5) & (y > -0.5) & (y < 1.5)
        xp, yp = x[plot_mask], y[plot_mask]

        # v8 Alex Comment 2: explicit LogNorm density mapping for shared colorbar
        counts, _, _, im = ax.hist2d(
            xp, yp, bins=80, cmap='Blues', cmin=1,
            range=[[-0.2, 0.8], [-0.2, 0.8]],
            norm=LogNorm(vmin=1, vmax=200))
        density_im = im  # keep last image handle for colorbar

        ax.plot([-0.2, 0.8], [-0.2, 0.8], 'k--', lw=0.8, alpha=0.6, zorder=5)

        both = np.isfinite(x) & np.isfinite(y)
        n = both.sum()
        r, p = stats.pearsonr(x[both], y[both]) if n > 3 else (np.nan, np.nan)
        med_x = np.nanmedian(x[both])
        med_y = np.nanmedian(y[both])
        bias = med_y - med_x

        stats_text = (f'n = {n:,}\nr = {r:.3f}\n'
                      f'Median bias = {bias:+.3f} m\n'
                      f'C2I med = {med_x:.3f} m\nAMSR2 med = {med_y:.3f} m')
        # v8 Alex Comment 1: stats moved to LOWER-RIGHT to avoid covering data
        ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
                fontsize=8.5, va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='0.7', alpha=0.92))

        # v8 Alex Comment 1: panel letter large, non-italic, bold, upper-left
        ax.text(0.04, 0.96, panel_letters[row][col_idx],
                transform=ax.transAxes,
                fontsize=15, fontweight='bold', fontstyle='normal',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='0.5', alpha=0.92))

        ax.plot(med_x, med_y, 's', color=COL[sector], ms=8,
                markeredgecolor='black', markeredgewidth=0.7, zorder=6)

        if row == 1:
            ax.set_xlabel('C2I snow depth (m)', fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(f'{product} snow depth (m)', fontsize=11)
        ax.set_xlim(-0.15, 0.7)
        ax.set_ylim(-0.05, 0.7)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.tick_params(axis='both', labelsize=10)
        ax.set_aspect('equal')
        if row == 0:
            ax.set_title(f'{SECTOR_LABELS[sector]}',
                         fontweight='bold', color=COL[sector], fontsize=12)

    axes1[row, 0].annotate(
        f'{product} ({period_label})',
        xy=(-0.28, 0.5), xycoords='axes fraction',
        fontsize=12, fontweight='bold', rotation=90,
        ha='center', va='center')

# v8 Alex Comment 2: shared logarithmic colorbar on the right
fig1.tight_layout(rect=[0.03, 0, 0.92, 0.96])
cbar_ax = fig1.add_axes([0.93, 0.15, 0.018, 0.7])
cb = fig1.colorbar(density_im, cax=cbar_ax)
cb.set_label('Point density per bin (log scale)', fontsize=11)
cb.ax.tick_params(labelsize=10)

# v8: suptitle removed (product names are in the row labels and the caption)

for ext in ['pdf', 'png']:
    fig1.savefig(os.path.join(OUT_DIR, f'R3_Fig_C2_01_snow_scatter.{ext}'),
                 bbox_inches='tight', dpi=200)
print(f'  Saved R3_Fig_C2_01_snow_scatter.pdf/.png (v8)')
plt.close(fig1)


# ===========================================================================
#  FIGURE 2: Δhᵢ DISTRIBUTIONS BY SECTOR — VIOLIN + BOX
#  *** Alex Comment 6: ***
#    a. Parameter string moved from title to bottom annotation
#    b. Subplot titles changed from "snow swap" to explicit descriptions
# ===========================================================================
print('Generating Figure 2: Δhᵢ distributions (Option One framing)...')

fig2, (ax2a, ax2b, ax2c) = plt.subplots(1, 3, figsize=(16, 6.0),
                                         gridspec_kw={'width_ratios': [1, 1, 1.1]})
ax2b.sharey(ax2a)

for ax, (product, dhi_col, period_label, explicit_title) in zip(
    [ax2a, ax2b],
    [('CCI v4.0', 'Delta_h_i_CCI', '2022–2023',
      'CCI v4.0 baseline AMSR2 snow replaced\nby CRYO2ICE retrieval (2022–2023)'),
     ('Cryo-TEMPO', 'Delta_h_i_CT', '2022–2025',
      'Cryo-TEMPO baseline AMSR2 snow replaced\nby CRYO2ICE retrieval (2022–2025)')],
):
    data_by_sector = []
    labels = []
    colors = []

    for sector in SECTORS:
        mask = (df[SECTOR] == sector) & df[dhi_col].notna()
        vals = df.loc[mask, dhi_col].values
        vals_clipped = vals[(vals > -5) & (vals < 5)]
        data_by_sector.append(vals_clipped)
        labels.append(SECTOR_LABELS[sector])
        colors.append(COL[sector])

    positions = [1, 2, 3]

    parts = ax.violinplot(data_by_sector, positions=positions,
                          showmeans=False, showmedians=False,
                          showextrema=False)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
        pc.set_edgecolor(color)

    bp = ax.boxplot(data_by_sector, positions=positions, widths=0.25,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='0.4', linewidth=0.8),
                    capprops=dict(color='0.4', linewidth=0.8))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('0.3')

    for i, (sector, vals) in enumerate(zip(SECTORS, data_by_sector)):
        med = np.nanmedian(vals)
        mean = np.nanmean(vals)
        n = len(vals)
        ax.plot(positions[i], med, 'D', color='black', ms=5, zorder=5)
        ax.plot(positions[i], mean, 'o', color='white', ms=5, zorder=5,
                markeredgecolor='black', markeredgewidth=0.8)
        ax.text(positions[i] + 0.35, med,
                f'med={med:+.2f} m\nmean={mean:+.2f} m\nn={n:,}',
                fontsize=7.5, va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='0.7', alpha=0.85))

    ax.axhline(0, color='0.4', ls=':', lw=0.8, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(explicit_title, fontweight='bold', fontsize=10)
    ax.set_ylabel(
        '$\\Delta h_i = h_i^{\\rm Imp} - h_i^{\\rm Base}$ (m)'
        if ax == ax2a else '')
    # *** v7: y-axis range adjusted for predominantly positive values ***
    ax.set_ylim(-2, 4)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))

    # *** v7: explicit positive/negative labels for reader/reviewer clarity ***
    ax.text(0.02, 0.97,
            'Positive: CRYO2ICE recovers thickness\nabove AMSR2 baseline',
            transform=ax.transAxes, fontsize=8, va='top', ha='left',
            color='#B22222', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEFEF',
                      edgecolor='#B22222', alpha=0.7))
    ax.text(0.02, 0.03,
            'Negative: CRYO2ICE yields lower\nthickness than AMSR2 baseline',
            transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
            color='#1F4E79', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EFF4FB',
                      edgecolor='#1F4E79', alpha=0.7))

legend_elements = [
    Line2D([0], [0], marker='D', color='black', label='Median',
           markerfacecolor='black', markersize=5, linestyle='None'),
    Line2D([0], [0], marker='o', color='black', label='Mean',
           markerfacecolor='white', markersize=5, linestyle='None',
           markeredgewidth=0.8),
]
ax2b.legend(handles=legend_elements, loc='center right', framealpha=0.9)

# ---- panel (c): Delta h_i versus the delta used to produce the C2I snow ----
ax = ax2c
if df_sweep is not None:
    style = {'CCI': '-', 'CT': '--'}
    for product, plab in [('CCI', 'CCI v4.0'), ('CT', 'Cryo-TEMPO')]:
        for sector in SECTORS:
            sub = df_sweep[(df_sweep['product'] == product) & (df_sweep['sector'] == sector)].sort_values('delta')
            if len(sub) == 0:
                continue
            ax.plot(sub['delta'], sub['median_dh_i'], style[product], color=COL[sector], lw=1.8,
                    marker='o', ms=3.5, label=f'{plab}, {SECTOR_LABELS[sector]}')
    ax.axhline(0, color='0.4', ls=':', lw=0.8)
    ax.axvline(DELTA_BASELINE, color='black', ls='--', lw=1.0)
    ax.text(DELTA_BASELINE + 0.01, ax.get_ylim()[0] + 0.05, f'$\\delta$ = {DELTA_BASELINE}', fontsize=8, va='bottom')
    for dstar, lab, c in [(0.55, '$\\delta^*_{\\rm med}$', '#CC79A7'), (0.58, '$\\delta^*_{\\rm mean}$', '#E69F00')]:
        ax.axvline(dstar, color=c, ls='-.', lw=0.9)
    ax.set_xlabel('Penetration fraction $\\delta$ used for the CRYO2ICE snow')
    ax.set_ylabel('Median $\\Delta h_i$ (m)')
    ax.set_title('Dependence of the replacement effect\non the assumed penetration fraction', fontweight='bold', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='upper left', framealpha=0.9)
else:
    ax.text(0.5, 0.5, 'delta sweep not available', ha='center', va='center', transform=ax.transAxes)
    ax.set_axis_off()

for ax_, letter in zip([ax2a, ax2b, ax2c], 'abc'):
    ax_.text(0.02, 1.02, f'({letter})', transform=ax_.transAxes, fontsize=12, fontweight='bold', va='bottom')

fig2.suptitle(
    'Sea-ice thickness improvement when AMSR2 baseline snow is replaced by CRYO2ICE',
    fontsize=13, fontweight='bold', y=0.99)

fig2.text(0.5, 0.01, PARAM_STR, ha='center', va='bottom',
          fontsize=8.5, fontstyle='italic', color='0.4')

fig2.tight_layout(rect=[0, 0.04, 1, 0.94])

for ext in ['pdf', 'png']:
    fig2.savefig(os.path.join(OUT_DIR, f'R3_Fig_C2_02_dhi_distributions.{ext}'))
print(f'  Saved R3_Fig_C2_02_dhi_distributions.pdf/.png')
plt.close(fig2)


# ===========================================================================
#  FIGURE 3: PER-SECTOR PER-YEAR HEATMAP — 2×2 PANEL
#  *** EXPANDED: CCI (left column) + Cryo-TEMPO (right column) ***
#  Row 1: Median Δhᵢ from snow substitution
#  Row 2: AMSR2 climatological snow overestimation relative to C2I (%)
#  CCI covers 2022–2023 only; 2024–2025 cells are left empty (grey)
# ===========================================================================
print('Generating Figure 3: Per-sector per-year heatmap (2×2, CCI + CT)...')

CCI_YEARS = [2022, 2023]

# --- Build data matrices for both products ---
# CCI matrices (use full YEARS array but NaN for 2024–2025)
dhi_cci = np.full((len(SECTORS), len(YEARS)), np.nan)
overest_cci = np.full((len(SECTORS), len(YEARS)), np.nan)
n_cci = np.full((len(SECTORS), len(YEARS)), 0, dtype=int)

for i, sector in enumerate(SECTORS):
    for j, year in enumerate(YEARS):
        if year not in CCI_YEARS:
            continue
        mask = ((df[SECTOR] == sector) & (df['year'] == year) &
                df['Delta_h_i_CCI'].notna())
        n = mask.sum()
        n_cci[i, j] = n
        if n > 0:
            dhi_cci[i, j] = df.loc[mask, 'Delta_h_i_CCI'].median()
            c2i_med = df.loc[mask, C2I_SNOW].median()
            cci_med = df.loc[mask, CCI_SNOW].median()
            if c2i_med > 0:
                overest_cci[i, j] = 100 * (cci_med - c2i_med) / c2i_med

# Cryo-TEMPO matrices (full 2022–2025)
dhi_ct = np.full((len(SECTORS), len(YEARS)), np.nan)
overest_ct = np.full((len(SECTORS), len(YEARS)), np.nan)
n_ct = np.full((len(SECTORS), len(YEARS)), 0, dtype=int)

for i, sector in enumerate(SECTORS):
    for j, year in enumerate(YEARS):
        mask = ((df[SECTOR] == sector) & (df['year'] == year) &
                df['Delta_h_i_CT'].notna())
        n = mask.sum()
        n_ct[i, j] = n
        if n > 0:
            dhi_ct[i, j] = df.loc[mask, 'Delta_h_i_CT'].median()
            c2i_med = df.loc[mask, C2I_SNOW].median()
            ct_med = df.loc[mask, CT_SNOW].median()
            if c2i_med > 0:
                overest_ct[i, j] = 100 * (ct_med - c2i_med) / c2i_med

sector_labels_list = [SECTOR_LABELS[s] for s in SECTORS]
year_labels = [str(y) for y in YEARS]

# --- Shared colorbar ranges ---
all_dhi = np.concatenate([dhi_cci[np.isfinite(dhi_cci)],
                          dhi_ct[np.isfinite(dhi_ct)]])
dhi_vmin = np.min(all_dhi) if len(all_dhi) > 0 else -1.5
dhi_vmax = max(np.max(all_dhi), 0.01) if len(all_dhi) > 0 else 0.5

all_overest = np.concatenate([overest_cci[np.isfinite(overest_cci)],
                              overest_ct[np.isfinite(overest_ct)]])
overest_vmax = max(np.max(all_overest), 1) if len(all_overest) > 0 else 200

# --- Create 2×2 figure ---
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 8.0))

# *** v7: Common diverging norm for Delta h_i panels (white at 0) ***
# Under Option One framing, values are predominantly positive (CRYO2ICE
# recovery above AMSR2 baseline). Norm is centred at zero with the
# positive range extending further than the negative range.
norm_dhi = TwoSlopeNorm(vmin=min(dhi_vmin, -0.1), vcenter=0.0,
                         vmax=max(dhi_vmax, 0.5))

# --- Helper function for heatmap plotting ---
def plot_heatmap(ax, matrix, n_matrix, cmap, norm_or_vmin, vmax=None,
                 fmt_func=None, title='', cb_label='', is_nan_grey=True,
                 value_unit='m'):
    """Plot a sector x year heatmap with cell annotations."""
    if vmax is not None:
        im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                       vmin=norm_or_vmin, vmax=vmax)
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', norm=norm_or_vmin)

    if is_nan_grey:
        ax.set_facecolor('#F0F0F0')

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            n = n_matrix[i, j]
            if np.isfinite(val):
                text_color = fmt_func(val) if fmt_func else 'black'
                if value_unit == '%':
                    ax.text(j, i, f'+{val:.0f}%\n(n={n:,})',
                            ha='center', va='center', fontsize=8,
                            fontweight='bold', color=text_color)
                else:
                    # *** v7: explicit sign for Delta h_i values ***
                    ax.text(j, i, f'{val:+.2f} m\n(n={n:,})',
                            ha='center', va='center', fontsize=8,
                            fontweight='bold', color=text_color)
            elif n == 0 and is_nan_grey:
                ax.text(j, i, 'N/A', ha='center', va='center',
                        fontsize=8, color='#AAAAAA', fontstyle='italic')

    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels(year_labels)
    ax.set_yticks(range(len(SECTORS)))
    ax.set_yticklabels(sector_labels_list)
    ax.set_title(title, fontweight='bold', fontsize=10, pad=8)

    cb = fig3.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(cb_label, fontsize=9)
    return im, cb

# *** v7: Inverted diverging colormap — 'RdBu' (NOT '_r') so that:
#         warm RED = positive values = CRYO2ICE recovery above AMSR2 baseline
#         cool BLUE = negative values = rare cases of CRYO2ICE below AMSR2
# (Previously 'RdBu_r' had the inverse mapping which would mislead readers
#  under the new positive-recovery direction.)
DHI_CMAP = 'RdBu'

# --- Panel (a): CCI Delta h_i ---
im_a, cb_a = plot_heatmap(
    axes3[0, 0], dhi_cci, n_cci, DHI_CMAP, norm_dhi,
    fmt_func=lambda v: 'white' if abs(v) > 0.6 else 'black',
    title='(a) CCI v4.0: Median $\\Delta h_i = h_i^{\\rm Imp} - h_i^{\\rm Base}$ (m)\n(2022–2023)',
    cb_label='$\\Delta h_i$ (m)\n(+ = CRYO2ICE recovery,\n– = CRYO2ICE reduction)',
    value_unit='m')

# --- Panel (b): Cryo-TEMPO Delta h_i ---
im_b, cb_b = plot_heatmap(
    axes3[0, 1], dhi_ct, n_ct, DHI_CMAP, norm_dhi,
    fmt_func=lambda v: 'white' if abs(v) > 0.6 else 'black',
    title='(b) Cryo-TEMPO: Median $\\Delta h_i = h_i^{\\rm Imp} - h_i^{\\rm Base}$ (m)\n(2022–2025)',
    cb_label='$\\Delta h_i$ (m)\n(+ = CRYO2ICE recovery,\n– = CRYO2ICE reduction)',
    value_unit='m')

# --- Panel (c): CCI AMSR2 excess relative to CRYO2ICE ---
im_c, cb_c = plot_heatmap(
    axes3[1, 0], overest_cci, n_cci, 'YlOrRd', 0, vmax=200,
    fmt_func=lambda v: 'white' if v > 150 else 'black',
    title='(c) CCI v4.0: AMSR2 excess relative to CRYO2ICE (%)\n(2022–2023)',
    cb_label='AMSR2 excess vs CRYO2ICE (%)',
    value_unit='%')

# --- Panel (d): Cryo-TEMPO AMSR2 excess relative to CRYO2ICE ---
im_d, cb_d = plot_heatmap(
    axes3[1, 1], overest_ct, n_ct, 'YlOrRd', 0, vmax=200,
    fmt_func=lambda v: 'white' if v > 150 else 'black',
    title='(d) Cryo-TEMPO: AMSR2 excess relative to CRYO2ICE (%)\n(2022–2025)',
    cb_label='AMSR2 excess vs CRYO2ICE (%)',
    value_unit='%')

# *** v7: Add explicit interpretive note below the figure ***
interp_note = (
    'Top row (panels a, b): positive values indicate that replacing the '
    'AMSR2 climatology with the CRYO2ICE retrieval recovers sea-ice thickness '
    'above the published-product baseline. '
    'Negative values indicate the rare cases where CRYO2ICE yields lower thickness than the AMSR2 baseline.'
)

# --- Title and parameter annotation ---
fig3.suptitle(
    'Sea-ice thickness improvement when AMSR2 baseline snow is replaced by CRYO2ICE',
    fontsize=12, fontweight='bold', y=1.01)

fig3.text(0.5, -0.02, interp_note,
          ha='center', va='top',
          fontsize=8.5, fontstyle='italic', color='#333333',
          wrap=True)
fig3.text(0.5, -0.06, PARAM_STR, ha='center', va='top',
          fontsize=8, fontstyle='italic', color='0.4')

fig3.tight_layout(rect=[0, 0.04, 1, 0.97])

for ext in ['pdf', 'png']:
    fig3.savefig(os.path.join(OUT_DIR, f'R3_Fig_C2_03_sector_year_heatmap.{ext}'))
print(f'  Saved R3_Fig_C2_03_sector_year_heatmap.pdf/.png')
plt.close(fig3)


# ===========================================================================
#  SUMMARY
# ===========================================================================
print(f'\n{"="*70}')
print(f'ALL FIGURES SAVED TO: {OUT_DIR}')
print(f'{"="*70}')
print(f'  R3_Fig_C2_01_snow_scatter.pdf/.png')
print(f'  R3_Fig_C2_02_dhi_distributions.pdf/.png')
print(f'  R3_Fig_C2_03_sector_year_heatmap.pdf/.png  (2x2: CCI + CT)')
print(f'\n  Option One framing: AMSR2 baseline -> CRYO2ICE replacement')
print(f'  Delta h_i = h_i_Imp - h_i_Base  (positive = CRYO2ICE recovery)')
print(f'  Regional parameters: rho_i FYI={RHO_I_FYI}, MYI={RHO_I_MYI}, '
      f'cap={RHO_I_CAP}; rho_s={RHO_S_CENTRAL}, '
      f'rho_w={RHO_W_CENTRAL}, delta={DELTA_BASELINE}')
print('Done.')

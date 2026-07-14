#!/usr/bin/env python3
"""
R3_Step09_snow_swap_v5.py
==========================
Controlled snow-thickness substitution experiment under Option One framing.

*** v5 CHANGES (1 May 2026, Alex Option One framing): ***
  - Reference baseline INVERTED: AMSR2 climatology (CCI v4.0 and Cryo-TEMPO)
    is now the reference baseline. CRYO2ICE is the replacement that
    quantifies improvement upon the published operational products.
  - Sign convention INVERTED: Delta h_i = h_i_Imp - h_i_Base
    Positive values = CRYO2ICE recovers thickness above AMSR2 baseline
    Negative values = CRYO2ICE yields lower thickness than AMSR2 (rare)
  - Column renames:
      h_i_O        -> h_i_Base_CCI / h_i_Base_CT (AMSR2-informed baselines)
      h_i_S_CCI    -> h_i_Imp_CCI (CCI replaced by CRYO2ICE)
      h_i_S_CT     -> h_i_Imp_CT  (Cryo-TEMPO replaced by CRYO2ICE)
      dh_i_snow_*  -> Delta_h_i_CCI / Delta_h_i_CT (inverted sign)
  - Scenario renames:
      Scenario Obs       -> Scenarios Base_CCI and Base_CT
      Scenario Sub_CCI   -> Scenario Imp_CCI
      Scenario Sub_CT    -> Scenario Imp_CT

v4 features preserved:
  - Per-segment regional rho_i Alexandrov parameterisation
      Western Weddell (MYI): 882.0 kg/m^3
      Eastern Weddell (FYI): 916.7 kg/m^3
      Ross (FYI):            916.7 kg/m^3
  - Soft cap at 940 kg/m^3
  - rho_s = 330 kg/m^3 (Kacimi & Kwok 2020)

Note: The h_i_Base_CCI and h_i_Base_CT scenarios both use the same
freeboards from the C2I matchup, but with the CCI and Cryo-TEMPO
climatological snow respectively. They differ in sample size because
CCI L2P only covers 2022-2023 while Cryo-TEMPO covers 2022-2025.

Physics (unchanged):
  h_i = (rho_w * h_f^IS2 - (rho_w - rho_s) * h_s) / (rho_w - rho_i)

Input:  R3_L2P_collocation_all_winters.csv (from Step 08c)
Output: R3_SIT_snow_swap_all_winters_v5.csv
        R3_SIT_snow_swap_summary_v5.txt

Run: source ~/cryo2ice_env/bin/activate && python R3_Step09_snow_swap_v5.py
"""

import numpy as np
import pandas as pd
import os, sys
import time as timer
from scipy import stats

# ===========================================================================
#  IMPORT SHARED CONFIGURATION (Component 1 parameters)
# ===========================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import (
    RHO_S_CENTRAL, RHO_W_CENTRAL,
)

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
IN_CSV  = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation/R3_L2P_collocation_all_winters.csv'
OUT_DIR = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation'

# Physical constants from R3_config_v4.py (cross-component consistency)
RHO_W = RHO_W_CENTRAL              # 1024.0 kg/m^3 (fixed)
RHO_S = RHO_S_CENTRAL              # 330.0 kg/m^3 (Kacimi & Kwok 2020)

# Regional rho_i Alexandrov parameterisation (per-segment)
RHO_I_FYI = 916.7                  # Eastern Weddell, Ross — Alexandrov FYI
RHO_I_MYI = 882.0                  # Western Weddell — Alexandrov MYI
RHO_I_CAP = 940.0                  # Soft upper cap (Timco & Frederking 1996)

# Sub-sector to rho_i mapping
SECTOR_TO_RHO_I = {
    'Western_Weddell': RHO_I_MYI,
    'Eastern_Weddell': RHO_I_FYI,
    'Ross':            RHO_I_FYI,
}

# Column names from the collocation CSV
IS2_FB_COL   = 'is2_fb_wm'
CS2_RFB_COL  = 'cs2_rfb'
C2I_SNOW_COL = 'snow_thickness'
SECTOR_COL   = 'sector'

CCI_SNOW_COL = 'cci_snow_depth'
CCI_SIT_COL  = 'cci_sea_ice_thickness'
CT_SNOW_COL  = 'ct_snow_depth'

SECTORS = ['Western_Weddell', 'Eastern_Weddell', 'Ross']
YEARS   = [2022, 2023, 2024, 2025]


# ===========================================================================
#  PHYSICAL FUNCTIONS
# ===========================================================================

def compute_sit_per_segment(h_f_is2, h_s, rho_i_array,
                              rho_w=RHO_W, rho_s=RHO_S):
    """
    Hydrostatic sea-ice thickness with per-segment rho_i (Kacimi & Kwok 2020).
    h_i = (rho_w * h_f^IS2 - (rho_w - rho_s) * h_s) / (rho_w - rho_i)
    """
    assert np.all(rho_i_array <= RHO_I_CAP), \
        f'FATAL: some rho_i values exceed the soft cap of {RHO_I_CAP} kg/m^3'

    numerator   = rho_w * h_f_is2 - (rho_w - rho_s) * h_s
    denominator = rho_w - rho_i_array

    return numerator / denominator


def assign_rho_i_per_segment(df, sector_col=SECTOR_COL):
    """Map each segment to its regional rho_i value based on sub-sector."""
    rho_i_array = np.full(len(df), np.nan)

    for sector, rho_i_val in SECTOR_TO_RHO_I.items():
        mask = (df[sector_col] == sector).values
        rho_i_array[mask] = rho_i_val

    n_assigned = np.isfinite(rho_i_array).sum()
    n_total = len(df)
    if n_assigned < n_total:
        print(f'  WARNING: {n_total - n_assigned} segments have unrecognised '
              f'sector and were not assigned a rho_i value')

    return rho_i_array


# ===========================================================================
#  MAIN
# ===========================================================================
if __name__ == '__main__':
    t0 = timer.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    summary_lines = []
    def log(msg):
        print(msg)
        summary_lines.append(msg)

    # ------------------------------------------------------------------
    # 1. Load collocated data
    # ------------------------------------------------------------------
    df = pd.read_csv(IN_CSV)
    log(f'Loaded {len(df):,} rows from collocation CSV')

    required = [IS2_FB_COL, CS2_RFB_COL, C2I_SNOW_COL, SECTOR_COL]
    for col in required:
        assert col in df.columns, f'Missing required column: {col}'

    df['year'] = (df['year_month'] // 100).astype(int)

    # Assign per-segment rho_i values based on sub-sector
    rho_i_array = assign_rho_i_per_segment(df)
    df['rho_i_assigned'] = rho_i_array

    # Report constants and regional assignment
    log(f'\n{"="*70}')
    log(f'PHYSICAL PARAMETERS — v5 OPTION ONE FRAMING')
    log(f'{"="*70}')
    log(f'  *** Reference baseline: AMSR2 climatology (CCI v4.0 / Cryo-TEMPO)')
    log(f'  *** Improvement scenario: AMSR2 snow replaced by CRYO2ICE')
    log(f'  *** Delta h_i = h_i_Imp - h_i_Base (positive = CRYO2ICE recovers')
    log(f'      thickness above AMSR2 baseline)')
    log(f'')
    log(f'  rho_w  = {RHO_W:.1f} kg/m^3 (fixed)')
    log(f'  rho_s  = {RHO_S:.1f} kg/m^3 (Kacimi & Kwok 2020)')
    log(f'  rho_i regional (per-segment):')
    log(f'    FYI (Eastern Weddell, Ross): {RHO_I_FYI:.1f} kg/m^3 '
        f'(Alexandrov 2010 FYI)')
    log(f'    MYI (Western Weddell):       {RHO_I_MYI:.1f} kg/m^3 '
        f'(Alexandrov 2010 MYI)')
    log(f'    Soft cap:                    {RHO_I_CAP:.1f} kg/m^3 '
        f'(Timco & Frederking 1996)')

    # Per-segment rho_i assignment summary
    n_fyi = (rho_i_array == RHO_I_FYI).sum()
    n_myi = (rho_i_array == RHO_I_MYI).sum()
    log(f'\n  Per-segment rho_i assignment:')
    log(f'    Segments at FYI ({RHO_I_FYI:.1f}): {n_fyi:,}')
    log(f'    Segments at MYI ({RHO_I_MYI:.1f}): {n_myi:,}')

    # Weddell aggregate composition
    n_e_weddell = (df[SECTOR_COL] == 'Eastern_Weddell').sum()
    n_w_weddell = (df[SECTOR_COL] == 'Western_Weddell').sum()
    n_weddell_total = n_e_weddell + n_w_weddell
    if n_weddell_total > 0:
        log(f'\n  Weddell Sea aggregate composition:')
        log(f'    Eastern Weddell (FYI): {n_e_weddell:>6,d} segments '
            f'({100*n_e_weddell/n_weddell_total:>5.1f}%)')
        log(f'    Western Weddell (MYI): {n_w_weddell:>6,d} segments '
            f'({100*n_w_weddell/n_weddell_total:>5.1f}%)')

    # ------------------------------------------------------------------
    # 2. Scenario Base_CCI: CCI v4.0 published baseline (AMSR2 snow)
    # ------------------------------------------------------------------
    log(f'\n{"="*70}')
    log(f'SCENARIO Base_CCI: CCI v4.0 PUBLISHED BASELINE (AMSR2 snow)')
    log(f'{"="*70}')

    h_f_is2 = df[IS2_FB_COL].values

    has_cci_snow = df[CCI_SNOW_COL].notna()
    df['h_i_Base_CCI'] = np.nan
    if has_cci_snow.sum() > 0:
        idx_cci = has_cci_snow.values
        h_s_cci = df.loc[idx_cci, CCI_SNOW_COL].values
        df.loc[idx_cci, 'h_i_Base_CCI'] = compute_sit_per_segment(
            h_f_is2[idx_cci], h_s_cci, rho_i_array[idx_cci])
        log(f'  Valid segments: {has_cci_snow.sum():,}')
        log(f'  h_i_Base_CCI - median: '
            f'{df.loc[has_cci_snow, "h_i_Base_CCI"].median():.3f} m, '
            f'mean: {df.loc[has_cci_snow, "h_i_Base_CCI"].mean():.3f} m')

    # ------------------------------------------------------------------
    # 3. Scenario Base_CT: Cryo-TEMPO published baseline (AMSR2 snow)
    # ------------------------------------------------------------------
    log(f'\n{"="*70}')
    log(f'SCENARIO Base_CT: CRYO-TEMPO PUBLISHED BASELINE (AMSR2 snow)')
    log(f'{"="*70}')

    has_ct_snow = df[CT_SNOW_COL].notna()
    df['h_i_Base_CT'] = np.nan
    if has_ct_snow.sum() > 0:
        idx_ct = has_ct_snow.values
        h_s_ct = df.loc[idx_ct, CT_SNOW_COL].values
        df.loc[idx_ct, 'h_i_Base_CT'] = compute_sit_per_segment(
            h_f_is2[idx_ct], h_s_ct, rho_i_array[idx_ct])
        log(f'  Valid segments: {has_ct_snow.sum():,}')
        log(f'  h_i_Base_CT - median: '
            f'{df.loc[has_ct_snow, "h_i_Base_CT"].median():.3f} m, '
            f'mean: {df.loc[has_ct_snow, "h_i_Base_CT"].mean():.3f} m')

    # ------------------------------------------------------------------
    # 4. Scenario Imp_CCI: CCI baseline AMSR2 snow replaced by CRYO2ICE
    # ------------------------------------------------------------------
    log(f'\n{"="*70}')
    log(f'SCENARIO Imp_CCI: CCI BASELINE WITH AMSR2 REPLACED BY CRYO2ICE')
    log(f'{"="*70}')

    h_s_c2i = df[C2I_SNOW_COL].values

    df['h_i_Imp_CCI'] = np.nan
    # Compute on the same segment population as Base_CCI for comparability
    idx_imp_cci = has_cci_snow & df[C2I_SNOW_COL].notna()
    df.loc[idx_imp_cci, 'h_i_Imp_CCI'] = compute_sit_per_segment(
        h_f_is2[idx_imp_cci.values],
        h_s_c2i[idx_imp_cci.values],
        rho_i_array[idx_imp_cci.values])
    log(f'  Valid segments: {idx_imp_cci.sum():,}')
    log(f'  h_i_Imp_CCI - median: '
        f'{df.loc[idx_imp_cci, "h_i_Imp_CCI"].median():.3f} m, '
        f'mean: {df.loc[idx_imp_cci, "h_i_Imp_CCI"].mean():.3f} m')

    # ------------------------------------------------------------------
    # 5. Scenario Imp_CT: Cryo-TEMPO baseline AMSR2 snow replaced by CRYO2ICE
    # ------------------------------------------------------------------
    log(f'\n{"="*70}')
    log(f'SCENARIO Imp_CT: CRYO-TEMPO BASELINE WITH AMSR2 REPLACED BY CRYO2ICE')
    log(f'{"="*70}')

    df['h_i_Imp_CT'] = np.nan
    idx_imp_ct = has_ct_snow & df[C2I_SNOW_COL].notna()
    df.loc[idx_imp_ct, 'h_i_Imp_CT'] = compute_sit_per_segment(
        h_f_is2[idx_imp_ct.values],
        h_s_c2i[idx_imp_ct.values],
        rho_i_array[idx_imp_ct.values])
    log(f'  Valid segments: {idx_imp_ct.sum():,}')
    log(f'  h_i_Imp_CT - median: '
        f'{df.loc[idx_imp_ct, "h_i_Imp_CT"].median():.3f} m, '
        f'mean: {df.loc[idx_imp_ct, "h_i_Imp_CT"].mean():.3f} m')

    # ------------------------------------------------------------------
    # 6. Per-sector breakdown of Base scenarios
    # ------------------------------------------------------------------
    log(f'\n  Per-sector breakdown (Scenarios Base):')
    for sector in SECTORS:
        for product, base_col in [('CCI', 'h_i_Base_CCI'),
                                    ('CT',  'h_i_Base_CT')]:
            mask = df[base_col].notna() & (df[SECTOR_COL] == sector)
            if mask.sum() > 0:
                log(f'    {sector:<22s} {product:<4s} n={mask.sum():>6,d}  '
                    f'rho_i={SECTOR_TO_RHO_I[sector]:.1f}  '
                    f'h_i median={df.loc[mask, base_col].median():.3f} m  '
                    f'mean={df.loc[mask, base_col].mean():.3f} m')

    # ------------------------------------------------------------------
    # 7. Snow depth comparison (AMSR2 baseline vs CRYO2ICE replacement)
    # ------------------------------------------------------------------
    log(f'\n{"="*70}')
    log(f'SNOW DEPTH COMPARISON: AMSR2 BASELINE vs CRYO2ICE REPLACEMENT')
    log(f'{"="*70}')

    for product, prod_snow_col, mask_valid in [
        ('CCI v4.0',   CCI_SNOW_COL, idx_imp_cci),
        ('Cryo-TEMPO', CT_SNOW_COL,  idx_imp_ct),
    ]:
        n = mask_valid.sum()
        if n == 0:
            continue

        c2i_s  = df.loc[mask_valid, C2I_SNOW_COL].values
        prod_s = df.loc[mask_valid, prod_snow_col].values

        # Difference: baseline (AMSR2) minus replacement (CRYO2ICE)
        diff = prod_s - c2i_s
        ratio = prod_s / np.where(c2i_s > 0, c2i_s, np.nan)

        log(f'\n  {product} (n={n:,}):')
        log(f'    AMSR2 baseline snow - median: {np.nanmedian(prod_s):.3f} m, '
            f'mean: {np.nanmean(prod_s):.3f} m')
        log(f'    CRYO2ICE replacement snow - median: '
            f'{np.nanmedian(c2i_s):.3f} m, mean: {np.nanmean(c2i_s):.3f} m')
        log(f'    Difference (AMSR2 - CRYO2ICE) - median: '
            f'{np.nanmedian(diff):.3f} m, mean: {np.nanmean(diff):.3f} m')
        log(f'    AMSR2-to-CRYO2ICE ratio - '
            f'median: {np.nanmedian(ratio):.2f}x, '
            f'mean: {np.nanmean(ratio):.2f}x')
        log(f'    (ratio > 1 indicates AMSR2 baseline is deeper than '
            f'CRYO2ICE replacement)')

        both = np.isfinite(c2i_s) & np.isfinite(prod_s)
        if both.sum() > 3:
            r_pool, p_pool = stats.pearsonr(c2i_s[both], prod_s[both])
            log(f'    Pooled Pearson r: {r_pool:.4f}  (p = {p_pool:.2e})')

        log(f'    Within-sector correlations:')
        for sector in SECTORS:
            mask_s = mask_valid & (df[SECTOR_COL] == sector)
            n_s = mask_s.sum()
            if n_s < 3:
                continue
            c2i_sec  = df.loc[mask_s, C2I_SNOW_COL].values
            prod_sec = df.loc[mask_s, prod_snow_col].values
            both_s = np.isfinite(c2i_sec) & np.isfinite(prod_sec)
            if both_s.sum() > 3:
                r_s, p_s = stats.pearsonr(c2i_sec[both_s], prod_sec[both_s])
                med_c2i = np.nanmedian(c2i_sec)
                amsr2_excess_pct = (
                    100 * (np.nanmedian(prod_sec) - med_c2i) / med_c2i
                    if med_c2i > 0 else np.nan
                )
                log(f'      {sector:<22s} n={n_s:>6,d}  r={r_s:>7.4f}  '
                    f'AMSR2 excess vs CRYO2ICE={amsr2_excess_pct:>+7.1f}%')

    # ------------------------------------------------------------------
    # 8. Sea-ice thickness improvement (Delta h_i = Imp - Base, INVERTED)
    # ------------------------------------------------------------------
    log(f'\n{"="*70}')
    log(f'SEA-ICE THICKNESS IMPROVEMENT (Delta h_i = h_i_Imp - h_i_Base)')
    log(f'  *** Positive = CRYO2ICE recovers thickness above AMSR2 baseline')
    log(f'  *** Negative = CRYO2ICE yields lower thickness than AMSR2 (rare)')
    log(f'{"="*70}')

    # --- CCI improvement (2022-2023) ---
    if idx_imp_cci.sum() > 0:
        dhi_cci = (df.loc[idx_imp_cci, 'h_i_Imp_CCI'].values -
                    df.loc[idx_imp_cci, 'h_i_Base_CCI'].values)
        df.loc[idx_imp_cci, 'Delta_h_i_CCI'] = dhi_cci

        pct_cci = 100 * dhi_cci / np.where(
            df.loc[idx_imp_cci, 'h_i_Base_CCI'].values != 0,
            df.loc[idx_imp_cci, 'h_i_Base_CCI'].values, np.nan)

        log(f'\n  CCI IMPROVEMENT (n={idx_imp_cci.sum():,}, 2022-2023):')
        log(f'    Delta h_i - median: {np.nanmedian(dhi_cci):>+.3f} m, '
            f'mean: {np.nanmean(dhi_cci):>+.3f} m')
        log(f'    |Delta h_i| - median: {np.nanmedian(np.abs(dhi_cci)):.3f} m, '
            f'mean: {np.nanmean(np.abs(dhi_cci)):.3f} m')
        log(f'    Relative improvement vs Base - '
            f'median: {np.nanmedian(pct_cci):>+.1f}%, '
            f'mean: {np.nanmean(pct_cci):>+.1f}%')

    # --- Cryo-TEMPO improvement (2022-2025) ---
    if idx_imp_ct.sum() > 0:
        dhi_ct = (df.loc[idx_imp_ct, 'h_i_Imp_CT'].values -
                   df.loc[idx_imp_ct, 'h_i_Base_CT'].values)
        df.loc[idx_imp_ct, 'Delta_h_i_CT'] = dhi_ct

        pct_ct = 100 * dhi_ct / np.where(
            df.loc[idx_imp_ct, 'h_i_Base_CT'].values != 0,
            df.loc[idx_imp_ct, 'h_i_Base_CT'].values, np.nan)

        log(f'\n  CRYO-TEMPO IMPROVEMENT (n={idx_imp_ct.sum():,}, 2022-2025):')
        log(f'    Delta h_i - median: {np.nanmedian(dhi_ct):>+.3f} m, '
            f'mean: {np.nanmean(dhi_ct):>+.3f} m')
        log(f'    |Delta h_i| - median: {np.nanmedian(np.abs(dhi_ct)):.3f} m, '
            f'mean: {np.nanmean(np.abs(dhi_ct)):.3f} m')
        log(f'    Relative improvement vs Base - '
            f'median: {np.nanmedian(pct_ct):>+.1f}%, '
            f'mean: {np.nanmean(pct_ct):>+.1f}%')

    # ------------------------------------------------------------------
    # 9. Per-sector, per-year breakdown
    # ------------------------------------------------------------------
    log(f'\n{"="*70}')
    log(f'PER-SECTOR IMPROVEMENT BREAKDOWN')
    log(f'{"="*70}')

    # --- CCI-based (2022-2023) ---
    log(f'\n  CCI IMPROVEMENT (2022-2023):')
    log(f'  {"Sector":<22s} {"Year":>5s} {"n":>7s} '
        f'{"AMSR2_h_s":>10s} {"C2I_h_s":>9s} {"AMSR2_xs%":>10s} '
        f'{"h_i_Base":>9s} {"h_i_Imp":>9s} {"Delta":>9s} {"Delta%":>9s}')
    log(f'  {"-"*108}')

    for sector in SECTORS:
        for year in [2022, 2023]:
            mask = (idx_imp_cci &
                    (df[SECTOR_COL] == sector) &
                    (df['year'] == year))
            n = mask.sum()
            if n == 0:
                continue

            cci_med = df.loc[mask, CCI_SNOW_COL].median()
            c2i_med = df.loc[mask, C2I_SNOW_COL].median()
            amsr2_xs = (100 * (cci_med - c2i_med) / c2i_med
                         if c2i_med > 0 else np.nan)
            hi_B    = df.loc[mask, 'h_i_Base_CCI'].median()
            hi_I    = df.loc[mask, 'h_i_Imp_CCI'].median()
            dhi     = df.loc[mask, 'Delta_h_i_CCI'].median()
            dhi_pct = 100 * dhi / hi_B if hi_B != 0 else np.nan

            log(f'  {sector:<22s} {year:>5d} {n:>7,d} '
                f'{cci_med:>10.3f} {c2i_med:>9.3f} {amsr2_xs:>+10.1f} '
                f'{hi_B:>9.3f} {hi_I:>9.3f} {dhi:>+9.3f} {dhi_pct:>+9.1f}')

    log(f'\n  CCI sector summary (pooled across 2022-2023):')
    for sector in SECTORS:
        mask = idx_imp_cci & (df[SECTOR_COL] == sector)
        n = mask.sum()
        if n == 0:
            continue
        cci_med = df.loc[mask, CCI_SNOW_COL].median()
        c2i_med = df.loc[mask, C2I_SNOW_COL].median()
        amsr2_xs = (100 * (cci_med - c2i_med) / c2i_med
                     if c2i_med > 0 else np.nan)
        dhi = df.loc[mask, 'Delta_h_i_CCI'].median()
        log(f'    {sector:<22s} n={n:>6,d}  '
            f'rho_i={SECTOR_TO_RHO_I[sector]:.1f}  '
            f'Delta h_i={dhi:>+.3f} m  '
            f'AMSR2 excess vs CRYO2ICE={amsr2_xs:>+.1f}%')

    # --- Cryo-TEMPO-based (2022-2025) ---
    log(f'\n  CRYO-TEMPO IMPROVEMENT (2022-2025):')
    log(f'  {"Sector":<22s} {"Year":>5s} {"n":>7s} '
        f'{"AMSR2_h_s":>10s} {"C2I_h_s":>9s} {"AMSR2_xs%":>10s} '
        f'{"h_i_Base":>9s} {"h_i_Imp":>9s} {"Delta":>9s} {"Delta%":>9s}')
    log(f'  {"-"*108}')

    for sector in SECTORS:
        for year in YEARS:
            mask = (idx_imp_ct &
                    (df[SECTOR_COL] == sector) &
                    (df['year'] == year))
            n = mask.sum()
            if n == 0:
                continue

            ct_med  = df.loc[mask, CT_SNOW_COL].median()
            c2i_med = df.loc[mask, C2I_SNOW_COL].median()
            amsr2_xs = (100 * (ct_med - c2i_med) / c2i_med
                         if c2i_med > 0 else np.nan)
            hi_B    = df.loc[mask, 'h_i_Base_CT'].median()
            hi_I    = df.loc[mask, 'h_i_Imp_CT'].median()
            dhi     = df.loc[mask, 'Delta_h_i_CT'].median()
            dhi_pct = 100 * dhi / hi_B if hi_B != 0 else np.nan

            log(f'  {sector:<22s} {year:>5d} {n:>7,d} '
                f'{ct_med:>10.3f} {c2i_med:>9.3f} {amsr2_xs:>+10.1f} '
                f'{hi_B:>9.3f} {hi_I:>9.3f} {dhi:>+9.3f} {dhi_pct:>+9.1f}')

    log(f'\n  Cryo-TEMPO sector summary (pooled across 2022-2025):')
    for sector in SECTORS:
        mask = idx_imp_ct & (df[SECTOR_COL] == sector)
        n = mask.sum()
        if n == 0:
            continue
        ct_med  = df.loc[mask, CT_SNOW_COL].median()
        c2i_med = df.loc[mask, C2I_SNOW_COL].median()
        amsr2_xs = (100 * (ct_med - c2i_med) / c2i_med
                     if c2i_med > 0 else np.nan)
        dhi = df.loc[mask, 'Delta_h_i_CT'].median()
        log(f'    {sector:<22s} n={n:>6,d}  '
            f'rho_i={SECTOR_TO_RHO_I[sector]:.1f}  '
            f'Delta h_i={dhi:>+.3f} m  '
            f'AMSR2 excess vs CRYO2ICE={amsr2_xs:>+.1f}%')

    # ------------------------------------------------------------------
    # 10. Weddell aggregate breakdown (sample-weighted union)
    # ------------------------------------------------------------------
    log(f'\n{"="*70}')
    log(f'WEDDELL AGGREGATE: SAMPLE-WEIGHTED UNION OF E + W WEDDELL')
    log(f'{"="*70}')

    mask_weddell_agg_cci = (idx_imp_cci & df[SECTOR_COL].isin(
        ['Eastern_Weddell', 'Western_Weddell']))
    mask_weddell_agg_ct = (idx_imp_ct & df[SECTOR_COL].isin(
        ['Eastern_Weddell', 'Western_Weddell']))

    if mask_weddell_agg_cci.sum() > 0:
        cci_med = df.loc[mask_weddell_agg_cci, CCI_SNOW_COL].median()
        c2i_med = df.loc[mask_weddell_agg_cci, C2I_SNOW_COL].median()
        amsr2_xs = (100 * (cci_med - c2i_med) / c2i_med
                     if c2i_med > 0 else np.nan)
        dhi = df.loc[mask_weddell_agg_cci, 'Delta_h_i_CCI'].median()
        log(f'  CCI Weddell aggregate     '
            f'n={mask_weddell_agg_cci.sum():>6,d}  '
            f'Delta h_i={dhi:>+.3f} m  '
            f'AMSR2 excess vs CRYO2ICE={amsr2_xs:>+.1f}%')

    if mask_weddell_agg_ct.sum() > 0:
        ct_med  = df.loc[mask_weddell_agg_ct, CT_SNOW_COL].median()
        c2i_med = df.loc[mask_weddell_agg_ct, C2I_SNOW_COL].median()
        amsr2_xs = (100 * (ct_med - c2i_med) / c2i_med
                     if c2i_med > 0 else np.nan)
        dhi = df.loc[mask_weddell_agg_ct, 'Delta_h_i_CT'].median()
        log(f'  Cryo-TEMPO Weddell agg.   '
            f'n={mask_weddell_agg_ct.sum():>6,d}  '
            f'Delta h_i={dhi:>+.3f} m  '
            f'AMSR2 excess vs CRYO2ICE={amsr2_xs:>+.1f}%')

    # ------------------------------------------------------------------
    # 11. Negative thickness diagnostics
    # ------------------------------------------------------------------
    log(f'\n{"="*70}')
    log(f'NEGATIVE THICKNESS DIAGNOSTICS')
    log(f'{"="*70}')

    for scenario, col in [('Base_CCI (AMSR2 baseline)', 'h_i_Base_CCI'),
                           ('Base_CT  (AMSR2 baseline)', 'h_i_Base_CT'),
                           ('Imp_CCI  (CRYO2ICE replaces AMSR2)', 'h_i_Imp_CCI'),
                           ('Imp_CT   (CRYO2ICE replaces AMSR2)', 'h_i_Imp_CT')]:
        valid = df[col].notna()
        neg = valid & (df[col] < 0)
        n_valid = valid.sum()
        n_neg = neg.sum()
        if n_valid > 0:
            log(f'  Scenario {scenario:<40s}: {n_neg:>6,d} / {n_valid:>6,d} '
                f'negative ({100*n_neg/n_valid:.1f}%)')

    # ------------------------------------------------------------------
    # 12. Save outputs
    # ------------------------------------------------------------------
    output_cols = [
        'lat', 'lon', CS2_RFB_COL, IS2_FB_COL, C2I_SNOW_COL,
        'snow_thickness_unc', 'AMSR2_SIC', 'AMSR2_snow_depth',
        'date', 'year_month', 'year', SECTOR_COL,
        'rho_i_assigned',
        CCI_SNOW_COL, CT_SNOW_COL,
        'cci_match_dist_km', 'ct_match_dist_km',
        'h_i_Base_CCI', 'h_i_Base_CT',
        'h_i_Imp_CCI',  'h_i_Imp_CT',
    ]
    for col in ['Delta_h_i_CCI', 'Delta_h_i_CT']:
        if col in df.columns:
            output_cols.append(col)
    output_cols = [c for c in output_cols if c in df.columns]

    out_csv = os.path.join(OUT_DIR, 'R3_SIT_snow_swap_all_winters_v5.csv')
    df[output_cols].to_csv(out_csv, index=False)
    log(f'\nOutput CSV saved: {out_csv}')
    log(f'  Columns saved: {len(output_cols)}')

    elapsed = timer.time() - t0
    log(f'Total elapsed: {elapsed:.1f} s')

    summary_path = os.path.join(OUT_DIR, 'R3_SIT_snow_swap_summary_v5.txt')
    with open(summary_path, 'w') as fout:
        fout.write('\n'.join(summary_lines))
    print(f'Summary report saved: {summary_path}')
    print('Done.')

#!/usr/bin/env python3
"""
R3_Step08b_verify_sector_correlation.py
========================================
Verify whether the pooled Pearson r = 0.189 between C2I snow thickness
and CCI/Cryo-TEMPO climatological snow depth is inflated by the inter-
sector contrast (Weddell vs Ross).

Method: Compute within-sector correlations for Western Weddell, Eastern
        Weddell, and Ross separately, then compare against the pooled r.

Justification:
  Simpson's paradox can inflate pooled correlations when subgroups have
  different means. If the Weddell sectors have systematically higher snow
  depths in both datasets and Ross has systematically lower values, the
  pooled r captures this between-group gradient rather than meaningful
  within-group covariation. If within-sector r drops to near zero while
  the pooled r remains ~0.19, the correlation is driven entirely by the
  sector-level contrast — both datasets agree on the Weddell–Ross
  gradient direction but disagree on absolute magnitudes at the segment
  level. This is itself a scientifically meaningful result that parallels
  the AMSR2 scale-mismatch finding in M2 (r² < 0.011).

Input:
  - R3_L2P_collocation_aug2022.csv (output from Step 08)

Author: Xinlong Liu
Date:   09 Apr 2026
Environment: cryo2ice_env on NCI Gadi (project gv90)
"""

import numpy as np
import pandas as pd
from scipy import stats

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
COLLOC_CSV = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation/R3_L2P_collocation_aug2022.csv'

C2I_SNOW_COL = 'snow_thickness'
CCI_SNOW_COL = 'cci_snow_depth'
CT_SNOW_COL  = 'ct_snow_depth'
SECTOR_COL   = 'sector'

SECTORS = ['Western_Weddell', 'Eastern_Weddell', 'Ross']

# ===========================================================================
#  MAIN
# ===========================================================================
if __name__ == '__main__':

    df = pd.read_csv(COLLOC_CSV)
    print(f'Loaded {len(df):,} rows from collocation CSV')
    print(f'Columns: {list(df.columns)}\n')

    for product, prod_snow_col in [('CCI v4.0', CCI_SNOW_COL),
                                    ('Cryo-TEMPO', CT_SNOW_COL)]:

        print(f'{"="*70}')
        print(f'CORRELATION ANALYSIS: C2I snow vs {product} snow')
        print(f'{"="*70}')

        # --- Pooled correlation (all sectors combined) ---
        mask_all = df[C2I_SNOW_COL].notna() & df[prod_snow_col].notna()
        c2i_all = df.loc[mask_all, C2I_SNOW_COL].values
        prod_all = df.loc[mask_all, prod_snow_col].values
        r_pooled, p_pooled = stats.pearsonr(c2i_all, prod_all)
        r2_pooled = r_pooled ** 2

        print(f'\n  POOLED (all sectors, n={mask_all.sum():,}):')
        print(f'    Pearson r  = {r_pooled:.4f}')
        print(f'    r²         = {r2_pooled:.4f}')
        print(f'    p-value    = {p_pooled:.2e}')
        print(f'    C2I snow  — median: {np.median(c2i_all):.3f} m, mean: {np.mean(c2i_all):.3f} m')
        print(f'    {product:10s} snow — median: {np.median(prod_all):.3f} m, mean: {np.mean(prod_all):.3f} m')

        # --- Within-sector correlations ---
        print(f'\n  WITHIN-SECTOR CORRELATIONS:')
        print(f'  {"Sector":<22s} {"n":>7s} {"r":>8s} {"r²":>8s} {"p-value":>12s} '
              f'{"C2I med":>9s} {"Prod med":>9s} {"C2I mean":>9s} {"Prod mean":>9s}')
        print(f'  {"-"*100}')

        for sector in SECTORS:
            mask_s = (df[SECTOR_COL] == sector) & mask_all
            n_s = mask_s.sum()
            if n_s < 3:
                print(f'  {sector:<22s} {n_s:>7d}   (insufficient data)')
                continue

            c2i_s = df.loc[mask_s, C2I_SNOW_COL].values
            prod_s = df.loc[mask_s, prod_snow_col].values
            r_s, p_s = stats.pearsonr(c2i_s, prod_s)
            r2_s = r_s ** 2

            print(f'  {sector:<22s} {n_s:>7,d} {r_s:>8.4f} {r2_s:>8.4f} {p_s:>12.2e} '
                  f'{np.median(c2i_s):>9.3f} {np.median(prod_s):>9.3f} '
                  f'{np.mean(c2i_s):>9.3f} {np.mean(prod_s):>9.3f}')

        # --- Interpretation guidance ---
        print(f'\n  INTERPRETATION:')
        print(f'    If within-sector r values are near zero (|r| < 0.05) while the')
        print(f'    pooled r ≈ 0.19, the pooled correlation is driven by the inter-')
        print(f'    sector contrast (Simpson\'s paradox). This confirms that the')
        print(f'    climatology and the dual-altimetry retrieval agree on the Weddell–')
        print(f'    Ross snow gradient direction but show no meaningful point-level')
        print(f'    covariation — consistent with the AMSR2 scale-mismatch finding')
        print(f'    in M2 (r² < 0.011).')
        print(f'    If within-sector r values remain ≈ 0.19 or higher, the pooled')
        print(f'    correlation reflects genuine within-sector covariation and is not')
        print(f'    a Simpson\'s paradox artifact.\n')

#!/usr/bin/env python3
"""
R3_verify_table_summary_v2.py
==============================
Verification script to extract pooled-years median values from the
Step 09 v5 (Option One framing) snow-substitution CSV output.

*** v2 CHANGES (1 May 2026, Alex Option One framing): ***
  - Column name updates:
      h_i_O        -> h_i_Base_CCI / h_i_Base_CT (AMSR2-informed baselines)
      h_i_S_CCI    -> h_i_Imp_CCI  (CCI baseline AMSR2 replaced by CRYO2ICE)
      h_i_S_CT     -> h_i_Imp_CT   (Cryo-TEMPO AMSR2 replaced by CRYO2ICE)
      dh_i_snow_*  -> Delta_h_i_CCI / Delta_h_i_CT (inverted sign)
  - Sign convention: Delta h_i = h_i_Imp - h_i_Base (POSITIVE when
    CRYO2ICE recovers thickness above the AMSR2 baseline)
  - Column header text updated to reflect Option One framing
  - Input CSV path updated to v5

Purpose:
  Confirms the exact pooled-years median values for the per-sector,
  per-product entries in Table tab:R3_summary of Chapter 5 under the
  Option One framing.

Output:
  Prints per-sector, per-product median h_i_Base, h_i_Imp, and Delta h_i
  values to the terminal in a tabulated format suitable for direct
  comparison against the values to appear in the LaTeX summary table.

Run on NCI Gadi:
  source ~/cryo2ice_env/bin/activate
  python R3_verify_table_summary_v2.py
"""

import pandas as pd
import os

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
CSV_PATH = ('/g/data/gv90/xl1657/phd/M3_workspace/output/'
            'L2P_collocation/R3_SIT_snow_swap_all_winters_v5.csv')

SECTORS = ['Western_Weddell', 'Eastern_Weddell', 'Ross']

# Product to (base column, improved column, baseline-snow column) mapping
PRODUCTS = [
    ('CCI', 'h_i_Base_CCI', 'h_i_Imp_CCI', 'cci_snow_depth'),
    ('CT',  'h_i_Base_CT',  'h_i_Imp_CT',  'ct_snow_depth'),
]


# ===========================================================================
#  MAIN
# ===========================================================================
if __name__ == '__main__':

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f'Input CSV not found: {CSV_PATH}')

    df = pd.read_csv(CSV_PATH)
    print(f'Loaded {len(df):,} rows from {os.path.basename(CSV_PATH)}\n')

    # ------------------------------------------------------------------
    # Per-sector, per-product breakdown
    # ------------------------------------------------------------------
    print('=' * 110)
    print('OPTION ONE FRAMING: AMSR2 baseline -> CRYO2ICE replacement')
    print('  Delta h_i = h_i_Imp - h_i_Base  (POSITIVE = CRYO2ICE recovers')
    print('                                   thickness above AMSR2 baseline)')
    print('=' * 110)

    print(f'\n{"Sector":<20s} {"Product":<8s} {"n":>7s} '
          f'{"AMSR2_h_s":>10s} {"C2I_h_s":>9s} {"AMSR2 xs%":>10s} '
          f'{"h_i_Base":>10s} {"h_i_Imp":>9s} {"Delta_h_i":>11s}')
    print('=' * 110)

    for sector in SECTORS:
        for product, hB_col, hI_col, snow_col in PRODUCTS:

            # Filter to this sector and require valid h_i_Imp (which
            # implicitly requires both AMSR2 and CRYO2ICE snow at this
            # segment, since h_i_Imp uses CRYO2ICE on the same
            # segment population as h_i_Base)
            m = df[df['sector'] == sector].dropna(subset=[hI_col, hB_col])
            n = len(m)

            if n == 0:
                print(f'{sector:<20s} {product:<8s} {0:>7d}  (no matched data)')
                continue

            # Snow depths (medians)
            hs_amsr2 = m[snow_col].median()
            hs_c2i   = m['snow_thickness'].median()

            # AMSR2 excess relative to CRYO2ICE
            amsr2_xs = (100 * (hs_amsr2 - hs_c2i) / hs_c2i
                         if hs_c2i > 0 else float('nan'))

            # Sea-ice thickness (medians)
            hB_med = m[hB_col].median()
            hI_med = m[hI_col].median()

            # Delta h_i = h_i_Imp - h_i_Base (positive when CRYO2ICE
            # recovers thickness)
            dhi_med = m[hI_col].median() - m[hB_col].median()

            print(f'{sector:<20s} {product:<8s} {n:>7,d} '
                  f'{hs_amsr2:>10.3f} {hs_c2i:>9.3f} {amsr2_xs:>+10.1f} '
                  f'{hB_med:>10.3f} {hI_med:>9.3f} {dhi_med:>+11.3f}')

        # Blank line between sectors
        if sector != SECTORS[-1]:
            print('-' * 110)

    print('=' * 110)

    # ------------------------------------------------------------------
    # Weddell aggregate (sample-weighted union of E + W Weddell)
    # ------------------------------------------------------------------
    print('\n' + '=' * 110)
    print('WEDDELL AGGREGATE (sample-weighted union of E + W Weddell)')
    print('=' * 110)

    print(f'\n{"Sector":<20s} {"Product":<8s} {"n":>7s} '
          f'{"AMSR2_h_s":>10s} {"C2I_h_s":>9s} {"AMSR2 xs%":>10s} '
          f'{"h_i_Base":>10s} {"h_i_Imp":>9s} {"Delta_h_i":>11s}')
    print('-' * 110)

    weddell_mask = df['sector'].isin(['Western_Weddell', 'Eastern_Weddell'])

    for product, hB_col, hI_col, snow_col in PRODUCTS:
        m = df[weddell_mask].dropna(subset=[hI_col, hB_col])
        n = len(m)

        if n == 0:
            print(f'Weddell aggregate    {product:<8s} {0:>7d}  (no data)')
            continue

        hs_amsr2 = m[snow_col].median()
        hs_c2i   = m['snow_thickness'].median()
        amsr2_xs = (100 * (hs_amsr2 - hs_c2i) / hs_c2i
                     if hs_c2i > 0 else float('nan'))
        hB_med = m[hB_col].median()
        hI_med = m[hI_col].median()
        dhi_med = m[hI_col].median() - m[hB_col].median()

        print(f'{"Weddell aggregate":<20s} {product:<8s} {n:>7,d} '
              f'{hs_amsr2:>10.3f} {hs_c2i:>9.3f} {amsr2_xs:>+10.1f} '
              f'{hB_med:>10.3f} {hI_med:>9.3f} {dhi_med:>+11.3f}')

    print('=' * 110)
    print('\nUse these values to populate Table tab:R3_summary of Chapter 5')
    print('under the Option One (AMSR2 baseline, CRYO2ICE improvement) framing.')
    print('Done.')

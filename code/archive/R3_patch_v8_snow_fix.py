#!/usr/bin/env python3
"""
R3_patch_v8_snow_fix.py — apply the three code fixes found by Step 18 (27 Sep 2026)
===================================================================================
Idempotent: running it twice changes nothing the second time. Each edited file is
backed up once as <name>.bak_pre_v8.

  1. R3_Step09_snow_swap_v7.py   : the delivered 'snow_thickness' column carries no
     penetration factor (implied delta = 1.000). Recompute the C2I snow from the
     freeboards at delta = 0.70, as the Monte Carlo does.
  2. R3_Step14_amsr2_gridscale.py: mask AU_SI12 flag codes (110-160 cm stored as
     1.10-1.60 m) so they cannot enter the grid-scale comparison. Harmless if none exist.
  3. R3_Step10_snow_swap_figures_v9.py: drop the Fig. C.1 suptitle (same problem Alex
     flagged in Fig. dhi).

Run in .../M3_workspace/code:  python R3_patch_v8_snow_fix.py
"""
import os, shutil

PATCHES = [
    ('R3_Step09_snow_swap_v7.py',
     "    df['year'] = (df['year_month'] // 100).astype(int)\n",
     "    df['year'] = (df['year_month'] // 100).astype(int)\n"
     "    # v8 (Step 18): delivered snow_thickness has implied delta = 1.00; recompute at delta = 0.70\n"
     "    ETA_S = (1 + 0.51 * RHO_S / 1000.0) ** 1.5\n"
     "    df[C2I_SNOW_COL] = (df[IS2_FB_COL] - df[CS2_RFB_COL]) / (DELTA_BASELINE * ETA_S)\n"
     "    log(f'  v8: C2I snow recomputed at delta={DELTA_BASELINE}: median {df[C2I_SNOW_COL].median():.3f} m')\n",
     'v8 (Step 18)'),
    ('R3_Step14_amsr2_gridscale.py',
     "am = df['AMSR2_snow_depth'].values.astype(float)\n",
     "am = df['AMSR2_snow_depth'].values.astype(float)\n"
     "flag = np.isin(np.round(am * 100), [110, 120, 130, 140, 150, 160])   # v8: AU_SI12 flag codes (cm)\n"
     "print(f'AU_SI12 flag values masked: {int(flag.sum()):,}')\n"
     "am[flag] = np.nan\n"
     "df['AMSR2_snow_depth'] = am\n",
     'v8: AU_SI12 flag'),
    ('R3_Step10_snow_swap_figures_v9.py',
     "fig1.suptitle(\n    'AMSR2 baseline snow depth (CCI v4.0 / Cryo-TEMPO) compared with CRYO2ICE retrieval',\n"
     "    fontsize=13, fontweight='bold', y=0.98)\n",
     "# v8: suptitle removed (product names are in the row labels and the caption)\n",
     'v8: suptitle removed'),
]

for fname, old, new, marker in PATCHES:
    if not os.path.exists(fname):
        print(f'!! {fname} not found (run this in the code directory)')
        continue
    s = open(fname).read()
    if marker in s:
        print(f'   {fname}: already patched')
        continue
    if old not in s:
        print(f'!! {fname}: anchor not found, patch NOT applied; send me the file')
        continue
    if not os.path.exists(fname + '.bak_pre_v8'):
        shutil.copy(fname, fname + '.bak_pre_v8')
    open(fname, 'w').write(s.replace(old, new, 1))
    print(f'OK {fname}: patched')

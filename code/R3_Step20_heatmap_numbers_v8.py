#!/usr/bin/env python3
"""
R3_Step20_heatmap_numbers_v8.py - prints every number shown in Figure R3_Fig_C2_03 (sector x year heatmap)
from the v8-fixed snow-swap CSV (C2I snow at delta = 0.70), using exactly the Step 10 v9 definitions:
  excess (%) = 100 * (median L2 snow - median C2I snow) / median C2I snow, per sector-year
  dh_i (m)   = median over matchups of (h_i with C2I snow - h_i with L2 snow), per sector-year
Also prints the pooled values, which must match Table 3 (CCI +23/+36/+42 %, +0.308/+0.392/+0.319 m;
CT +51/+44/+43 %, +0.766/+0.406/+0.327 m).
Run (login node, seconds):
  module load python3/3.9.2; source ~/cryo2ice_env/bin/activate
  python R3_Step20_heatmap_numbers_v8.py
"""
import numpy as np, pandas as pd

SWAP = '/g/data/gv90/xl1657/phd/M3_workspace/output/L2P_collocation/R3_SIT_snow_swap_all_winters_v7.csv'
SECT = ['Western_Weddell', 'Eastern_Weddell', 'Ross']
PROD = {'CCI': ('cci_snow_depth', 'h_i_Imp_CCI', 'h_i_Base_CCI', [2022, 2023]),
        'CT':  ('ct_snow_depth',  'h_i_Imp_CT',  'h_i_Base_CT',  [2022, 2023, 2024, 2025])}

df = pd.read_csv(SWAP)
df['year'] = (df['year_month'] // 100).astype(int)

def cell(m, l2, imp, base):
    d = df.loc[m, imp] - df.loc[m, base]
    c2i, l2s = df.loc[m, 'snow_thickness'].median(), df.loc[m, l2].median()
    return int(m.sum()), 100 * (l2s - c2i) / c2i, d.median()

for p, (l2, imp, base, years) in PROD.items():
    ok = df[imp].notna() & df[base].notna()
    print(f'\n=== {p} ===   (n | AMSR2 excess % | median dh_i m)')
    print(f'{"sector":17s}' + ''.join(f'{y:>26d}' for y in years) + f'{"pooled":>26s}')
    for s in SECT:
        row = f'{s:17s}'
        for y in years + ['all']:
            m = ok & (df['sector'] == s) & ((df['year'] == y) if y != 'all' else True)
            n, ex, dh = cell(m, l2, imp, base)
            row += f'   {n:6,d} {ex:+5.0f}% {dh:+6.2f} m'
        print(row)

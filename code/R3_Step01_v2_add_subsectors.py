#!/usr/bin/env python3
"""
R3_Step01_v2_add_subsectors.py — Add sub-sectors + positive freeboard filter
=============================================================================
Changes from v1 (Alex 08 Apr 2026):
  1. Filters out segments with IS2 total freeboard <= 0 or CS2 radar freeboard <= 0
  2. Uses R3_config_v4

Run: source ~/cryo2ice_env/bin/activate && python R3_Step01_v2_add_subsectors.py
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import *

pkl_path = f'{M3_OUTPUT}/R3_input_d070_verified.pkl'
df = pd.read_pickle(pkl_path)
n_orig = len(df)
print(f"Loaded {n_orig:,} matchups from {pkl_path}")

# ── Positive freeboard filter (Alex 08 Apr) ──
is2_fb = df['is2_fb_wm'].values
cs2_fb = df['cs2_rfb'].values

neg_is2 = (is2_fb <= QF_IS2_MIN).sum()
neg_cs2 = (cs2_fb <= QF_CS2_MIN).sum()
neg_both = ((is2_fb <= QF_IS2_MIN) | (cs2_fb <= QF_CS2_MIN)).sum()

print(f"\n--- Positive Freeboard Filter ---")
print(f"  IS2 fb <= 0: {neg_is2:,} ({100*neg_is2/n_orig:.2f}%)")
print(f"  CS2 fb <= 0: {neg_cs2:,} ({100*neg_cs2/n_orig:.2f}%)")
print(f"  Either <= 0:  {neg_both:,} ({100*neg_both/n_orig:.2f}%)")

mask_positive = (is2_fb > QF_IS2_MIN) & (cs2_fb > QF_CS2_MIN)
df = df[mask_positive].reset_index(drop=True)
n_filtered = len(df)
print(f"  After filter: {n_filtered:,} matchups ({n_orig - n_filtered:,} removed)")

# ── Add sub-sectors ──
df['subsector'] = assign_subsector_vec(df['lon'].values)

print(f"\n--- Region Counts (after positive fb filter) ---")
sectors = df['sector'].values
subsectors = df['subsector'].values
for r in REGIONS:
    mask = get_region_mask(sectors, subsectors, r)
    print(f"  {r:12s}: {mask.sum():>7,}")

# ── Save ──
df.to_pickle(pkl_path)
print(f"\nRe-saved with positive fb filter + subsector column: {pkl_path}")
print(f"  {n_orig:,} -> {n_filtered:,} ({n_orig - n_filtered:,} negative-fb segments removed)")

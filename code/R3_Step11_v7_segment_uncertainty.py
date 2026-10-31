#!/usr/bin/env python3
"""
R3_Step11_v7_segment_uncertainty.py
===================================
Segment-scale sea-ice thickness and snow depth uncertainty, benchmarked against
the CRISTAL Mission Requirements Document (ESA-EOPSM-CPTM-MRD-3350, Issue 3.0).

v7 changes (September 2026, RSE-D-26-03119 resubmission)
  * bulk ice density follows the per-matchup MIXTURE prior (Step 02 v7): within a
    segment the MYI weight is the segment-mean collocated Melsheimer MYI
    concentration f_seg; each realisation draws ONE component (Bernoulli f_seg)
    and ONE density for the whole segment, keeping the common-mode structure
  * snow density is unbounded (the [150, 550] clip of v1 was never applied in
    Step 02 and has been removed from the manuscript)
  * input is the ice-type pickle from Step 13; strata reported for REGIONS_V6
    (segments classed MYI where f_seg >= 0.5)
  * default --tag v7

  MRD-280 (p.40)         freeboard accuracy 0.03 m over segments <= 25 km
                         (goal 0.01 m; minimum retrievable freeboard 0.03 m)
  MRD-330 (p.41)         thickness uncertainty < 0.15 m over segments <= 25 km
                         (Note 2: minimum thickness 0.35-0.5 m single-frequency)
  MRD-380 Note 4 (p.44)  snow depth uncertainty goal 0.05 m over 25 km segments

METHOD
------
Snow density, penetration fraction and bulk ice density are COMMON-MODE within a
segment: the same assumed values apply to every matchup in it, so their
contribution does not average down with segment length. Only freeboard
measurement noise is independent and averages as 1/sqrt(n).

The Monte Carlo therefore draws parameters ONCE PER SEGMENT PER REALISATION and
freeboards PER MATCHUP. Aggregating per-matchup sigma as sigma/sqrt(n) would be
wrong and would understate the segment uncertainty.

Loops are ordered realisation-outside so that every draw is vectorised across all
segments and matchups simultaneously (see np.bincount for the segment means).
The reverse ordering is ~4000x slower.

A parameters-fixed variant (freeboard noise only) is run alongside each
configuration. Plotting the two against segment length shows one curve falling as
1/sqrt(n) and one flat at the common-mode floor.

Author: Xinlong Liu | 30 August 2026
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import (
    M3_OUTPUT, N_MC, FB_SIGMA_RAW,
    RHO_S_CENTRAL, RHO_S_SIGMA, RHO_W_CENTRAL,
    RHO_I_FYI_MU, RHO_I_FYI_SIGMA, RHO_I_MYI_MU, RHO_I_MYI_SIGMA,
    RHO_I_UPPER_CAP, DELTA_BASELINE, DELTA_SIGMA, DELTA_CLIP, ETA_S,
    REGIONS_V6, MYI_THRESHOLD_PCT, assign_subsector_vec,
    mixture_rho_i_central, get_region_mask_v6, compute_ice_thickness,
)

# ── CRISTAL requirements, all at <= 25 km ────────────────────────────────
MRD_SNOW_M      = 0.05    # MRD-380 Note 4
MRD_THICK_M     = 0.15    # MRD-330
MRD_FB_MIN_M    = 0.03    # MRD-280 Note 2, minimum retrievable freeboard
MRD_THICK_MIN_M = 0.35    # MRD-330 Note 2, lower end of 0.35-0.50 m

SEG_LENS_KM  = [5.0, 10.0, 25.0, 50.0, 100.0]
MAX_GAP_KM   = 5.0        # split rather than straddle a data hole
MIN_PER_SEG  = 3          # segments below this are dropped
EARTH_R_KM   = 6371.0


# ═════════════════════════════════════════════════════════════════════════
# Segment construction
# ═════════════════════════════════════════════════════════════════════════
def haversine_steps_km(lat, lon):
    """Great-circle distance between consecutive points; first element is 0."""
    p1, p2 = np.radians(lat[:-1]), np.radians(lat[1:])
    dp = p2 - p1
    dl = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return np.concatenate([[0.0], 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))])


def build_segments(lat, lon, day_key, seg_len_km, max_gap_km=MAX_GAP_KM):
    """
    Bin consecutive along-track matchups into segments of at most seg_len_km.

    The matchup file carries no track identifier, so continuity is inferred:
    a new segment starts on a change of acquisition day, on an along-track gap
    exceeding max_gap_km, or when the running distance would exceed seg_len_km.

    Returns seg_idx (int64, one per matchup) and the actual length of each
    segment in km.
    """
    step = haversine_steps_km(lat, lon)
    new_day = np.concatenate([[True], day_key[1:] != day_key[:-1]])
    force_break = new_day | (step > max_gap_km)

    n = len(lat)
    seg_idx = np.empty(n, dtype=np.int64)
    seg_len = []
    sid, cum = -1, 0.0
    for i in range(n):
        if force_break[i] or (cum + step[i]) > seg_len_km:
            if sid >= 0:
                seg_len.append(cum)
            sid += 1
            cum = 0.0
        else:
            cum += step[i]
        seg_idx[i] = sid
    seg_len.append(cum)
    return seg_idx, np.asarray(seg_len)


def compact_segments(seg_idx, keep_mask):
    """Drop unwanted matchups and renumber the surviving segments 0..M-1."""
    kept = seg_idx[keep_mask]
    uniq, new_idx = np.unique(kept, return_inverse=True)
    return new_idx, uniq


# ═════════════════════════════════════════════════════════════════════════
# Segment-scale Monte Carlo
# ═════════════════════════════════════════════════════════════════════════
def segment_mc(is2, cs2, seg_idx, seg_n, f_seg,
               n_mc=N_MC, seed=1101, params_fixed=False, chunk=None):
    """
    Return (n_mc, n_seg) arrays of segment-mean snow depth and thickness.

    Parameters are drawn once per segment per realisation and broadcast to the
    matchups within that segment. Freeboards are drawn per matchup.

    f_seg is the segment-mean MYI weight (0-1). Ice density is drawn from the
    two-component mixture ONCE per segment per realisation: the component is
    selected by a Bernoulli(f_seg) draw and the density from that component's
    Gaussian, capped at RHO_I_UPPER_CAP.

    params_fixed=True holds snow density, penetration and ice density at their
    central values (density = mixture expected value), isolating the
    freeboard-noise contribution. That variant should fall as 1/sqrt(n); the
    full run should not.
    """
    rng = np.random.default_rng(seed)
    n_seg = len(seg_n)
    n_match = len(is2)
    dtype = np.float32 if (chunk is not None) else np.float64

    hs_seg = np.empty((n_mc, n_seg), dtype=dtype)
    hi_seg = np.empty((n_mc, n_seg), dtype=dtype)

    inv_n = 1.0 / seg_n
    mu_seg = f_seg * RHO_I_MYI_MU + (1.0 - f_seg) * RHO_I_FYI_MU

    for k in range(n_mc):
        if params_fixed:
            d_s = np.full(n_seg, DELTA_BASELINE)
            rs_s = np.full(n_seg, RHO_S_CENTRAL)
            ri_s = mu_seg
        else:
            d_s = np.clip(rng.normal(DELTA_BASELINE, DELTA_SIGMA, n_seg),
                          *DELTA_CLIP)
            rs_s = rng.normal(RHO_S_CENTRAL, RHO_S_SIGMA, n_seg)      # unbounded
            is_myi = rng.random(n_seg) < f_seg
            mu_c = np.where(is_myi, RHO_I_MYI_MU, RHO_I_FYI_MU)
            sg_c = np.where(is_myi, RHO_I_MYI_SIGMA, RHO_I_FYI_SIGMA)
            ri_s = np.minimum(rng.normal(mu_c, sg_c), RHO_I_UPPER_CAP)

        # broadcast the per-segment draw to every matchup in that segment
        d_m = d_s[seg_idx]
        rs_m = rs_s[seg_idx]
        ri_m = ri_s[seg_idx]

        # independent per-matchup freeboard noise
        is2_k = is2 + rng.normal(0.0, FB_SIGMA_RAW, n_match)
        cs2_k = cs2 + rng.normal(0.0, FB_SIGMA_RAW, n_match)

        eta_m = (1 + 0.51 * rs_m / 1000.0) ** 1.5
        hs_m = (is2_k - cs2_k) / (d_m * eta_m)
        hi_m = compute_ice_thickness(is2_k, hs_m, rs_m, ri_m, RHO_W_CENTRAL)

        hs_seg[k] = np.bincount(seg_idx, weights=hs_m, minlength=n_seg) * inv_n
        hi_seg[k] = np.bincount(seg_idx, weights=hi_m, minlength=n_seg) * inv_n

    return hs_seg, hi_seg


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg-len', type=float, default=None,
                    help='Run a single segment length (km) instead of all.')
    ap.add_argument('--filter', type=str, default=None,
                    help='Run a single filter: F0_none, F1_mrd280, F2_mrd280_330')
    ap.add_argument('--n-mc', type=int, default=N_MC)
    ap.add_argument('--skip-fixed', action='store_true',
                    help='Skip the freeboard-only (parameters fixed) variant.')
    ap.add_argument('--tag', type=str, default='v7')
    args = ap.parse_args()

    t_start = time.time()
    print('=' * 74)
    print('R3 Step 11 v7: Segment-scale uncertainty vs CRISTAL requirements (mixture rho_i prior)')
    print(f'  MRD-380 Note 4  snow      <= {MRD_SNOW_M:.2f} m over 25 km')
    print(f'  MRD-330         thickness <= {MRD_THICK_M:.2f} m over 25 km')
    print(f'  MRD-280 Note 2  minimum retrievable freeboard {MRD_FB_MIN_M:.2f} m')
    print(f'  MRD-330 Note 2  minimum thickness {MRD_THICK_MIN_M:.2f} m')
    print(f'  N_MC = {args.n_mc:,}   parameters drawn PER SEGMENT, '
          f'freeboards PER MATCHUP')
    print('=' * 74)

    # ── Load and order along-track ───────────────────────────────────────
    df = pd.read_pickle(f'{M3_OUTPUT}/R3_input_d070_icetype.pkl')
    if 'subsector' not in df.columns:
        df['subsector'] = assign_subsector_vec(df['lon'].values)

    # Sorting is essential: segments are built from consecutive rows.
    df = df.sort_values(['date', 'lat', 'lon']).reset_index(drop=True)

    lat = df['lat'].values.astype(np.float64)
    lon = df['lon'].values.astype(np.float64)
    day = df['date'].values.astype(str)
    is2 = df['is2_fb_wm'].values.astype(np.float64)
    cs2 = df['cs2_rfb'].values.astype(np.float64)
    sectors = df['sector'].values
    subsectors = df['subsector'].values
    f_all = np.clip(df['myi_frac'].values.astype(np.float64) / 100.0, 0.0, 1.0)
    print(f'Loaded {len(df):,} matchups; mean MYI weight {f_all.mean():.3f}')

    # ── Deterministic central-value estimate, used for the MRD filters ───
    # Filters are applied on the central estimate rather than per realisation,
    # so that segment membership is fixed and the variance reflects sampling
    # alone rather than sampling plus selection.
    hs_c = (is2 - cs2) / (DELTA_BASELINE * ETA_S)
    hfi_c = is2 - hs_c                       # sea-ice freeboard
    mu_all = mixture_rho_i_central(df['myi_frac'].values)
    hi_c = compute_ice_thickness(is2, hs_c, RHO_S_CENTRAL, mu_all,
                                 RHO_W_CENTRAL)

    FILTERS = {
        'F0_none':       np.ones(len(df), dtype=bool),
        'F1_mrd280':     hfi_c >= MRD_FB_MIN_M,
        'F2_mrd280_330': (hfi_c >= MRD_FB_MIN_M) & (hi_c >= MRD_THICK_MIN_M),
    }
    print('\nMRD retrievability filters:')
    for fname, fmask in FILTERS.items():
        print(f'  {fname:<16s} {fmask.sum():>7,} matchups '
              f'({100 * fmask.mean():5.1f}%)')

    seg_lens = [args.seg_len] if args.seg_len else SEG_LENS_KM
    fnames = [args.filter] if args.filter else list(FILTERS)

    rows, summary = [], {}

    for L in seg_lens:
        seg_idx_all, _ = build_segments(lat, lon, day, L)
        for fname in fnames:
            fmask = FILTERS[fname]
            seg_idx_f, _ = compact_segments(seg_idx_all, fmask)
            seg_n = np.bincount(seg_idx_f)

            # drop under-populated segments, then renumber again
            big = np.isin(seg_idx_f, np.flatnonzero(seg_n >= MIN_PER_SEG))
            idx_f = np.flatnonzero(fmask)[big]
            seg_idx, _ = compact_segments(seg_idx_f, big)
            seg_n = np.bincount(seg_idx).astype(np.float64)
            n_seg = len(seg_n)

            f_seg = np.bincount(seg_idx, weights=f_all[idx_f]) / seg_n
            icetype_seg = np.where(f_seg >= MYI_THRESHOLD_PCT / 100.0, 'MYI', 'FYI')

            for fixed in ([False] if args.skip_fixed else [False, True]):
                label = 'freeboard-only' if fixed else 'full'
                t0 = time.time()
                hs_seg, hi_seg = segment_mc(
                    is2[idx_f], cs2[idx_f], seg_idx, seg_n,
                    f_seg, n_mc=args.n_mc,
                    seed=1101 + int(L), params_fixed=fixed)
                hs_sig = hs_seg.std(axis=0)
                hi_sig = hi_seg.std(axis=0)
                dt = time.time() - t0

                # per-segment sector label: the modal sector of its matchups
                first = np.zeros(n_seg, dtype=np.int64)
                np.maximum.at(first, seg_idx, np.arange(len(seg_idx)))
                sec_seg = sectors[idx_f][first]
                sub_seg = subsectors[idx_f][first]

                for i in range(n_seg):
                    rows.append({
                        'seg_len_km': L, 'filter': fname, 'variant': label,
                        'sector': sec_seg[i], 'subsector': sub_seg[i],
                        'icetype': icetype_seg[i], 'myi_weight': float(f_seg[i]),
                        'n_matchups': int(seg_n[i]),
                        'hs_mean': float(hs_seg[:, i].mean()),
                        'hs_sigma': float(hs_sig[i]),
                        'hi_mean': float(hi_seg[:, i].mean()),
                        'hi_sigma': float(hi_sig[i]),
                        'meets_mrd380': bool(hs_sig[i] <= MRD_SNOW_M),
                        'meets_mrd330': bool(hi_sig[i] <= MRD_THICK_M),
                    })

                key = f'{L:g}km_{fname}_{label}'
                summary[key] = {
                    'seg_len_km': L, 'filter': fname, 'variant': label,
                    'n_segments': int(n_seg),
                    'median_matchups_per_seg': float(np.median(seg_n)),
                    'median_hs_sigma_m': float(np.median(hs_sig)),
                    'median_hi_sigma_m': float(np.median(hi_sig)),
                    'frac_meeting_mrd380': float((hs_sig <= MRD_SNOW_M).mean()),
                    'frac_meeting_mrd330': float((hi_sig <= MRD_THICK_M).mean()),
                    'runtime_s': round(dt, 1),
                }
                by_region = {}
                for r in REGIONS_V6:
                    m = get_region_mask_v6(sec_seg, sub_seg, icetype_seg, r)
                    if m.sum():
                        by_region[r] = {
                            'n_segments': int(m.sum()),
                            'median_hs_sigma_m': float(np.median(hs_sig[m])),
                            'median_hi_sigma_m': float(np.median(hi_sig[m])),
                            'frac_meeting_mrd380':
                                float((hs_sig[m] <= MRD_SNOW_M).mean()),
                            'frac_meeting_mrd330':
                                float((hi_sig[m] <= MRD_THICK_M).mean()),
                        }
                summary[key]['by_region'] = by_region

                flag380 = 'PASS' if np.median(hs_sig) <= MRD_SNOW_M else 'FAIL'
                flag330 = 'PASS' if np.median(hi_sig) <= MRD_THICK_M else 'FAIL'
                print(f'  {L:>5.0f} km  {fname:<15s} {label:<15s} '
                      f'n_seg={n_seg:>6,}  '
                      f'hs_sig={np.median(hs_sig):.4f} m [{flag380}]  '
                      f'hi_sig={np.median(hi_sig):.4f} m [{flag330}]  '
                      f'({dt:.0f}s)')

                del hs_seg, hi_seg

    csv_path = f'{M3_OUTPUT}/R3_segment_uncertainty_{args.tag}.csv'
    json_path = f'{M3_OUTPUT}/R3_segment_summary_{args.tag}.json'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, 'w') as fh:
        json.dump({
            'mrd_reference': 'ESA-EOPSM-CPTM-MRD-3350 Issue 3.0, 29 May 2020',
            'mrd_snow_m': MRD_SNOW_M, 'mrd_thickness_m': MRD_THICK_M,
            'mrd_min_freeboard_m': MRD_FB_MIN_M,
            'mrd_min_thickness_m': MRD_THICK_MIN_M,
            'n_mc': args.n_mc,
            'method': ('Parameters drawn once per segment per realisation '
                       '(common-mode); freeboards drawn per matchup '
                       '(independent). rho_i from the per-matchup mixture prior '
                       'with segment-mean MYI weight; rho_s unbounded.'),
            'configurations': summary,
        }, fh, indent=2)

    print(f'\n  Saved: {csv_path}')
    print(f'  Saved: {json_path}')
    print(f'Step 11 v7 complete in {time.time() - t_start:.1f} s')


if __name__ == '__main__':
    main()

"""
R3_config_v4.py — Shared configuration (v4, Alex 29 April 2026 refinement)
==========================================================================
Change from previous version (sample-weighted Weddell aggregate):
  - Added per-segment regional rho_i parameterisation lookup
  - Western Weddell sub-sector: Alexandrov MYI parameters (882, 23)
  - Eastern Weddell sub-sector: Alexandrov FYI parameters (916.7, 35.7)
  - Ross Sea: Alexandrov FYI parameters (916.7, 35.7)
  - Weddell Sea aggregate: now reflects the differentiated treatment
    automatically because it is the union of the eastern and western
    sub-sectors (sample-weighted by segment counts)
  - 940 kg/m³ soft upper cap retained for all parameterisations
  - No lower bound applied

Alex's 29 April 2026 refinement:
  The Weddell Sea aggregate was previously computed under uniform
  Alexandrov FYI parameterisation, which is inconsistent with the
  differentiated treatment of the two sub-sectors. The aggregate is
  now computed as a sample-weighted union of the eastern Weddell
  (n = 28,002 segments under FYI) and western Weddell (n = 9,831
  segments under MYI). The aggregate mean ice thickness will therefore
  reflect the realistic mixed ice-type regime of the full Weddell Sea
  rather than an artificial uniform FYI assumption.

  Alex has also requested that the chapter explicitly acknowledge the
  simplification implicit in the pure MYI assumption for the western
  Weddell sub-sector, because the Bremen ASCAT-AMSR2 ice-type imagery
  shows that pure MYI is not the realistic regime in any single year.

Reference:
  Alexandrov, V., Sandven, S., Wahlin, J., and Johannessen, O. M. (2010):
  The relation between sea ice thickness and freeboard in the Arctic,
  The Cryosphere, 4, 373–380, https://doi.org/10.5194/tc-4-373-2010
  Timco, G.W. and Frederking, R.M.W. (1996):
  A review of sea ice density,
  Cold Regions Science and Technology, 24(1), 1–6,
  https://doi.org/10.1016/0165-232X(95)00007-X
"""

import numpy as np

# ── Paths ──
M2_CSV = '/g/data/gv90/xl1657/phd/M2_workspace/output/snow_thickness/snow_thickness_with_amsr2_all.csv'
M3_BASE = '/g/data/gv90/xl1657/phd/M3_workspace'
M3_OUTPUT = f'{M3_BASE}/output'
M3_FIGURES = f'{M3_BASE}/figures'
M3_CODE = f'{M3_BASE}/code'

# ── IS2/CS2 freeboard ──
FB_SIGMA_RAW = 0.030

# ── Snow density ρ_s ──
RHO_S_CENTRAL = 330.0
RHO_S_SIGMA = 70.0

# ── Sea-ice density ρ_i: regional parameterisation ──
# *** Alexandrov et al. (2010) parameterisation, 940 kg/m³ cap ***
# *** Dual-confirmed by Alex (27 April 2026) and Stuart (28 April 2026) ***
# *** Regional differentiation refined by Alex (29 April 2026) ***
RHO_I_FYI_MU = 916.7        # Alexandrov FYI for E. Weddell, Ross
RHO_I_FYI_SIGMA = 35.7
RHO_I_MYI_MU = 882.0        # Alexandrov MYI for W. Weddell only
RHO_I_MYI_SIGMA = 23.0
RHO_I_UPPER_CAP = 940.0     # Timco & Frederking (1996) review upper bound

# Default parameterisation (FYI) — used for backward compatibility with
# downstream scripts that have not been updated to the regional treatment
RHO_I_CONFIGURED_MU = RHO_I_FYI_MU
RHO_I_SIGMA_DEFAULT = RHO_I_FYI_SIGMA

# Reference value for figure annotations (pure-ice density)
RHO_I_PURE_ICE = 917.0

# Density sweep test values (kept for backward compatibility with sweep code)
RHO_I_CENTRALS = [882, 900, 916.7, 925]
RHO_I_SIGMAS = [15, 20, 25, 30, 35.7, 40]

# Single distribution shape — Gaussian with upper cap only
DIST_SHAPES = ['gaussian']

# ── Ku-band penetration δ ──
DELTA_BASELINE = 0.70
DELTA_SIGMA = 0.20
DELTA_CLIP = (0.30, 1.00)

# ── Seawater density ρ_w: FIXED ──
RHO_W_CENTRAL = 1024.0

# ── Derived constants ──
ETA_S = (1 + 0.51 * 0.330)**1.5
FB_SIGMA_PROP = FB_SIGMA_RAW / (DELTA_BASELINE * ETA_S)

# ── Quality Filter ──
QF_IS2_MIN = 0.0
QF_CS2_MIN = 0.0
QF_CS2_MAX = 10.0

# ── Sector Definitions ──
WEDDELL_LON = (-62.0, 15.0)
WEDDELL_WEST_LON = (-62.0, -45.0)
WEDDELL_EAST_LON = (-45.0, 15.0)
ROSS_LON_A = (160.0, 180.0)
ROSS_LON_B = (-180.0, -140.0)

# ── MC Parameters ──
N_MC = 10_000
CONSTANT_SNOW = 0.15

# ── Histogram bins for full MC draw accumulation ──
HIST_BINS = np.linspace(-2, 8, 201)

# ── Colours (Antarctic palette) ──
C_WED = '#0072B2'
C_WED_W = '#882255'
C_WED_E = '#44AA99'
C_ROSS = '#D55E00'
C_ALL = '#333333'

REGIONS = ['Weddell', 'Weddell_W', 'Weddell_E', 'Ross']
REGION_LABELS = {
    'Weddell': 'Weddell Sea', 'Weddell_W': 'Western Weddell',
    'Weddell_E': 'Eastern Weddell', 'Ross': 'Ross Sea',
}
REGION_COLORS = {
    'Weddell': C_WED, 'Weddell_W': C_WED_W,
    'Weddell_E': C_WED_E, 'Ross': C_ROSS,
}


def assign_sector_vec(lons):
    sectors = np.full(len(lons), 'Other', dtype='U12')
    ross = ((lons >= ROSS_LON_A[0]) & (lons <= ROSS_LON_A[1])) | \
           ((lons >= ROSS_LON_B[0]) & (lons <= ROSS_LON_B[1]))
    sectors[ross] = 'Ross'
    wed = (lons >= WEDDELL_LON[0]) & (lons <= WEDDELL_LON[1])
    sectors[wed] = 'Weddell'
    return sectors


def assign_subsector_vec(lons):
    subsectors = np.full(len(lons), '', dtype='U12')
    wed_w = (lons >= WEDDELL_WEST_LON[0]) & (lons < WEDDELL_WEST_LON[1])
    wed_e = (lons >= WEDDELL_EAST_LON[0]) & (lons <= WEDDELL_EAST_LON[1])
    subsectors[wed_w] = 'Weddell_W'
    subsectors[wed_e] = 'Weddell_E'
    return subsectors


def get_region_mask(sectors, subsectors, region):
    if region == 'Weddell':
        return sectors == 'Weddell'
    elif region == 'Weddell_W':
        return subsectors == 'Weddell_W'
    elif region == 'Weddell_E':
        return subsectors == 'Weddell_E'
    elif region == 'Ross':
        return sectors == 'Ross'
    else:
        return sectors == region


def derive_month(dates):
    return (dates // 100) % 100


def get_regional_rho_i_params(subsectors):
    """
    Return per-segment Alexandrov parameters based on sub-sector membership.

    *** Alex's 29 April 2026 refinement ***

    Western Weddell sub-sector:  Alexandrov MYI parameters (882, 23 kg/m³)
    All other regions:           Alexandrov FYI parameters (916.7, 35.7 kg/m³)

    This produces the correct sample-weighted Weddell Sea aggregate when
    the eastern and western Weddell segments are combined, rather than
    applying a uniform FYI treatment that would be inconsistent with the
    differentiated regional physics.

    Parameters
    ----------
    subsectors : np.ndarray of str
        Per-segment sub-sector labels from assign_subsector_vec()

    Returns
    -------
    mu_arr : np.ndarray
        Per-segment Alexandrov mu (kg/m³)
    sigma_arr : np.ndarray
        Per-segment Alexandrov sigma (kg/m³)
    """
    n = len(subsectors)
    mu_arr = np.full(n, RHO_I_FYI_MU, dtype=np.float64)
    sigma_arr = np.full(n, RHO_I_FYI_SIGMA, dtype=np.float64)
    ww_mask = (subsectors == 'Weddell_W')
    mu_arr[ww_mask] = RHO_I_MYI_MU
    sigma_arr[ww_mask] = RHO_I_MYI_SIGMA
    return mu_arr, sigma_arr


def sample_density(shape, central, sigma, n, rng=None):
    """
    Sample sea-ice density from a Gaussian with soft upper cap at 940 kg/m³.

    The 'central' and 'sigma' arguments can be scalars (uniform across draws)
    or per-element arrays of length n (different parameters per draw).

    *** 940 kg/m³ cap dual-confirmed by Alex and Stuart (April 2026) ***
    Justification: Timco and Frederking (1996) review upper bound for
    in-situ first-year ice below the waterline, just below the
    Cox and Weeks (1983) 945 kg/m³ air-free theoretical maximum.
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.normal(central, sigma, size=n)
    samples = np.minimum(samples, RHO_I_UPPER_CAP)
    return samples


def sample_density_regional(subsectors, n_per_seg, rng=None):
    """
    Sample sea-ice density for each segment using regional parameterisation.

    *** Alex's 29 April 2026 refinement ***

    For each of the n_segments segments, draws n_per_seg samples from a
    Gaussian with the Alexandrov parameters appropriate to its sub-sector
    membership, with the 940 kg/m³ soft upper cap applied uniformly.

    Parameters
    ----------
    subsectors : np.ndarray of str, shape (n_segments,)
        Per-segment sub-sector labels
    n_per_seg : int
        Number of MC draws per segment
    rng : np.random.Generator, optional

    Returns
    -------
    samples : np.ndarray, shape (n_per_seg, n_segments)
        MC draws of rho_i for each segment
    """
    if rng is None:
        rng = np.random.default_rng()
    mu_arr, sigma_arr = get_regional_rho_i_params(subsectors)
    n_segments = len(subsectors)
    samples = rng.normal(loc=mu_arr[np.newaxis, :],
                         scale=sigma_arr[np.newaxis, :],
                         size=(n_per_seg, n_segments))
    samples = np.minimum(samples, RHO_I_UPPER_CAP)
    return samples


def sample_delta(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.normal(DELTA_BASELINE, DELTA_SIGMA, size=n)
    samples = np.clip(samples, DELTA_CLIP[0], DELTA_CLIP[1])
    return samples


def compute_effective_mean(central, sigma, n=500000):
    """Compute realised mean from the capped Gaussian sampler."""
    rng = np.random.default_rng(0)
    s = sample_density('gaussian', central, sigma, n, rng)
    return s.mean()


def compute_ice_thickness(is2_fb, h_s, rho_s, rho_i, rho_w, delta=None):
    return (rho_w * is2_fb - (rho_w - rho_s) * h_s) / (rho_w - rho_i)


def compute_snow_from_freeboards(is2_fb, cs2_fb, delta, rho_s=RHO_S_CENTRAL):
    eta_s = (1 + 0.51 * rho_s / 1000.0)**1.5
    return (is2_fb - cs2_fb) / (delta * eta_s)


if __name__ == '__main__':
    print("=" * 72)
    print("R3 Configuration v4 (regional ρ_i, Alex 29 April 2026 refinement)")
    print("=" * 72)
    print()
    print("PARAMETER DISTRIBUTION SUMMARY:")
    print("-" * 72)
    print(f"  1. IS2 total freeboard  : Gaussian, σ = {FB_SIGMA_RAW} m")
    print(f"  2. CS2 radar freeboard  : Gaussian, σ = {FB_SIGMA_RAW} m")
    print(f"  3. Snow density ρ_s     : Gaussian, μ = {RHO_S_CENTRAL}, σ = {RHO_S_SIGMA} (no clipping)")
    print(f"  4. Sea-ice density ρ_i  : Gaussian with regional Alexandrov parameters")
    print(f"                            FYI (E. Weddell, Ross): μ = {RHO_I_FYI_MU}, σ = {RHO_I_FYI_SIGMA}")
    print(f"                            MYI (W. Weddell only):   μ = {RHO_I_MYI_MU}, σ = {RHO_I_MYI_SIGMA}")
    print(f"                            soft upper cap = {RHO_I_UPPER_CAP} kg/m³")
    print(f"                            *** Alexandrov et al. (2010) ***")
    print(f"                            *** 940 cap: Timco & Frederking (1996) ***")
    print(f"                            Weddell Sea aggregate: sample-weighted")
    print(f"                            union of E. and W. Weddell sub-sectors")
    print(f"  5. Penetration δ        : Gaussian, μ = {DELTA_BASELINE}, σ = {DELTA_SIGMA}")
    print(f"                            clipped {DELTA_CLIP}")
    print(f"  6. Seawater ρ_w         : FIXED at {RHO_W_CENTRAL}")
    print("-" * 72)
    print()

    rng = np.random.default_rng(0)

    # FYI distribution
    s_fyi = sample_density('gaussian', RHO_I_FYI_MU, RHO_I_FYI_SIGMA, 500000, rng)
    print(f"  FYI (E. Weddell, Ross) with 940 cap:")
    print(f"    Mean = {s_fyi.mean():.2f}, Std = {s_fyi.std():.2f}")
    print(f"    Min = {s_fyi.min():.1f}, Max = {s_fyi.max():.1f}")
    print(f"    % at cap (940): {100*np.mean(s_fyi >= RHO_I_UPPER_CAP - 1e-6):.2f}%")
    print()

    # MYI distribution
    s_myi = sample_density('gaussian', RHO_I_MYI_MU, RHO_I_MYI_SIGMA, 500000, rng)
    print(f"  MYI (W. Weddell) with 940 cap:")
    print(f"    Mean = {s_myi.mean():.2f}, Std = {s_myi.std():.2f}")
    print(f"    Min = {s_myi.min():.1f}, Max = {s_myi.max():.1f}")
    print(f"    % at cap (940): {100*np.mean(s_myi >= RHO_I_UPPER_CAP - 1e-6):.3f}%")
    print()

    # Hydrostatic implications
    denom_fyi = RHO_W_CENTRAL - RHO_I_FYI_MU
    denom_myi = RHO_W_CENTRAL - RHO_I_MYI_MU
    denom_cap = RHO_W_CENTRAL - RHO_I_UPPER_CAP
    print(f"  Hydrostatic denominator (ρ_w − ρ_i):")
    print(f"    FYI mean: {denom_fyi:.1f} kg/m³,  amplification factor = {RHO_W_CENTRAL/denom_fyi:.2f}")
    print(f"    MYI mean: {denom_myi:.1f} kg/m³,  amplification factor = {RHO_W_CENTRAL/denom_myi:.2f}")
    print(f"    At cap:   {denom_cap:.1f} kg/m³,  amplification factor = {RHO_W_CENTRAL/denom_cap:.2f}")
    print()

    # Test the regional sampler
    print(f"  Regional sampler test (synthetic 4-segment input):")
    test_subs = np.array(['Weddell_W', 'Weddell_E', '', 'Weddell_W'])
    mu_test, sigma_test = get_regional_rho_i_params(test_subs)
    for i, ss in enumerate(test_subs):
        label = ss if ss else '(other/Ross)'
        print(f"    Segment {i} subsector={label:12s} -> μ={mu_test[i]:.1f}, σ={sigma_test[i]:.1f}")

    # δ sanity check
    d = sample_delta(500000, rng)
    print(f"\n  δ:   mean={d.mean():.3f}, std={d.std():.3f}, "
          f"min={d.min():.3f}, max={d.max():.3f}")

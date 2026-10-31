# Coupled Monte Carlo uncertainty analysis of Antarctic sea-ice thickness from laser–radar altimetry

Analysis code and derived data for:

> Liu, X., Tilling, R. L., Fraser, A. D., Corney, S. P., and Heil, P.
> **Bulk sea-ice density and Ku-band penetration co-dominate the parametric uncertainty of Antarctic sea-ice thickness retrieval from coupled laser–radar altimetry.**
> Resubmitted to *Remote Sensing of Environment*, 29 October 2026 (revision of manuscript RSE-D-26-03119).

Archived on Zenodo: https://doi.org/10.5281/zenodo.20409432. The citation will be updated on acceptance.

| Release | Manuscript version | Notes |
|---|---|---|
| `v2.0-resubmitted` (this branch) | Resubmission, 29 Oct 2026 | Results quoted below |
| [`v1.0-submitted`](https://github.com/xinlongliu0307/antarctic-sit-laser-radar-uncertainty/tree/v1.0-submitted) | First submission, 13 Jul 2026 | Superseded; kept for comparison ([compare](https://github.com/xinlongliu0307/antarctic-sit-laser-radar-uncertainty/compare/v1.0-submitted...v2.0-resubmitted)) |

## Overview

Snow depth retrieved from the difference between ICESat-2 laser and CryoSat-2 Ku-band radar freeboards depends on the assumed Ku-band penetration fraction δ, so snow depth and penetration cannot be determined separately from the two freeboards. This pipeline propagates that dependence: a coupled Monte Carlo framework draws freeboards, snow density, bulk sea-ice density and δ from their priors and re-derives snow depth in every realisation, for 65,165 CRYO2ICE matchups (the positive-freeboard subset of 82,341) over the Weddell and Ross seas in the austral winters of 2022–2025. The bulk sea-ice density prior is set at each matchup from the observed multi-year ice concentration (Melsheimer et al., 2023).

Within this error model:

- bulk sea-ice density and Ku-band penetration account for **40.9 %** and **35.1 %** of the propagated sea-ice thickness variance; freeboards and snow density for 14.5 %; parameter interactions for 9.5 % (shares are not renormalised);
- the two leading terms exchange rank at σ_δ ≈ 0.20–0.25, and σ_δ ≈ 0.10 would leave density as the single dominant term;
- at the 25 km segment scale of the CRISTAL requirements, the median sea-ice thickness uncertainty is 0.452 m (requirement 0.15 m, 3,443 segments); freeboard noise alone gives 0.068 m;
- replacing the AMSR-E/AMSR2 snow climatology in the CCI v4.0 and Cryo-TEMPO level-2 pre-processed products with CRYO2ICE snow increases sea-ice thickness by a per-matchup median of 0.352 m (CCI) and 0.406 m (Cryo-TEMPO) at δ = 0.70, and by 0.10–0.16 m at the empirical optimum δ* ≈ 0.55.

## Repository structure

```
├── README.md, LICENSE, CITATION.cff, requirements.txt
├── code/              Pipeline scripts, run in step order
│   ├── archive/       Superseded scripts kept for provenance (not part of the pipeline)
│   └── pbs/           PBS job scripts used on NCI Gadi
├── data/              Derived outputs sufficient to reproduce every table and figure
└── figures/           The twelve manuscript figures (PDF)
```

## Pipeline

`R3_config_v4.py` holds the physical constants, priors, region definitions and file paths, and is the only file to edit to run elsewhere. `R3_sweep_engine_v7.py` is the shared Monte Carlo engine (one-at-a-time and simultaneous perturbations, region-mean shares) imported by the decomposition and sweep scripts.

| Stage | Script | Purpose |
|---|---|---|
| Input | `R3_Step01_v2_add_subsectors.py` | Assign Weddell sub-sectors; positive-freeboard filter (65,165 matchups) |
| | `R3_Step12_download_melsheimer.py` | Download the University of Bremen ASCAT–AMSR2 Antarctic sea-ice type products (Melsheimer et al., 2023) |
| | `R3_Step13_icetype_collocation.py` | Per-matchup multi-year ice concentration and ice type (nearest 12.5 km cell, EPSG:3412); writes the Monte Carlo input |
| Monte Carlo | `R3_Step02_v7_mixture.py` | Experiments A (simultaneous) and B (one-at-a-time) under the per-matchup mixture density prior; interaction residual |
| | `R3_Step02_v5_coupled_20260830.py` | Sensitivity case: sector-rule density prior |
| | `R3_Step02_v6_icetype.py` | Sensitivity case: binary ice-type density prior |
| | `R3_Step02b_v7_shape_robustness.py` | Skew-normal versus Gaussian density components |
| | `R3_Step03_v7_rho_sweep.py` | Variance shares versus the sea-ice density spread σ_ρi |
| | `R3_Step03b_v7_delta_sweep.py` | Variance shares versus the penetration-fraction spread σ_δ (0.02–0.40) |
| | `R3_Step04_v7_experiment_C.py` | Snow-source comparison; retrieved thickness versus δ, with snow-density envelope |
| | `R3_Step15_state_dependent_delta.py` | State-dependent penetration-fraction priors (Appendix F) |
| | `R3_Step16_v7_alt_priors.py` | Alternative priors of Table 1 (common random numbers) |
| Figures | `R3_Step05_v7_figures.py` | Monte Carlo thickness figures (Figure 1) |
| | `R3_Step05_v7_variance_figures.py` | Variance-share figures (Figures 2, 3 and 3b) |
| | `R3_Step05b_fig04_penetration_v8.py` | Figure 4 (final version) |
| Snow replacement | `R3_Step08c_L2P_collocation_full.py` | Collocation of CCI v4.0 and Cryo-TEMPO L2P with CRYO2ICE, all winters (CCI 2022–2023, Cryo-TEMPO 2022–2025) |
| | `R3_Step09_snow_swap_v7.py` | Snow-replacement experiment, including the δ sweep from 0.40 to 1.00 |
| | `R3_Step10_snow_swap_figures_v9.py` | Snow scatter (Figure C.1) |
| | `R3_Step10b_dhi_figures_v10.py` | Δh_i distributions and Δh_i versus δ |
| | `R3_Step10c_heatmap_v11.py` | Sector × winter heatmap, with plotted values written to CSV |
| | `R3_Step14_amsr2_gridscale.py` | Same-day CRYO2ICE–AMSR2 comparison at matchup, 12.5 km and 25 km scales |
| CRISTAL | `R3_Step11_v7_segment_uncertainty.py` | 25 km segment-scale uncertainty against MRD-330 and MRD-380, parameters drawn once per segment |
| Checks | `R3_Step17_v7_text_checks.py` | Table values and consistency checks |
| | `R3_Step18_c2i_snow_consistency.py` | Consistency of the CRYO2ICE snow column with the companion study (found the bug fixed on 27 Sep 2026) |
| | `R3_Step19_v8_manuscript_numbers.py` | Every number in the manuscript, recomputed and keyed (`data/step19_keys.txt`) |
| | `R3_Step20_heatmap_numbers_v8.py` | Every number shown in the heatmap |

`code/archive/` holds scripts that are not part of the pipeline but document how it was built: the input verification (`R3_Step01_verify_input_data.py`), the August 2022 L2P proof of concept and its sector-correlation check (`R3_Step08_*`, `R3_Step08b_*`), and earlier figure versions. `code/archive/R3_patch_v8_snow_fix.py` documents the 27 September 2026 correction: the snow column first used in Step 09 carried no penetration factor (implied δ = 1.00) and was recomputed from the freeboards at δ = 0.70, as in the Monte Carlo; AU_SI12 flag codes are masked in Step 14. The scripts in `code/` already contain the fix.

## Data

| File | Contents |
|---|---|
| `R3_input_d070_icetype.csv.gz` | Per-matchup Monte Carlo input (65,165 rows): freeboards, sector and sub-sector, observed multi-year ice concentration and ice type |
| `R3_step02_v7_subsample_idx.npy` | Indices of the 5,000-matchup decomposition subsample (seed 42) |
| `R3_icetype_sector_composition.csv` | Ice-type composition of the matchups by sector |
| `R3_ExpA_d070_v7.npz` | Experiment A (simultaneous perturbation) output, mixture prior |
| `R3_ExpB_d070_v7.json` | Experiment B single-parameter shares and interaction residual by stratum, mixture prior (adopted) |
| `R3_ExpB_d070_v5.json`, `R3_ExpB_d070_v6.json` | The same under the sector-rule and binary ice-type priors (sensitivity cases) |
| `R3_ExpC_d070_v7.json` | Snow-source comparison summary |
| `R3_density_sweep_d070_v7.csv` | Values plotted in Figure 3 |
| `R3_delta_sweep_v7.csv` | Values plotted in Figure 3b |
| `R3_state_dependent_delta_v7.csv`, `R3_state_dependent_delta_v7.json` | Shares under state-dependent penetration priors (Appendix F) |
| `R3_segment_uncertainty_v7.csv.gz`, `R3_segment_summary_v7.json` | Per-segment snow-depth and thickness uncertainty, and summary, for the CRISTAL benchmark |
| `R3_shape_robustness_v7.json` | Shares under skew-normal density components |
| `R3_alt_priors_v7.csv`, `R3_alt_priors_central_hi_v7.csv` | Shares and central-value thickness under the alternative priors |
| `R3_penetration_sensitivity_d070_v7.csv` | Values plotted in Figure 4 |
| `R3_SIT_snow_swap_all_winters_v7.csv.gz` | Per-matchup snow-replacement results, all winters |
| `R3_snow_swap_delta_sweep_v7.csv` | Median Δh_i versus δ by product and sub-sector |
| `R3_SIT_snow_swap_summary_v7.txt` | Summary of the snow-replacement experiment |
| `R3_Fig_C2_03_sector_year_heatmap_values.csv` | Values plotted in the heatmap figure |
| `R3_amsr2_gridscale_v7.csv`, `R3_amsr2_gridscale_cells_v7.csv` | AMSR2 grid-scale comparison: summary and per-cell values |
| `step19_keys.txt` | Every manuscript number with the key used in the text |

`.csv.gz` files are read directly by `pandas.read_csv`. The pipeline itself reads the Monte Carlo input as a pandas pickle (`R3_input_d070_icetype.pkl`); to rerun from this repository, write it back with `pd.read_csv('data/R3_input_d070_icetype.csv.gz').to_pickle(...)` into the output folder set in `R3_config_v4.py`.

Upstream inputs are not redistributed: the CRYO2ICE along-track snow-depth dataset (https://doi.org/10.5281/zenodo.19220192), CCI v4.0 and Cryo-TEMPO L2P, AMSR2 AU_SI12, and the Melsheimer et al. (2023) ice-type product; sources are listed in the manuscript's Data availability statement.

## Changes since v1.0-submitted

- Sea-ice type is assigned per matchup from an ice-type product; the bulk density prior is a two-component mixture weighted by multi-year ice concentration (the sector rule and a binary class remain as sensitivity cases).
- Variance shares are fractions of the simultaneous-perturbation variance and are no longer renormalised; the 9.5 % interaction residual is reported with its pairwise components.
- New: penetration-uncertainty sweep to σ_δ = 0.40, state-dependent penetration priors, alternative priors, CRISTAL segment-scale benchmark, same-day AMSR2 grid-scale comparison, and the δ dependence of the snow-replacement result.
- Snow-replacement results recomputed with the CRYO2ICE snow column at δ = 0.70 (see `code/archive/R3_patch_v8_snow_fix.py`).
- Removed: the gridded bridge and spread-attribution steps and the Western Weddell parameterisation sensitivity of v1.0, which the resubmitted manuscript no longer reports.

## Environment

Run on NCI Gadi under Python 3.9.2 (`module load python3/3.9.2`) with the versions in `requirements.txt`. Paths in `R3_config_v4.py` point to `/g/data/gv90/...`. The PBS scripts in `code/pbs/` record the resources used and are not needed to run the Python scripts elsewhere.

```
pip install -r requirements.txt
cd code
python R3_Step01_v2_add_subsectors.py
```

## Conventions

- The positive-freeboard filter is applied once, in Step 01 v2; downstream scripts assume it.
- Sea-ice density draws are capped at 940 kg m⁻³; snow density is an unbounded Gaussian N(330, 70) kg m⁻³.
- Snow-replacement sign convention: Δh_i = h_i(CRYO2ICE snow) − h_i(AMSR-E/AMSR2 snow).
- Colours follow the Okabe–Ito palette.

## Licence

Code and derived data are released under the MIT License (see `LICENSE`).

## Contact

Xinlong Liu, Institute for Marine and Antarctic Studies, University of Tasmania, and Australian Antarctic Program Partnership. GitHub: [@xinlongliu0307](https://github.com/xinlongliu0307)

# Coupled Monte Carlo Uncertainty Analysis of Antarctic Sea-Ice Thickness from Laser–Radar Altimetry

Analysis pipeline and derived data products for:

> Liu, X., Fraser, A. D., Corney, S.P., Tilling, R.L., and Heil, P.
> **Bulk sea-ice density and Ku-band penetration co-dominate the uncertainty in Antarctic sea-ice thickness from coupled laser–radar altimetry retrieval.**
> Under review at *Remote Sensing of Environment*, 13 July 2026.

This repository will be archived on Zenodo: https://zenodo.org/records/20409432. The citation above will be updated on acceptance.

## Overview

This pipeline quantifies the uncertainty budget of Antarctic sea-ice thickness (SIT) retrieved from coupled laser–radar altimetry, using CRYO2ICE near-coincident CryoSat-2 (Ku-band radar) and ICESat-2 (laser) observations over the Weddell and Ross seas across the 2022–2025 winters. A coupled Monte Carlo error-propagation framework perturbs freeboard, snow depth, snow density, sea-ice density, and the Ku-band penetration fraction simultaneously, then decomposes the resulting SIT variance by contributor.

Within this error model, the two co-dominant propagated contributors to SIT uncertainty are bulk sea-ice density (42%) and the Ku-band penetration fraction (40%). A controlled snow-substitution experiment shows that replacing AMSR2-informed climatological snow with CRYO2ICE snow raises median SIT by 0.62 m against the CCI v4.0 baseline and 0.67 m against the Cryo-TEMPO baseline. The input set comprises 82,341 CRYO2ICE matchups; the Monte Carlo experiments use the 65,165-segment positive-freeboard subset.

## Repository structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── code/                 Pipeline scripts (versions used for the submitted manuscript)
│   ├── archive/          Superseded figure-script versions and one-off diagnostics
├── data/                 Derived outputs (Monte Carlo summaries, sweeps, substitution results)
```

## Pipeline

Scripts are run in numerical order. `R3_config_v4.py` holds all shared physical parameters, distributions, and file paths, and is the only file that needs editing to rerun the pipeline in a different environment.

| Script | Purpose |
|---|---|
| `R3_config_v4.py` | Shared configuration: physical constants, regional sea-ice density parameterisation (Alexandrov et al., 2010; 940 kg m⁻³ cap after Timco and Frederking, 1996), Monte Carlo settings, paths |
| `R3_Step01_verify_input_data.py` | Verify the CRYO2ICE collocation CSV (82,341 matchups) against the source-manuscript statistics before any Monte Carlo runs |
| `R3_Step01_v2_add_subsectors.py` | Assign Weddell sub-sectors and apply the positive-freeboard filter (65,165 segments retained) |
| `R3_Step02_v4_coupled.py` | Experiments A and B: coupled Monte Carlo with per-segment regional sea-ice density (western Weddell MYI 882 kg m⁻³; eastern Weddell and Ross FYI 916.7 kg m⁻³) |
| `R3_Step02b_shape_robustness.py` | Robustness of the variance decomposition to the assumed sea-ice density distribution shape |
| `R3_Step02c_western_weddell.py` | Western Weddell sub-sector sensitivity across four sea-ice density parameterisations |
| `R3_Step03_variance_decomposition_ExpB.py` | Experiment B: one-at-a-time perturbation variance decomposition |
| `R3_Step03b_delta_sweep.py` | Sensitivity sweep of the penetration-fraction uncertainty σ_δ (10,000 draws), locating the σ_δ ≈ 0.10–0.15 crossover |
| `R3_Step04_full_analysis.py` | Full analysis over the positive-freeboard subset |
| `R3_Step04_v3_experiment_C.py` | Experiment C: snow-source comparison and penetration sensitivity |
| `R3_Step05_v6_figures.py` | Publication figures for the uncertainty-budget experiments |
| `R3_Step06_bridge_script.py` | Bridge CRYO2ICE segments to the 216 × 216 EASE-2 50 km gridded products for the substitution experiment |
| `R3_Step07_v2_spread_attribution.py` | Spread attribution with consistent denominators, reported at both thickness and freeboard level |
| `R3_Step08_L2P_collocation_aug2022.py` | Proof of concept: nearest-neighbour collocation of CCI v4.0 and Cryo-TEMPO L2P along-track data with CRYO2ICE segments (August 2022) |
| `R3_Step08b_verify_sector_correlation.py` | Check whether the pooled snow correlation is inflated by the inter-sector Weddell–Ross contrast |
| `R3_Step08c_L2P_collocation_full.py` | Full-period L2P collocation (CCI v4.0: winters 2022–2023; Cryo-TEMPO: winters 2022–2025) |
| `R3_Step09_snow_swap_v5.py` | Controlled snow-substitution experiment: AMSR2-informed baselines (CCI v4.0, Cryo-TEMPO) with snow replaced by CRYO2ICE |
| `R3_Step10_snow_swap_figures_v8.py` | Publication figures for the substitution experiment |
| `R3_verify_table_summary_v2.py` | Extract pooled-years table values from the Step 09 output |

`code/archive/` retains superseded figure-script versions (`Step05_v5`, `Step10_v7`) and three short diagnostic scripts, kept for provenance; they are not part of the submitted pipeline.

## Data

`data/` contains the derived outputs sufficient to reproduce the manuscript tables and figures without rerunning the Monte Carlo:

| File | Contents |
|---|---|
| `R3_SIT_snow_swap_all_winters_v5.csv` | Per-segment snow-substitution results, all winters (82,341 rows; baseline and improvement scenarios for CCI v4.0 and Cryo-TEMPO) |
| `R3_SIT_snow_swap_summary_v5.txt` | Human-readable summary of the substitution experiment |
| `R3_density_sweep_d070_v4_coupled.csv` | Coupled sweep over sea-ice density uncertainty |
| `R3_delta_sweep_v1.csv` | Sweep over penetration-fraction uncertainty σ_δ |
| `R3_penetration_sensitivity_d070_v3.csv` | Penetration sensitivity at the δ = 0.70 reference scenario |
| `R3_ExpB_d070_v4.json`, `R3_ExpC_d070_v3.json` | Experiment B and C summary statistics |
| `R3_spread_attribution_v2.json` | Spread-attribution results |
| `R3_shape_robustness_v5.json` | Distribution-shape robustness results |
| `R3_western_weddell_sensitivity_v1.json` | Western Weddell parameterisation sensitivity |

Upstream inputs (CRYO2ICE collocation CSV from the companion snow-thickness pipeline, CCI v4.0 and Cryo-TEMPO L2P products, AMSR2 snow depth) are not redistributed here; sources are listed in the manuscript's Data Availability statement. The companion CRYO2ICE snow-thickness pipeline is at [xinlongliu0307](https://github.com/xinlongliu0307).

## Environment and reproducibility

The pipeline was run on NCI Gadi (project gv90) under Python 3.9 with the pinned versions in `requirements.txt`. Paths in `R3_config_v4.py` point to the Gadi filesystem (`/g/data/gv90/...`) and must be edited to rerun elsewhere. Scripts are deliberately kept as plain, sequential Python with no scheduler dependencies; each script prints its verification statistics on completion.

```
pip install -r requirements.txt
python code/R3_Step01_verify_input_data.py
```

## Key conventions

- Positive-freeboard filtering is applied once, in Step 01 v2; downstream scripts assume it.
- Sea-ice density draws are capped at 940 kg m⁻³; the cap binds for roughly 26% of FYI draws and under 1% of MYI draws, and the clipped-distribution effect is disclosed in the manuscript.
- The snow-substitution sign convention is Δh_i = h_i(improved) − h_i(baseline), with AMSR2-informed products as the baseline.
- Colours follow the Okabe-Ito palette (Weddell `#0072B2`, Ross `#D55E00`).

## License

Code and derived data are released under the MIT License (see `LICENSE`).

## Contact

Xinlong Liu — Institute for Marine and Antarctic Studies, University of Tasmania / Australian Antarctic Program Partnership.
GitHub: [@xinlongliu0307](https://github.com/xinlongliu0307)

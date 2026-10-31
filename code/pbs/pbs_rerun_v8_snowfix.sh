#!/bin/bash
#PBS -P gv90
#PBS -q normalbw
#PBS -l ncpus=1
#PBS -l mem=32GB
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/gv90
#PBS -l wd
#PBS -N v8_snowfix
#PBS -o /g/data/gv90/xl1657/phd/M3_workspace/logs/v8_snowfix.o
#PBS -e /g/data/gv90/xl1657/phd/M3_workspace/logs/v8_snowfix.e
# Reruns everything that depends on the C2I snow column or on the AU_SI12 values.
# The Monte Carlo (Steps 02-05, 11, 15, 16) is NOT affected and is not rerun.
set -euo pipefail
WS=/g/data/gv90/xl1657/phd/M3_workspace
cd "$WS/code"
module load python3/3.9.2
source "$HOME/cryo2ice_env/bin/activate"
PY="$HOME/cryo2ice_env/bin/python"

# back up the v7 outputs once
for f in "$WS"/output/L2P_collocation/R3_SIT_snow_swap_all_winters_v7.csv \
         "$WS"/output/L2P_collocation/R3_snow_swap_delta_sweep_v7.csv \
         "$WS"/output/L2P_collocation/R3_SIT_snow_swap_summary_v7.txt \
         "$WS"/output/R3_amsr2_gridscale_v7.csv "$WS"/output/R3_amsr2_gridscale_cells_v7.csv; do
  [ -f "$f" ] && [ ! -f "${f%.*}_pre_v8.${f##*.}" ] && cp "$f" "${f%.*}_pre_v8.${f##*.}"
done

"$PY" R3_patch_v8_snow_fix.py
"$PY" R3_Step09_snow_swap_v7.py
"$PY" R3_Step14_amsr2_gridscale.py
"$PY" R3_Step10_snow_swap_figures_v9.py
"$PY" R3_Step10b_dhi_figures_v10.py
"$PY" R3_Step19_v8_manuscript_numbers.py

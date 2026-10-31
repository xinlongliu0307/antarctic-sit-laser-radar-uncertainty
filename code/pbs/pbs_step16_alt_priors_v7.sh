#!/bin/bash
#PBS -P gv90
#PBS -q normalbw
#PBS -l ncpus=1
#PBS -l mem=16GB
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/gv90
#PBS -l wd
#PBS -N step16_altpri
#PBS -o /g/data/gv90/xl1657/phd/M3_workspace/logs/step16_alt_priors_v7.o
#PBS -e /g/data/gv90/xl1657/phd/M3_workspace/logs/step16_alt_priors_v7.e
set -euo pipefail

WS=/g/data/gv90/xl1657/phd/M3_workspace
cd "$WS/code"
module load python3/3.9.2
source "$HOME/cryo2ice_env/bin/activate"
PY="$HOME/cryo2ice_env/bin/python"

"$PY" "$WS/code/R3_Step16_v7_alt_priors.py"
"$PY" "$WS/code/R3_Step17_v7_text_checks.py"

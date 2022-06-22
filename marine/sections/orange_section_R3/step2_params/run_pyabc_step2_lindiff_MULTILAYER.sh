#!/bin/bash

#PBS -N R3_param_inversion_lindiff_MULTILAYER
#PBS -q comm_small_day
#PBS -j oe
#PBS -l nodes=10:ppn=10
#PBS -M charles.shobe@mail.wvu.edu
#PBS -m abe

source /shared/software/conda/conda_init.sh
conda activate $HOME/.conda/envs/charlie-env-2

python marine/sections/orange_section_R3/step2_params/R3_step2_NUMBA_infer_params_LinDiff_MULTILAYER.py

conda deactivate
rm -r marine/sections/orange_section_R3/step2_params/__pycache__

#!/bin/bash

#PBS -N R4_param_inversion_lindiff
#PBS -q comm_small_day
#PBS -j oe
#PBS -l nodes=10:ppn=10
#PBS -M charles.shobe@mail.wvu.edu
#PBS -m abe

source /shared/software/conda/conda_init.sh
conda activate $HOME/.conda/envs/charlie-env-2

python marine/sections/orange_section_R4/step2_params/R4_step2_NUMBA_infer_params_LinDiff.py

conda deactivate
rm -r marine/sections/orange_section_R4/step2_params/__pycache__

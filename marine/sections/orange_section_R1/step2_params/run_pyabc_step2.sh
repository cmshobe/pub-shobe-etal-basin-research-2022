#!/bin/bash

#PBS -N R1_param_inversion
#PBS -q comm_small_week
#PBS -j oe
#PBS -l nodes=10:ppn=10
#PBS -M charles.shobe@mail.wvu.edu
#PBS -m abe

source /shared/software/conda/conda_init.sh
conda activate $HOME/.conda/envs/charlie-env-2

python marine/sections/orange_section_R1/step2_params/R1_step2_NUMBA_infer_params.py

conda deactivate
rm -r marine/sections/orange_section_R1/step2_params/__pycache__

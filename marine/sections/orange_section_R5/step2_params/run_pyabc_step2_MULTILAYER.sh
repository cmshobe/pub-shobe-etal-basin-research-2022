#!/bin/bash

#PBS -N R5_param_inversion_MULTILAYER
#PBS -q comm_small_week
#PBS -j oe
#PBS -l nodes=1:ppn=40,walltime=72:00:00
#PBS -M charles.shobe@mail.wvu.edu
#PBS -m abe

source /shared/software/conda/conda_init.sh
conda activate $HOME/.conda/envs/charlie-env-2

python marine/sections/orange_section_R5/step2_params/R5_step2_NUMBA_infer_params_MULTILAYER.py

conda deactivate
rm -r marine/sections/orange_section_R5/step2_params/__pycache__

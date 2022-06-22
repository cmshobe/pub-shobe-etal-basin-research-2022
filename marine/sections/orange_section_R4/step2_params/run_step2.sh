#!/bin/bash

#PBS -N R4_single_model
#PBS -q standby
#PBS -j oe
#PBS -l nodes=1:ppn=1

source /shared/software/conda/conda_init.sh
conda activate $HOME/.conda/envs/charlie-env-2

python marine/sections/orange_section_R4/step2_params/R4_single_model_step2.py

conda deactivate

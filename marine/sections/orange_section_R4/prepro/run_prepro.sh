#!/bin/bash

#PBS -N test-prepro
#PBS -q debug
#PBS -j oe
#PBS -l nodes=1:ppn=1

source /shared/software/conda/conda_init.sh
conda activate $HOME/.conda/envs/charlie-env-2

python marine/sections/orange_section_R4/prepro/R4_prepro.py

conda deactivate

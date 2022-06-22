#!/bin/bash

#PBS -N R1_prepro
#PBS -q standby
#PBS -j oe
#PBS -l nodes=1:ppn=1

source /shared/software/conda/conda_init.sh
conda activate $HOME/.conda/envs/charlie-env-2

python marine/sections/orange_section_R1/prepro/R1_prepro.py

conda deactivate

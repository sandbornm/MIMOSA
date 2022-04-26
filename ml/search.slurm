#!/bin/bash

#SBATCH --account=h_oguz_lab_acc
#SBATCH --partition=pascal
#SBATCH --gres=gpu:2

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm/search-job.log



# Max job duration
#SBATCH --job-name=mimosa_search

# Communication
#SBATCH --mail-user=zachary.a.stoebner@vanderbilt.edu
#SBATCH --mail-type=ALL

ml Anaconda3/2021.11
conda activate MIMOSA-ML

python3 search.py -g 2

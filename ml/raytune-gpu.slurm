#!/bin/bash

#SBATCH --account=h_oguz_lab_acc
#SBATCH --partition=pascal
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G

#SBATCH --time=5-00:00:00
#SBATCH --output=slurm/raytune-gpu.log
#SBATCH --job-name=mimosa_search

# Communication
#SBATCH --mail-user=zachary.a.stoebner@vanderbilt.edu
#SBATCH --mail-type=ALL

ml Anaconda3/5.0.1
source activate MIMOSA-ML

python3 raytune.py -x /scratch/stoebnza/MIMOSA/ml/data/imgs -y /scratch/stoebnza/MIMOSA/ml/data/config_multilabel.csv -c /data/h_oguz_lab/stoebnza/MIMOSA/ray_results -g 2

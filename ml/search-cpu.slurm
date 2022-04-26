#!/bin/bash

#SBATCH --account=h_oguz_lab

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=5-00:00:00
#SBATCH --output=slurm/search-cpu.log
#SBATCH --job-name=mimosa_search

# Communication
#SBATCH --mail-user=zachary.a.stoebner@vanderbilt.edu
#SBATCH --mail-type=ALL

ml Anaconda3/5.0.1
source activate MIMOSA-ML

python3 search.py -x /scratch/stoebnza/MIMOSA/ml/data/imgs  -y /scratch/stoebnza/MIMOSA/ml/data/config_multilabel.csv  -c /data/h_oguz_lab/stoebnza/MIMOSA/ray_results -g 0

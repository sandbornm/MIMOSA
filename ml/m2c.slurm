#!/bin/bash

#SBATCH --account=h_oguz_lab_acc
#SBATCH --partition=pascal
#SBATCH --gres=gpu:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G

#SBATCH --time=0-05:00:00
#SBATCH --output=slurm/m2c-gpu.log
#SBATCH --job-name=mimosa_m2c

# Communication
#SBATCH --mail-user=zachary.a.stoebner@vanderbilt.edu
#SBATCH --mail-type=ALL

ml Anaconda3/5.0.1
source activate MIMOSA-ML

X=/scratch/stoebnza/MIMOSA/ml/data/imgs
Y=/scratch/stoebnza/MIMOSA/ml/data/config_multilabel.csv
N=M2C_resnext50_top0
M=predict
I=image
A=resnext_50
B=16
C=13
E=50
L=1e-3
O=adam
S1=64
S2=64
V=0.2
VA=dense
python run.py -x $X -y $Y -n $N -m $M -i $I -a $A -b $B -cl $C -e $E -l $L -o $O -s $S1 $S2 -v $V -va $VA

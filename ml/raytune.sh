#X=/Users/Zachary/Desktop/School/Vandy/Masters/M_Year1/Spring_2022/CS_8395_04/project/MIMOSA/ml/data/imgs
#Y=/Users/Zachary/Desktop/School/Vandy/Masters/M_Year1/Spring_2022/CS_8395_04/project/MIMOSA/ml/data/config_multilabel.csv

X=/home/zstoebs/Desktop/Research/projects/mimosa_2022/MIMOSA/ml/data/imgs  # path to examples
Y=/home/zstoebs/Desktop/Research/projects/mimosa_2022/MIMOSA/ml/data/config_multilabel.csv  # path to labels
C=rays/

python3 raytune.py -x $X -y $Y -c $C

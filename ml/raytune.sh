#X=/Users/Zachary/Desktop/School/Vandy/Masters/M_Year1/Spring_2022/CS_8395_04/project/MIMOSA/ml/data/imgs
#Y=/Users/Zachary/Desktop/School/Vandy/Masters/M_Year1/Spring_2022/CS_8395_04/project/MIMOSA/ml/data/config_multilabel.csv

X=/home/michael2/MIMOSA/ml/data/imgs
Y=/home/michael2/MIMOSA/ml/data/config_multilabel.csv

python3 raytune.py -x $X  -y $Y -g 2

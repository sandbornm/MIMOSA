#! /bin/bash

#python run.py -x data/imgs -y data/config_multilabel.csv -n M2C_debug -i image -a resnext_50
#caffeinate python run.py -x data/bins -y data/config_multilabel.csv -n M2C_ff_1 -i bytes -a ff -hi 1024 1024 512 512
#caffeinate python run.py -x data/imgs -y data/config_multilabel.csv -n M2C_resnext50_0 -i image -a resnext_50

# training
X=data/bp_sample_imgs  # path to examples
Y=data/config_multilabel.csv  # path to labels
N=M2C_resnext50_top1  # experiment name
M=train  # mode [train* | cv* | predict]  *=requires labels csv, o/w doesn't
I=image  # modality [image | bytes]
A=resnext_50  # architecture *see util/__init__.py
B=16  # batch size
C=13  # number of classes (aka dimensionality of the output vector)
E=50  # number of epochs (only applies to train and cross val modes)
L=1e-3  # learning rate (only applies to train and cross val modes)
O=adam  # optimizer [adam | sgd | LMSprop]
S1=64  # input size dim 1
S2=64  # input size dim 2
V=0.2  # percent validation
VA=dense  # output layer type [dense | branch]

# the command
python run.py -x $X -y $Y -n $N -m $M -i $I -a $A -b $B -cl $C -e $E -l $L -o $O -s $S1 $S2 -v $V -va $VA

# prediction
X=data/bp_sample_imgs  # path to examples
N=M2C_resnext50_top1_BP_pred  # experiment name
M=predict  # mode [train* | cv* | predict]  *=requires labels csv, o/w doesn't
F=cp/M2C_resnext50_top1/M2C_resnext50_top1_final.pth  # path to saved model (required for test and predict modes)
I=image  # modality [image | bytes]
A=resnext_50  # architecture *see util/__init__.py
B=16  # batch size
C=13  # number of classes (aka dimensionality of the output vector)
S1=64  # input size dim 1
S2=64  # input size dim 2
VA=dense  # output layer type [dense | branch]

# the command
python run.py -x $X -n $N -f $F -m $M -i $I -a $A -b $B -cl $C -s $S1 $S2 -va $VA

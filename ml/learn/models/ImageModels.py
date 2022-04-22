import torch
import torch.nn as nn
from torchvision import models

criterion = nn.BCELoss()


class ResNet34MultilabelClassifier(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        """
        Image multilabel classifier using ResNet with a binary classifier leaf for each class
        https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        """
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)
        resnet.fc = nn.Identity()  # drop last layer

        self.leaf = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=1)
        )
        self.leaves = nn.ModuleList([self.leaf] * n_classes)

        self.base_model = resnet
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        return torch.cat([self.sigmoid(l(x)) for l in self.leaves], dim=1)


class ResNeXt50MultilabelClassifier(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        """
        Image multilabel classifier using ResNeXt
        https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        """
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=pretrained)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.base_model(x))

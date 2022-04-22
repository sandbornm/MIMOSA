import torch.nn as nn
from torchvision import models

criterion = nn.BCELoss()


class BytesMultilabelClassifier(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        """
        Bytes-vector multilabel classifier using ResNeXt
        """
        super().__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.base_model(x))

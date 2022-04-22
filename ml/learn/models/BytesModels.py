import torch
import torch.nn as nn

criterion = nn.BCELoss()


class FFBytesMultilabelClassifier(nn.Module):
    def __init__(self, n_input, n_classes):
        """
        Bytes-vector multilabel classifier using ResNeXt
        """
        super().__init__()
        self.base_model = nn.Sequential(

        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.base_model(x))


class Conv1DBytesMultilabelClassifier(nn.Module):
    def __init__(self, n_classes):
        """
        Bytes-vector multilabel classifier using ResNeXt
        """
        super().__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.base_model(x))

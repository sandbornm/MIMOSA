import torch
import torch.nn as nn
import logging

criterion = nn.BCELoss()


class FFBytesMultilabelClassifier(nn.Module):
    def __init__(self, name, n_input, n_classes, n_hidden=None, variant: str='dense'):
        """
        Bytes-vector multilabel classifier using ResNeXt

        Parameters:
            n_input = input vector size
            n_classes = number of output classes
            n_hidden = list of number of hidden features where first entry is the number of conv filters
            variant = model variant
        """
        super().__init__()
        if n_hidden is None:
            n_hidden = [512]

        self.name = name
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.variant = variant

        self.base_model = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=n_hidden[0]),
            nn.ReLU()
        )

        if len(n_hidden) > 1:
            for i, n in enumerate(n_hidden[1:], start=1):
                self.base_model.append(nn.Linear(in_features=n_hidden[i-1], out_features=n_hidden[i]))
                self.base_model.append(nn.ReLU())

        if variant.lower() == 'dense':
            self.base_model.append(nn.Linear(in_features=n_hidden[-1], out_features=n_classes))
        elif variant.lower() == 'branch':
            self.leaf = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=n_hidden[-1], out_features=1)
            )
            self.leaves = nn.ModuleList([self.leaf] * n_classes)
        else:
            raise ValueError('Unknown variant: ', variant)

        self.sigmoid = nn.Sigmoid()
        logging.info('Created %s variant of feed-forward bytes multilabel classifier' % variant)

    def forward(self, x):
        x = self.base_model(x)

        if self.variant.lower() == 'branch':
            return torch.cat([self.sigmoid(l(x)) for l in self.leaves], dim=1)

        return self.sigmoid(x)

    def reinit(self):
        for layer in self.base_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class Conv1DBytesMultilabelClassifier(nn.Module):
    def __init__(self, name, n_classes, n_hidden=None, variant: str='dense'):
        """
        Bytes-vector multilabel classifier using ResNeXt

        Parameters:
            n_classes = number of output classes
            n_hidden = list of number of hidden features where first entry is the number of conv filters
            variant = model variant
        """
        super().__init__()
        if n_hidden is None:
            n_hidden = [32, 512]

        self.name = name
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.variant = variant

        self.base_model = nn.Sequential(
            nn.Conv1d(1, n_hidden[0], 16, stride=2),
            nn.BatchNorm1d(n_hidden[0]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(n_hidden[0]),
            nn.Flatten(),
            nn.Linear(in_features=n_hidden[0]**2, out_features=n_hidden[1]),
            nn.ReLU()
        )

        if len(n_hidden) > 2:
            for i, n in enumerate(n_hidden[2:], start=2):
                self.base_model.append(nn.Linear(in_features=n_hidden[i-1], out_features=n_hidden[i]))
                self.base_model.append(nn.ReLU())

        if variant.lower() == 'dense':
            self.base_model.append(nn.Linear(in_features=n_hidden[-1], out_features=n_classes))
        elif variant.lower() == 'branch':
            self.leaf = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=n_hidden[-1], out_features=1)
            )
            self.leaves = nn.ModuleList([self.leaf] * n_classes)
        else:
            raise ValueError('Unknown variant: ', variant)

        self.sigmoid = nn.Sigmoid()
        logging.info('Created %s variant of 1D conv bytes multilabel classifier' % variant)

    def forward(self, x):
        x = self.base_model(x)

        if self.variant.lower() == 'branch':
            return torch.cat([self.sigmoid(l(x)) for l in self.leaves], dim=1)

        return self.sigmoid(x)

    def reinit(self):
        for layer in self.base_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

import torch
import torch.nn as nn
from torchvision import models

criterion = nn.BCELoss()


class ResNet34MultilabelClassifier(nn.Module):
    def __init__(self, name, n_classes, pretrained=False, variant: str='dense'):
        """
        Image multilabel classifier using ResNet with a binary classifier leaf for each class
        https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        """
        super().__init__()
        self.n_classes = n_classes
        self.variant = variant

        if '18' in name.lower():
            resnet = models.resnet18(pretrained=pretrained)
        elif '34' in name.lower():
            resnet = models.resnet34(pretrained=pretrained)
        elif '50' in name.lower():
            resnet = models.resnet34(pretrained=pretrained)
        elif '101' in name.lower():
            resnet = models.resnet101(pretrained=pretrained)
        elif '152' in name.lower():
            resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError('Model size spec not found in name: ', name)

        in_features = resnet.fc.in_features

        if variant.lower() == 'dense':
            resnet.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
            )
        elif variant.lower() == 'branch':
            resnet.fc = nn.Identity()  # drop last layer
            self.leaf = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=in_features, out_features=1)
            )
            self.leaves = nn.ModuleList([self.leaf] * n_classes)
        else:
            raise ValueError('Unknown variant: ', variant)

        self.base_model = resnet
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)

        if self.variant.lower() == 'branch':
            return torch.cat([self.sigmoid(l(x)) for l in self.leaves], dim=1)

        return self.sigmoid(x)


class ResNeXt50MultilabelClassifier(nn.Module):
    def __init__(self, name, n_classes, pretrained=False, variant: str='dense'):
        """
        Image multilabel classifier using ResNeXt
        https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        """
        super().__init__()
        self.n_classes = n_classes
        self.variant = variant

        if '50' in name.lower():
            resnet = models.resnext50_32x4d(pretrained=pretrained)
        elif '101' in name.lower():
            resnet = models.resnext101_32x8d(pretrained=pretrained)
        else:
            raise ValueError('Model size spec not found in name: ', name)

        in_features = resnet.fc.in_features

        if variant.lower() == 'dense':
            resnet.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
            )
        elif variant.lower() == 'branch':
            resnet.fc = nn.Identity()  # drop last layer
            self.leaf = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=in_features, out_features=1)
            )
            self.leaves = nn.ModuleList([self.leaf] * n_classes)
        else:
            raise ValueError('Unknown variant: ', variant)

        self.base_model = resnet
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)

        if self.variant.lower() == 'branch':
            return torch.cat([self.sigmoid(l(x)) for l in self.leaves], dim=1)

        return self.sigmoid(x)


class ConvNeXtMultilabelClassifier(nn.Module):
    def __init__(self, name, n_classes, pretrained=False, variant: str='dense'):
        """
        Image multilabel classifier using ConvNeXt
        https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        """
        super().__init__()
        self.name = name
        self.n_classes = n_classes
        self.variant = variant

        if 'tiny' in name.lower():
            convnet = models.convnext_tiny(pretrained=pretrained)
        elif 'small' in name.lower():
            convnet = models.convnext_small(pretrained=pretrained)
        elif 'base' in name.lower():
            convnet = models.convnext_base(pretrained=pretrained)
        elif 'large' in name.lower():
            convnet = models.convnext_large(pretrained=pretrained)
        else:
            raise ValueError('Model size spec not found in name: ', name)

        in_features = convnet.classifier[-1].in_features

        convnet.classifier[-1] = nn.Identity() # drop last layer
        if variant.lower() == 'dense':
            convnet.classifier.append(nn.Dropout(p=0.2))
            convnet.classifier.append(nn.Linear(in_features=in_features, out_features=n_classes))
        elif variant.lower() == 'branch':
            self.leaf = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=in_features, out_features=1)
            )
            self.leaves = nn.ModuleList([self.leaf] * n_classes)
        else:
            raise ValueError('Unknown variant: ', variant)

        self.base_model = convnet
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)

        if self.variant.lower() == 'branch':
            return torch.cat([self.sigmoid(l(x)) for l in self.leaves], dim=1)

        return self.sigmoid(x)

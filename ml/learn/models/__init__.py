import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import ImageModels
import BytesModels


def merge_dicts(dict_1, dict_2):
    """
    Merge dicts with common keys as list
    """
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = dict_1[key] + value  # since dict_1 val overwritten by above merge
    return dict_3


def build_model(args, n_classes):
    modality = args.modality.lower()
    arch = args.arch.lower()
    if modality == 'image':
        if arch == 'resnext':
            net = ImageModels.ResNeXt50MultilabelClassifier(n_classes, pretrained=args.pretrain)
        elif arch == 'resnet':
            net = ImageModels.ResNet34MultilabelClassifier(n_classes, pretrained=args.pretrain)
        else:
            raise ValueError('Unknown architecture: ', args.arch)
        criterion = ImageModels.criterion
    elif modality == 'bytes':
        if arch == 'conv':
            net = BytesModels.Conv1DBytesMultilabelClassifier(n_classes)
        elif arch == 'ff':
            net = BytesModels.FFBytesMultilabelClassifier(args.size, n_classes)
        else:
            raise ValueError('Unknown architecture: ', args.arch)
        criterion = BytesModels.criterion
    else:
        raise ValueError('Unknown modality: ', args.modality)

    return net, criterion


# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': [precision_score(y_true=target, y_pred=pred, average='micro', zero_division=1)],
            'micro/recall': [recall_score(y_true=target, y_pred=pred, average='micro', zero_division=1)],
            'micro/f1': [f1_score(y_true=target, y_pred=pred, average='micro', zero_division=1)],
            'macro/precision': [precision_score(y_true=target, y_pred=pred, average='macro', zero_division=1)],
            'macro/recall': [recall_score(y_true=target, y_pred=pred, average='macro', zero_division=1)],
            'macro/f1': [f1_score(y_true=target, y_pred=pred, average='macro', zero_division=1)],
            'samples/precision': [precision_score(y_true=target, y_pred=pred, average='samples', zero_division=1)],
            'samples/recall': [recall_score(y_true=target, y_pred=pred, average='samples', zero_division=1)],
            'samples/f1': [f1_score(y_true=target, y_pred=pred, average='samples', zero_division=1)],
            'accuracy': [accuracy_score(y_true=target, y_pred=pred)]
            }


# https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f
def train_epoch(net, device, dataloader, loss_fn, optimizer, **kwargs):
    n_train = kwargs['n_train']
    epoch = kwargs['epoch']
    epochs = kwargs['epochs']

    train_loss = 0.0
    train_metrics = None
    net.train()
    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in dataloader:
            examples = batch['example']
            labels = batch['label']
            labels = torch.squeeze(labels).float()
            examples, labels = examples.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(examples)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * examples.size(0)

            metrics = calculate_metrics(output.detach().cpu(), labels.detach().cpu().numpy())
            train_metrics = metrics if not train_metrics else merge_dicts(train_metrics, metrics)

            pbar.update(examples.shape[0])

    return train_loss, train_metrics


def val_epoch(net, device, dataloader, loss_fn):
    val_loss = 0.0
    val_metrics = None
    net.eval()
    for batch in dataloader:
        examples = batch['example']
        labels = batch['label']
        labels = torch.squeeze(labels).float()
        examples, labels = examples.to(device), labels.to(device)

        with torch.no_grad():
            output = net(examples)
            loss = loss_fn(output, labels)
            val_loss += loss.item() * examples.size(0)

        metrics = calculate_metrics(output.detach().cpu(), labels.detach().cpu().numpy())
        val_metrics = metrics if not val_metrics else merge_dicts(val_metrics, metrics)

    return val_loss, val_metrics

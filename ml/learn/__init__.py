import logging
from os.path import join
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import RepeatedKFold

from ray import tune

from comet_ml import Experiment  # importing after torch disables autologging

import util

from . import datasets
from . import models


def log_metrics(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch):
    with experiment.train():
        experiment.log_metric('Loss', train_loss, epoch=epoch)
        for metric, values in train_metrics.items():
            experiment.log_metric(metric, np.mean(values), epoch=epoch)
    with experiment.validate():
        experiment.log_metric('Loss', val_loss, epoch=epoch)
        for metric, values in val_metrics.items():
            experiment.log_metric(metric, np.mean(values), epoch=epoch)


def log_reports(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch):
    with experiment.train():
        experiment.log_metric('Loss', train_loss, epoch=epoch)
    with experiment.validate():
        experiment.log_metric('Loss', val_loss, epoch=epoch)

    labels = train_metrics.keys()
    scores = train_metrics[0].keys()
    for label in labels:
        for score in scores:
            with experiment.train():
                experiment.log_metric(label+'/'+score, np.mean(train_metrics[label][score]), epoch=epoch)
            with experiment.validate():
                experiment.log_metric(label+'/'+score, np.mean(val_metrics[label][score]), epoch=epoch)


def merge_metrics(dict_1: dict, dict_2: dict):
    """
    Merge dicts with common keys as list
    """
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = dict_1[key] + value  # since dict_1 val overwritten by above merge
    return dict_3


def merge_reports(master: dict, report: dict):
    """
    Merge classification reports into a master list
    """
    keys = master.keys()
    ret = copy.deepcopy(master)
    for key in keys:
        scores = report[key]
        for score, value in scores.items():
            ret[key][score] += [value]

    return ret

# generate a classification report
def report(pred, target, configs, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return classification_report(target, pred, output_dict=True, target_names=configs, zero_division=1)


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
def train_epoch(net, device, dataloader, loss_fn, classes, optimizer, **kwargs):
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
            train_metrics = metrics if not train_metrics else merge_metrics(train_metrics, metrics)

            # metrics = report(output.detach().cpu(), labels.detach().cpu().numpy(), classes)
            # train_metrics = metrics if not train_metrics else merge_reports(train_metrics, metrics)

            pbar.update(examples.shape[0])

    return train_loss, train_metrics


def val_epoch(net, device, dataloader, loss_fn, classes):
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
        val_metrics = metrics if not val_metrics else merge_metrics(val_metrics, metrics)

        # metrics = report(output.detach().cpu(), labels.detach().cpu().numpy(), classes)
        # val_metrics = metrics if not val_metrics else merge_reports(val_metrics, metrics)

    return val_loss, val_metrics


def cross_val(net,
              dataset,
              device,
              epochs,
              batch_size,
              lr,
              criterion,
              exp_name
              ):
    """
    Repeated K-Fold cross-validation for a PyTorch classifier
    """
    # comet setup
    experiment = Experiment(
        api_key="k86kE4n1wy7wQkkCmvZeFAV3M",
        project_name="mimosa",
        workspace="zstoebs",
    )
    experiment.set_name(exp_name)

    hyper_params = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    experiment.log_parameters(hyper_params)

    classes = dataset.classes

    foldperf = {}
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    for fold, (train_idx, val_idx) in enumerate(cv.split(np.arange(len(dataset)))):

        logging.info('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        model = copy.deepcopy(net)
        model.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            train_loss, train_metrics = train_epoch(model, device, train_loader, criterion, classes, optimizer)
            val_loss, val_metrics = val_epoch(model, device, val_loader, criterion, classes)

            train_loss /= len(train_sampler)
            val_loss /= len(val_sampler)

            log_metrics(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch)
            # log_reports(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch)

            mean_train_acc = np.mean(train_metrics['accuracy'])
            mean_val_acc = np.mean(val_metrics['accuracy'])

            scheduler.step(mean_val_acc)

            logging.info(
                "Epoch:{}/{} \n"
                " AVG Training Loss:{:.3f} \n"
                "AVG Test Loss:{:.3f} \n"
                "AVG Training Acc {:.2f} \n"
                "AVG Test Acc {:.2f}".format(
                    epoch + 1,
                    epochs,
                    train_loss,
                    val_loss,
                    mean_train_acc,
                    mean_val_acc))

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(mean_train_acc)
            history['val_acc'].append(mean_val_acc)

        foldperf['fold{}'.format(fold + 1)] = history


def train(net,
          dataset,
          device,
          epochs,
          batch_size,
          lr,
          val_percent,
          frequency,
          criterion,
          exp_name
          ):
    """
    Standard iterative training for a PyTorch classifier
    """

    # comet setup
    experiment = Experiment(
        api_key="k86kE4n1wy7wQkkCmvZeFAV3M",
        project_name="mimosa",
        workspace="zstoebs",
    )
    experiment.set_name(exp_name)

    save_dir = join('cp', exp_name)

    net.train()

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,
                            drop_last=True)

    hyper_params = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_size": n_train,
        "val_size": n_val,
        "device": device.type,
    }
    experiment.log_parameters(hyper_params)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    classes = dataset.classes

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    for epoch in range(epochs):
        train_kwargs = {'n_train': n_train, 'epoch': epoch, 'epochs': epochs}
        train_loss, train_metrics = train_epoch(net, device, train_loader, criterion, classes, optimizer, **train_kwargs)
        val_loss, val_metrics = val_epoch(net, device, val_loader, criterion, classes)

        train_loss /= n_train
        val_loss /= n_val

        log_metrics(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch)
        # log_reports(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch)

        mean_train_acc = np.mean(train_metrics['accuracy'])
        mean_val_acc = np.mean(val_metrics['accuracy'])

        scheduler.step(mean_val_acc)

        logging.info(
            "Epoch:{}/{} \n "
            "AVG Training Loss:{:.3f} \n "
            "AVG Val Loss:{:.3f} \n "
            "AVG Training Acc {:.2f} \n "
            "AVG Val Acc {:.2f}".format(
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                mean_train_acc,
                mean_val_acc))

        # save checkpoint
        if frequency and epoch % frequency == 0:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)
    #         util.makedir(save_dir)
    #         torch.save(net.state_dict(),
    #                   join(save_dir, f'{exp_name}_epoch{epoch + 1}.pth'))
    #         logging.info(f'Checkpoint {epoch + 1} saved !')

    # save final model
    util.makedir(save_dir)
    torch.save(net.state_dict(),
               join(save_dir, f'{exp_name}_final.pth'))
    logging.info(f'Final model saved !')

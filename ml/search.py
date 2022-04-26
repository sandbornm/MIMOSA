from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from learn import train


def search(checkpoint_dir='cp', data_dir=None):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(5e-5, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    return

if __name__ == '__main__':

    search(data_dir='data/imgs')

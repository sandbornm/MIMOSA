import argparse

import torch

import optuna
from learn import train

# architectures to search
vision_archs = ['resnext_50', 'resnext_101',
                'resnet_18', 'resnet_32', 'resnet_50', 'resnet_101', 'resnet_152',
                'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']


def get_args():
    parser = argparse.ArgumentParser(description='Run a hyperparam search on a multilabel classifier')
    parser.add_argument('--samples', '-s', type=int, default=20, help='number of times to sample grid search params')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='max number of epochs that any trial can run')
    parser.add_argument('--gpus', '-g', type=int, default=1, help='number of gpus per trial')
    parser.add_argument('--examples_dir', '-x', type=str, default="", help='path to image dir')
    parser.add_argument('--labels_csv', '-y', type=str, default="", help='path to labels csv')
    parser.add_argument('--cp_dir', '-c', type=str, default='.', help='checkpoint base dir. default=./cp')

    return parser.parse_args()


# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
def search(examples_dir, labels_csv, cp_dir='~/ray/results', num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    torch.cuda.empty_cache()

    study = optuna.create_study(direction="maximize")
    study.optimize(train, n_trials=100, timeout=600)


if __name__ == "__main__":
    args = get_args()

    # You can change the number of GPUs per trial here:
    search(args.examples_dir, args.labels_csv, args.cp_dir, num_samples=args.samples, max_num_epochs=args.epochs, gpus_per_trial=args.gpus)

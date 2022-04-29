from functools import partial
import argparse

import torch

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

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

    args = {
        'examples_dir': examples_dir,
        'labels_csv': labels_csv,
        'cp_dir': cp_dir,
        'name': 'vhps',
        'load': False,
        'frequency': 0,
        'mode': 'train',
        'modality': 'image',
        'arch': tune.grid_search(vision_archs),
        'size': [tune.choice([64, 128, 256, 512, 1024, 2048, 4096]),
                 tune.choice([64, 128, 256, 512, 1024, 2048, 4096])],
        'epochs': tune.choice([10, 20, 50, 80, 100]),
        "batchsize": tune.choice([2, 4, 8, 16, 32, 64]),
        "lr": tune.loguniform(5e-5, 1e-1),
        'val': tune.quniform(0.01, 0.3, 0.01),
        'pretrain': tune.choice([True, False]),
        'variant': tune.choice(['dense', 'branch']),
        'hidden': [512],
        # 'hidden': [tune.choice([32, 64, 512, 1024, 2048]) for _ in range(tune.randint(1, 10))],
        'optim': tune.choice(['adam', 'sgd', 'rmsprop']),
    }
    scheduler = ASHAScheduler(
        metric="ranking",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", 'ranking', "training_iteration"])
    result = tune.run(
        partial(train, tuning=True),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=args,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=cp_dir,
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        )
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    args = get_args()

    # You can change the number of GPUs per trial here:
    search(args.examples_dir, args.labels_csv, args.cp_dir, num_samples=args.samples, max_num_epochs=args.epochs, gpus_per_trial=args.gpus)

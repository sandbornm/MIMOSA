import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_parser():
    """
    Generic deep learning parser generator that can extensively add new arguments as needed
    """
    parser = argparse.ArgumentParser(description='Train a multilabel classifier')

    # metasetup
    parser.add_argument('--examples_dir', '-x', type=str, default="", help='path to image dir')
    parser.add_argument('--labels_csv', '-y', type=str, default="", help='path to labels csv')
    parser.add_argument('--name', '-n', type=str, default='M2C', help='Experiment name. default: M2C')
    parser.add_argument('--load', '-f', type=str, default='', help='path to net to load')
    parser.add_argument('--frequency', '-q', type=int, default=0, help='save frequency. default= 0 (end of training)')
    parser.add_argument('--mode', '-m', type=str, default='train',
                        help='training mode [train | cross_val | test | predict]. default: train')
    parser.add_argument('--modality', '-i', type=str, default='image',
                        help='input modality [image | bytes]. default: image')
    parser.add_argument('--arch', '-a', type=str, default='resnext_50',
                        help='architecture type given modality. image: [resnext_(50 | 101) | resnet_(18 | 32 | 50 | 101 | 152) | convnext_(tiny | small | base | large)], bytes: [conv | ff]. default: resnet')
    parser.add_argument('--cp_dir', '-c', type=str, default='.', help='checkpoint base dir. default=./cp')
    parser.add_argument('--n_classes', '-cl', type=int, choices=range(2, 100), default=2,
                        help='number of classes. default= 2')

    # hyperparams
    parser.add_argument('--size', '-s', type=int, nargs='+', default=[64, 64],
                        help='resize images to these dims (multiplied for bytes). default: --size 64 64')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='number of epochs. default: 20', dest='epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size. default: 16',
                        dest='batchsize')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-3,
                        help='Learning rate. default: 1e-3', dest='lr')
    parser.add_argument('-v', '--val_percent', metavar='V', type=float, default=0.2,
                        help='percent of training data to use for validation set. default: 0.2', dest='val')
    parser.add_argument('--pretrain', action='store_true', help='use pretrained vision models')
    parser.add_argument('--variant', '-va', type=str, default='dense',
                        help='model variant [dense | branch]. default: dense')
    parser.add_argument('--hidden', '-hi', type=int, nargs='+', default=[512],
                        help='dimensions for hidden dense layers. default: --hidden 512')
    parser.add_argument('--optim', '-o', type=str, default='adam',
                        help='optimizer to use [adam | sgd | rmsprop]. default: adam')
    parser.add_argument('--loss', '-lo', type=str, default='bce', help='loss fnc to use [bce | cce]. default: bce')

    return parser


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(
        estimator,
        title,
        X,
        y,
        scoring,
        axes=None,
        ylim=None,
        cv=None,
        n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        scoring=scoring,
        return_times=True
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

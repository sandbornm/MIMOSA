import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold, learning_curve, ShuffleSplit
from sklearn.metrics import accuracy_score, recall_score

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from util import plot_learning_curve


# bit_mapping
PROCESS_DETECTION = 0
DEBUGGER_PRESENT = 1
CPUID = 2
RDTSC = 3
CPU_COUNT = 4
INVALID_INST = 5
TICK_COUNT = 6
HCI = 7
BIOS = 8
DRIVER_CHECK = 9
SCSI_CHECK = 10
HDD_SIZE = 11
MEM_SIZE = 12
MAC_ADDR = 13
ACPI = 14


def get_args():
    parser = argparse.ArgumentParser(description='Train a Bitmap2Config multiclassifier')
    parser.add_argument('--pX', '-x', type=str, default="", help='path to bitmap data')
    parser.add_argument('--pY', '-y', type=str, default="", help='path to evasion data')

    return parser.parse_args()


def get_dataset(X_path, y_path):
    """
    Gets dataset from .npy files
    """
    X = np.load(X_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    X = np.asarray(X).astype('float32')
    y = np.asarray(y).astype('float32')
    print('Dataset loaded with dims: X=', X.shape, 'y=', y.shape)
    return X, y


def build_model(n_inputs=15, n_outputs=13):
    """
    Build an MLP with sigmoid output for probabilistic multilabel classification
    """
    model = Sequential()
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def accuracy(estimator, X, y):
    y_pred = estimator.predict_proba(X)
    y_pred = y_pred.round()
    return accuracy_score(y, y_pred)

def recall(estimator, X, y):
    y_pred = estimator.predict_proba(X)
    y_pred = y_pred.round()
    return recall_score(y, y_pred, average='micro')

def train(X, y):
    """
    Train a Keras MLP for Bitmap2Config multilabel classification
    """
    n_inputs, n_outputs = X.shape[1], y.shape[1]

    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
    # enumerate folds
    accs = []
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = build_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=100)
        # make a prediction on the test set
        yhat = model.predict(X_test)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
        acc = accuracy_score(y_test, yhat)
        # store result
        print('>%.3f' % acc)
        accs.append(acc)

        print('Accuracy: %.3f (%.3f)' % (np.mean(accs), np.std(accs)))

        # plot results
        plt.figure()
        plt.plot(accs)
        plt.xlabel('iteration')
        plt.ylabel('accuracy')
        plt.title('training curve')
        plt.savefig('results/Bitmap2Config0.png')

def train_and_plot(X, y):
    """
    Train a Keras MLP for Bitmap2Config multilabel classification + plot learning curves
    """
    n_inputs, n_outputs = X.shape[1], y.shape[1]

    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    title = "Learning Curves (MLP Multilabel Classifier)"
    estimator = KerasClassifier(build_fn=build_model, n_inputs=n_inputs, n_outputs=n_outputs, epochs=100, batch_size=10, verbose=0)
    plt = plot_learning_curve(
        estimator, title, X, y, recall, cv=cv, n_jobs=4
    )

    # plt.show()
    plt.savefig('results/Bitmap2Config_LC.png')


if __name__ == '__main__':
    args = get_args()

    # get bitmap dataset
    X, y = get_dataset(args.pX, args.pY)

    # train on bitmap
    train_and_plot(X, y)




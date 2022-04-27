import copy
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    multilabel_confusion_matrix, hamming_loss


def degree_overfit(train_metrics, val_metrics):
    """
    Degree of overfitting based on mean difference between training and validation metrics
    """
    diffs = 0
    for metric, values in train_metrics.items():
        t_mean = np.mean(values)
        v_mean = np.mean(val_metrics[metric])
        diffs += abs(t_mean - v_mean)

    return diffs / len(train_metrics)


def ranking_score(y_true, y_pred, predt):
    """
    Computes the ranking score i.e. how many choice along multilabel rank until a correct hit

    Parameters:
        y_true = one-hot multilabel ground truth (N, m)
        y_pred = probabilities of belonging to each class (N, m)
    Returns:
        mean_score = mean number of choices along ranking until a correct label hit
    """
    rankings = np.flip(np.argsort(y_pred, axis=1))  # sort descending
    N, m = y_true.shape
    scores = np.ones(N)*(m+1)
    for i, (ranking, target) in enumerate(zip(rankings, y_true)):
        score = 1
        for rank in ranking:
            if target[rank] and predt[rank]:  # hit
                scores[i] = score
                break
            elif target[rank] and not predt[rank]:  # miss
                score += 1

    mean_score = np.mean(scores)
    return mean_score


def log_metrics(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch):
    d_overfit = degree_overfit(train_metrics, val_metrics)
    experiment.log_metric('Degree overfit', d_overfit, step=epoch)

    with experiment.train():
        experiment.log_metric('Loss', train_loss, step=epoch)
        for metric, values in train_metrics.items():
            experiment.log_metric(metric, np.mean(values), step=epoch)
    with experiment.validate():
        experiment.log_metric('Loss', val_loss, step=epoch)
        for metric, values in val_metrics.items():
            experiment.log_metric(metric, np.mean(values), step=epoch)


def log_reports(experiment, train_metrics, val_metrics, train_loss, val_loss, epoch):
    with experiment.train():
        experiment.log_metric('Loss', train_loss, step=epoch)
    with experiment.validate():
        experiment.log_metric('Loss', val_loss, step=epoch)

    labels = train_metrics.keys()
    scores = train_metrics[0].keys()
    for label in labels:
        for score in scores:
            with experiment.train():
                experiment.log_metric(label + '/' + score, np.mean(train_metrics[label][score]), step=epoch)
            with experiment.validate():
                experiment.log_metric(label + '/' + score, np.mean(val_metrics[label][score]), step=epoch)


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
    predt = np.array(pred > threshold, dtype=float)
    return {'micro/precision': [precision_score(y_true=target, y_pred=predt, average='micro', zero_division=1)],
            'micro/recall': [recall_score(y_true=target, y_pred=predt, average='micro', zero_division=1)],
            'micro/f1': [f1_score(y_true=target, y_pred=predt, average='micro', zero_division=1)],
            'macro/precision': [precision_score(y_true=target, y_pred=predt, average='macro', zero_division=1)],
            'macro/recall': [recall_score(y_true=target, y_pred=predt, average='macro', zero_division=1)],
            'macro/f1': [f1_score(y_true=target, y_pred=predt, average='macro', zero_division=1)],
            'samples/precision': [precision_score(y_true=target, y_pred=predt, average='samples', zero_division=1)],
            'samples/recall': [recall_score(y_true=target, y_pred=predt, average='samples', zero_division=1)],
            'samples/f1': [f1_score(y_true=target, y_pred=predt, average='samples', zero_division=1)],
            'accuracy': [accuracy_score(y_true=target, y_pred=predt)],
            'hamming': [hamming_loss(y_true=target, y_pred=predt)],
            'ranking': [ranking_score(y_true=target, y_pred=pred.numpy(), predt=predt)],
            }

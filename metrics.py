import numpy as np


def mse(y, y_pred):
    """Returns the mean squared error between
    ground truths and predictions.
    """
    return np.mean((y - y_pred) ** 2)


def rmse(y, y_pred):
    """Returns the root mean squared error between
    ground truths and predictions.
    """
    return np.sqrt(np.mean((y - y_pred) ** 2))


def precision(y, y_pred, threshold):
    # precision
    # Calculate TP,FP,TN,FN at every threshold level (0.0 - 5.0)
    # y = test_data[:, 2]
    precision = 0
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(y.shape[0]):
        true_r = y[i]
        est = y_pred[i]
        if true_r >= threshold:
            if est >= threshold:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if est >= threshold:
                fp = fp + 1
            else:
                tn = tn + 1
        if tp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
    return precision


def recall(y, y_pred, threshold):
    # y = test_data[:, 2]
    recall = 0
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(y.shape[0]):
        true_r = y[i]
        est = y_pred[i]
        if true_r >= threshold:
            if est >= threshold:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if est >= threshold:
                fp = fp + 1
            else:
                tn = tn + 1
        if tp == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
    return recall


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

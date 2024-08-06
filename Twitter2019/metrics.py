import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_accuracy(y_true, y_pred_labels):
    y_true = y_true.cpu().numpy()
    y_pred_labels = y_pred_labels.cpu().numpy()
    accuracy = accuracy_score(y_true, y_pred_labels)
    return accuracy


def compute_f1_score(y_true, y_pred_labels):
    y_true = y_true.cpu().numpy()
    y_pred_labels = y_pred_labels.cpu().numpy()
    f1 = f1_score(y_true, y_pred_labels, average='binary')
    return f1


def compute_precision(y_true, y_pred_labels):
    y_true = y_true.cpu().numpy()
    y_pred_labels = y_pred_labels.cpu().numpy()
    precision = precision_score(y_true, y_pred_labels, average='binary', zero_division=1)
    return precision


def compute_recall(y_true, y_pred_labels):
    y_true = y_true.cpu().numpy()
    y_pred_labels = y_pred_labels.cpu().numpy()
    recall = recall_score(y_true, y_pred_labels, average='binary')
    return recall


def binary_class_accuracy(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    if (TP + FN) > 0:
        pos_class_acc = TP / (TP + FN)
    else:
        pos_class_acc = 0.0

    if (TN + FP) > 0:
        neg_class_acc = TN / (TN + FP)
    else:
        neg_class_acc = 0.0

    return pos_class_acc, neg_class_acc

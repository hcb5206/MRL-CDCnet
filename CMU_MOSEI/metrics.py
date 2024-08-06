import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr


def compute_accuracy(y_true, y_pred_prob):
    accuracy = accuracy_score(y_true, y_pred_prob)
    return accuracy


def compute_f1_score(y_true, y_pred_prob, average):
    f1 = 0.0
    if average == "weighted":
        f1 = f1_score(y_true, y_pred_prob, average="weighted")
    if average == "binary":
        f1 = f1_score(y_true, y_pred_prob, average="binary")
    return f1


def compute_MAE(y_true, y_pred_prob):
    y_true = y_true.squeeze().cpu().numpy()
    y_pred_prob = y_pred_prob.squeeze().cpu().numpy()
    mae = mean_absolute_error(y_true, y_pred_prob)
    return mae


def compute_pearsonr(y_true, y_pred_prob):
    y_true = y_true.squeeze().cpu().numpy()
    y_pred_prob = y_pred_prob.squeeze().cpu().numpy()
    cc = pearsonr(y_true, y_pred_prob)[0]
    p_value = pearsonr(y_true, y_pred_prob)[1]
    return cc, p_value


def compute_class_accuracy(y_true, y_pred_prob):
    _, y_true = y_true.max(dim=1)
    _, y_pred_labels = y_pred_prob.max(dim=1)
    class_accuracy = {}
    for label in torch.unique(y_true):
        correct_pred = (y_true == label) & (y_pred_labels == label)
        total_samples = (y_true == label).sum().item()
        class_accuracy[label.item()] = correct_pred.sum().item() / total_samples if total_samples > 0 else 0.0
    return class_accuracy

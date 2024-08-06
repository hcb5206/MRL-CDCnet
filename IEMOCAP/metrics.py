import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_accuracy(y_true, y_pred_prob):
    y_true = np.argmax(y_true.cpu(), axis=1)
    y_pred_labels = np.argmax(y_pred_prob.cpu(), axis=1)
    accuracy = accuracy_score(y_true, y_pred_labels)
    return accuracy


def compute_f1_score(y_true, y_pred_prob):
    y_true = np.argmax(y_true.cpu(), axis=1)
    y_pred_labels = np.argmax(y_pred_prob.cpu(), axis=1)
    f1 = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    return f1


def compute_class_accuracy(y_true, y_pred_prob):
    _, y_true = y_true.max(dim=1)
    _, y_pred_labels = y_pred_prob.max(dim=1)
    class_accuracy = {}
    for label in torch.unique(y_true):
        correct_pred = (y_true == label) & (y_pred_labels == label)
        total_samples = (y_true == label).sum().item()
        class_accuracy[label.item()] = correct_pred.sum().item() / total_samples if total_samples > 0 else 0.0
    return class_accuracy

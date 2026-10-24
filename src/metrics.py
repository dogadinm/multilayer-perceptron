"""Binary classification metrics; malignant (M = 1) is the positive class."""
import numpy as np
from src.network import cross_entropy_from_logits


def classification_metrics(y, probabilities):
    predicted = probabilities.argmax(axis=1)
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return dict(accuracy=float(np.mean(predicted == y)), precision=precision,
                recall=recall, f1=f1, tn=tn, fp=fp, fn=fn, tp=tp)


def evaluate_model(model, x, y):
    _, logits, probabilities = model.forward(x)
    return dict(loss=cross_entropy_from_logits(y, logits),
                **classification_metrics(y, probabilities))


def print_classification_metrics(metrics):
    print(f"Accuracy: {metrics['accuracy']:.4f} - Precision: {metrics['precision']:.4f}"
          f" - Recall: {metrics['recall']:.4f} - F1: {metrics['f1']:.4f}")
    print('Confusion matrix (rows: actual B/M, columns: predicted B/M):')
    print(f"[[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]")

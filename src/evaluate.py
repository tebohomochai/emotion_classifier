"""
Evaluation utilities: 
classification report, confusion matrix, ROC curves
and ROC-data export for downstream reporting.
"""

import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize


def print_classification_report(name: str, y_true, y_pred, classes):
    print(f"\n{'='*50}")
    print(f" {name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=classes))


def plot_confusion_matrix(name: str, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {name}")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(name: str, y_true, y_prob, classes):
    """Plot one-vs-rest ROC curves for each class."""
    y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves for {name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def save_roc_data(y_true, y_prob, classes, output_path: str, n_points: int = 50):
    """Save downsampled ROC curve points to JSON (e.g. for LaTeX plotting)."""
    y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))

    roc_data = {}
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        idx = np.linspace(0, len(fpr) - 1, n_points, dtype=int)
        roc_data[cls] = {
            "fpr": fpr[idx].tolist(),
            "tpr": tpr[idx].tolist(),
        }

    with open(output_path, "w") as f:
        json.dump(roc_data, f)
    print(f"Saved ROC data to {output_path}")


def evaluate(name: str, y_true, y_pred, y_prob, classes, roc_output_path: str = None):
    """Run the full evaluation suite: report, confusion matrix, ROC curves, optional ROC export."""
    print_classification_report(name, y_true, y_pred, classes)
    plot_confusion_matrix(name, y_true, y_pred, classes)
    plot_roc_curves(name, y_true, y_prob, classes)

    if roc_output_path:
        save_roc_data(y_true, y_prob, classes, roc_output_path)

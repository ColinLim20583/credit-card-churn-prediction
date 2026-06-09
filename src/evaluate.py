"""
evaluate.py
Evaluation utilities: confusion matrix, ROC curve, feature importance plots.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)


def load_model(path: str):
    """Load a saved model from disk."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def plot_confusion_matrix(y_test, y_pred, save_path: str = "../images/confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Existing", "Churned"],
        yticklabels=["Existing", "Churned"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — XGBoost", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def plot_roc_curve(model, X_test, y_test, save_path: str = "../images/roc_curve.png"):
    """Plot and save ROC curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — XGBoost", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def plot_feature_importance(model, feature_names, top_n: int = 15, save_path: str = "../images/feature_importance.png"):
    """Plot top N feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color="#2563EB",
        edgecolor="white",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — XGBoost", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def full_evaluation(model_path: str, X_test, y_test, feature_names):
    """Run all evaluation outputs."""
    model = load_model(model_path)
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Existing", "Churned"]))

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(model, feature_names)

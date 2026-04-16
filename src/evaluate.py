"""
MODEL EVALUATION METRICS
accuracy
precision
recall
roc

this runs to get the pkl files of the model

"""

import os
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def load_test_data():
    X_test = load_npz(os.path.join(DATA_DIR, "X_test.npz"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_test, y_test


def get_predictions(X_test):
    # Neural Network
    nn_model = joblib.load(os.path.join(MODELS_DIR, "nn_model.pkl"))
    nn_probs = nn_model.predict_proba(X_test)[:, 1]

    # XGBoost
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # combined average of models
    combined_probs = (nn_probs + xgb_probs) / 2

    return nn_probs, xgb_probs, combined_probs


def compute_metrics(y_true, y_probs, name):
    y_pred = (y_probs >= 0.5).astype(int)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_probs),
    }
    return metrics, y_pred


def plot_confusion_matrix(y_true, y_pred, title, filepath):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Safe", "Phishing"], yticklabels=["Safe", "Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_test, y_test = load_test_data()
    nn_probs, xgb_probs, combined_probs = get_predictions(X_test)
    results = []
    for name, probs in [("Neural Network", nn_probs), ("XGBoost", xgb_probs), ("Combined", combined_probs)]:
        metrics, y_pred = compute_metrics(y_test, probs, name)
        results.append(metrics)

        plot_confusion_matrix(
            y_test, y_pred, f"{name} Confusion Matrix",
            os.path.join(RESULTS_DIR, f"cm_{name.lower().replace(' ', '_')}.png"))

        print()
        print(f"{name}")
        print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    print()
    print("MODEL COMPARISON\n")
    print(f"{'Metric':<12} {'Neural Net':>12} {'XGBoost':>12} {'Combined':>12}")
    for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        vals = [r[metric] for r in results]
        print(f"{metric:<12} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

"""

model 1: neural network using mlpclassifier
gets the tfidf features from datapipeline, then saves the model as pkl file


"""

import os
import numpy as np
from scipy.sparse import load_npz
from sklearn.neural_network import MLPClassifier
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def main():
    # Load pre-computed features
    X_train = load_npz(os.path.join(DATA_DIR, "X_train.npz"))
    X_val = load_npz(os.path.join(DATA_DIR, "X_val.npz"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    print(f"trainig: {X_train.shape}, val: {X_val.shape}")


    # current (UPDATE IF CHANGED) 256 -> relu -> 64 -> relu -> sigmoid
    model = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        batch_size=256,
        max_iter=20,
        early_stopping=True,
        n_iter_no_change=3,
        validation_fraction=0.1,
        random_state=10,
        verbose=True,
    )

    model.fit(X_train, y_train)

    # validation
    val_acc = model.score(X_val, y_val)
    print(f"\nValidation Accuracy: {val_acc:.4f}")

    # save model to avoid rerunning every time
    model_path = os.path.join(MODELS_DIR, "nn_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

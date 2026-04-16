"""

XGBoost model basically same 

"""

import os
import numpy as np
import joblib
from scipy.sparse import load_npz
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def main():
    X_train = load_npz(os.path.join(DATA_DIR, "X_train.npz"))
    X_val = load_npz(os.path.join(DATA_DIR, "X_val.npz"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # XGboost model
    model = XGBClassifier(
        n_estimators=200, # num of trees
        max_depth=5, # depth of tree (higher = more overfitting risk)
        learning_rate=0.1, # can modify; 
        subsample=0.8, #drop 20% rows
        colsample_bytree=0.8, # dropout on features
        eval_metric="logloss",
        random_state=10,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    # val eval
    val_acc = model.score(X_val, y_val)
    print(f"\nValidation Accuracy: {val_acc:.4f}")

    # save model
    joblib.dump(model, os.path.join(MODELS_DIR, "xgb_model.pkl"))


if __name__ == "__main__":
    main()

"""
loads saved models and vectorizer, predicts on new email text.

"""

import os
import numpy as np
import joblib
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_models():
    """Load all saved model artifacts. Call once, cache the result."""
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    feature_names = tfidf.get_feature_names_out().tolist()
    nn_model = joblib.load(os.path.join(MODELS_DIR, "nn_model.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    xgb_explainer = shap.TreeExplainer(xgb_model)

    return {
        "tfidf": tfidf,
        "feature_names": feature_names,
        "nn_model": nn_model,
        "xgb_model": xgb_model,
        "xgb_explainer": xgb_explainer,
    }


def predict_email(text, models):
    """
    this predicts whether an email is phishing or safe
    displays the phishing/safe, and information for each model, and shap values from xgboost model for explainability
    """
    tfidf = models["tfidf"]
    nn_model = models["nn_model"]
    xgb_model = models["xgb_model"]
    explainer = models["xgb_explainer"]
    feature_names = models["feature_names"]

    X = tfidf.transform([text])

    nn_prob = float(nn_model.predict_proba(X)[0, 1])
    xgb_prob = float(xgb_model.predict_proba(X)[:, 1][0])
    avg_prob = (nn_prob + xgb_prob) / 2

    """
    tested with ranges from 0.5 to 0.9 and found good performance between 0.6 and 0.7
    lower threshold more accurate but results in more false positives, and too high can miss actual phishing
    the penalty for missing an actual phishing email outweighs mislabeling a safe email
    """
    label = "High Risk" if avg_prob >= 0.65 else "Safe"

    # SHAP values are for explainability, model determines which words contribute to the target
    shap_values = explainer.shap_values(X)
    shap_arr = np.array(shap_values).flatten()

    # display the 5 influential words by magn; return the indices to be used to find feature names
    top_indices = np.argsort(np.abs(shap_arr))[-5:][::-1]
    reasons = []
    for idx in top_indices:
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        contribution = float(shap_arr[idx])
        reasons.append((name, contribution))

    return {
        "label": label,
        "confidence": avg_prob if label == "High Risk" else 1 - avg_prob,
        "nn_prob": nn_prob,
        "xgb_prob": xgb_prob,
        "reasons": reasons,
    }

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import load_models, predict_email

st.set_page_config(page_title="Phishing Email Detector", layout="centered")


def get_models():
    return load_models()


st.title("Phishing Email Detector")
st.markdown("Paste an email below to check if it's safe or a phishing attempt")

email_text = st.text_area(
    "Email text",
    height=300,
    placeholder="Enter email here",
    label_visibility="collapsed",
)

if st.button("Analyze Email", type="primary"):
    if not email_text.strip():
        st.warning("Please paste an email to analyze phishing risk")
    else:
        with st.spinner("Analyzing"):
            models = get_models()
            result = predict_email(email_text, models)

        if result["label"] == "High Risk":
            st.error(f"{result['label']} — {result['confidence']:.0%} confidence")
        else:
            st.success(f"{result['label']} — {result['confidence']:.0%} confidence")

        st.subheader("SHAP values explainability for classification output")
        for feature_name, contribution in result["reasons"]:
            direction = "increases risk" if contribution > 0 else "decreases risk"
            st.markdown(f"- {feature_name}: {direction} ({contribution:+.3f})")

        st.subheader("Both models agree")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Neural Network", f"{result['nn_prob']:.1%} phishing")
        with mc2:
            st.metric("XGBoost", f"{result['xgb_prob']:.1%} phishing")

        if abs(result["nn_prob"] - result["xgb_prob"]) > 0.3:
            st.warning("Divergence in model predictions")

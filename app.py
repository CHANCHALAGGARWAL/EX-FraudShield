import streamlit as st
from pathlib import Path
from pages import transaction_input, shap_explain, bias_monitor, consent_panel, alerts, audit_log, common_utils

st.set_page_config(page_title="E-X FraudShield", layout="wide")

ROOT = Path(__file__).resolve().parent


def ensure_model():
    model, feat = common_utils.load_model()
    if model is None:
        with st.spinner("Training local model (this happens once)..."):
            model, feat = common_utils.train_and_save_model()
    return model, feat


def main():
    st.sidebar.title("E-X FraudShield")
    st.sidebar.markdown("Lightweight local prototype — ethical & explainable fraud detection")
    page = st.sidebar.radio("Navigation", [
        "Transaction Input & Prediction",
        "SHAP Explainability",
        "Bias Monitoring",
        "User Consent & Data Control",
        "Real-Time Alerts",
        "Audit / Decision Log",
    ])

    model, feature_columns = ensure_model()
    training_df = common_utils.load_transactions()

    st.header("E-X FraudShield")
    st.write("A minimal, local prototype demonstrating explainable fraud detection with privacy controls.")

    if page == "Transaction Input & Prediction":
        transaction_input.render(model, feature_columns, training_df)
    elif page == "SHAP Explainability":
        shap_explain.render(model, feature_columns, training_df)
    elif page == "Bias Monitoring":
        bias_monitor.render(model, feature_columns, training_df)
    elif page == "User Consent & Data Control":
        consent_panel.render()
    elif page == "Real-Time Alerts":
        alerts.render()
    elif page == "Audit / Decision Log":
        audit_log.render()


if __name__ == "__main__":
    main()
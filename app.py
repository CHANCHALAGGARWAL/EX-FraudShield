import streamlit as st
from pathlib import Path

st.set_page_config(page_title="E-X FraudShield", layout="wide")

ROOT = Path(__file__).resolve().parent

from pages import transaction, shap_explain, bias_monitor, consent, alerts, audit, common


def main():
    st.sidebar.title("E-X FraudShield")
    page = st.sidebar.radio("Navigation", [
        "Transaction Input & Prediction",
        "SHAP Explainability",
        "Bias Monitoring",
        "User Consent & Data Control",
        "Real-Time Alerts",
        "Audit / Decision Log",
    ])

    # Train/load model on startup (small sample dataset)
    model, feature_columns, training_df = common.train_model()

    if page == "Transaction Input & Prediction":
        transaction.render(model, feature_columns, training_df)
    elif page == "SHAP Explainability":
        shap_explain.render(model, feature_columns, training_df)
    elif page == "Bias Monitoring":
        bias_monitor.render(model, feature_columns, training_df)
    elif page == "User Consent & Data Control":
        consent.render()
    elif page == "Real-Time Alerts":
        alerts.render()
    elif page == "Audit / Decision Log":
        audit.render()

if __name__ == "__main__":
    main()
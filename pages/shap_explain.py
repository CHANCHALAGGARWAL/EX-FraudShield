import streamlit as st
from pages import common
import matplotlib.pyplot as plt


def render(model, feature_columns, training_df):
    st.header("SHAP-based Explainability")

    st.write("This page shows a simple SHAP-like explanation for a sample transaction or last input.")

    # Provide a sample or allow uploading a small JSON
    sample = {
        "amount": 120.0,
        "location": training_df["location"].iloc[0] if "location" in training_df.columns else "locA",
        "merchant": training_df["merchant"].iloc[0] if "merchant" in training_df.columns else "m1",
        "device_type": training_df["device_type"].iloc[0] if "device_type" in training_df.columns else "mobile",
        "hour": 2,
        "past_txn_count": 0,
        "avg_spend": 45.0,
        "gender": training_df["gender"].iloc[0] if "gender" in training_df.columns else "female",
    }

    st.subheader("Sample transaction")
    st.json(sample)

    if st.button("Explain sample transaction"):
        pred, prob, reason, shap_list, X_in = common.predict_and_explain(model, feature_columns, sample)

        st.write(f"Prediction: {'FRAUD' if pred==1 else 'SAFE'} — Probability: {prob:.2f}")
        st.write(reason)

        feats = [f for f, v in shap_list]
        vals = [v for f, v in shap_list]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(feats[::-1], vals[::-1], color=["#d9534f" if v<0 else "#5cb85c" for v in vals[::-1]])
        ax.set_title("Proxy SHAP feature influences")
        ax.set_xlabel("Influence (proxy)")
        st.pyplot(fig)
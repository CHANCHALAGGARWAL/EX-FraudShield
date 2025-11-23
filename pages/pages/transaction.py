import streamlit as st
from datetime import datetime
from pages import common


def render(model, feature_columns, training_df):
    st.header("Transaction Input & Prediction")

    if "consent_use_personal" not in st.session_state:
        st.session_state.consent_use_personal = True
    if "consent_save_audit" not in st.session_state:
        st.session_state.consent_save_audit = True

    with st.form("txn_form"):
        st.subheader("Enter transaction details")
        amount = st.number_input("Amount (USD)", min_value=0.0, value=50.0, step=1.0)
        location = st.selectbox("Location", options=sorted(training_df["location"].unique())) if "location" in training_df.columns else st.text_input("Location")
        merchant = st.selectbox("Merchant", options=sorted(training_df["merchant"].unique())) if "merchant" in training_df.columns else st.text_input("Merchant")
        device_type = st.selectbox("Device Type", options=sorted(training_df["device_type"].unique())) if "device_type" in training_df.columns else st.text_input("Device Type")
        hour = st.slider("Hour of day", 0, 23, 13)
        past_txn_count = st.number_input("Past transactions (30 days)", min_value=0, value=2)
        avg_spend = st.number_input("Average spend (USD)", min_value=0.0, value=75.0, step=1.0)
        gender = st.selectbox("Gender", options=sorted(training_df["gender"].unique())) if "gender" in training_df.columns else st.selectbox("Gender", options=["male","female","other"]) 
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_dict = {
            "amount": amount,
            "location": location,
            "merchant": merchant,
            "device_type": device_type,
            "hour": hour,
            "past_txn_count": past_txn_count,
            "avg_spend": avg_spend,
            "gender": gender,
        }

        pred, prob, reason, shap_list, X_in = common.predict_and_explain(model, feature_columns, input_dict)

        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            label = "FRAUD" if pred == 1 else "SAFE"
            color = "red" if pred == 1 else "green"
            st.markdown(f"**Prediction:** <span style='color:{color}; font-size:20px'>{label}</span>", unsafe_allow_html=True)
            st.write(f"Probability (fraud): {prob:.2f}")
            st.write("**Natural language explanation:**")
            expl = f"The model estimates a {prob:.0%} chance this transaction is fraud. {reason}"
            st.info(expl)

        with col2:
            st.write("**Feature influences (proxy SHAP)**")
            # draw lightweight bar
            import matplotlib.pyplot as plt
            feats = [f for f, v in shap_list]
            vals = [v for f, v in shap_list]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(feats[::-1], vals[::-1], color=["#d9534f" if v<0 else "#5cb85c" for v in vals[::-1]])
            ax.set_xlabel("Influence (proxy)")
            st.pyplot(fig)

        # Audit log
        if st.session_state.get("consent_save_audit", True):
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "transaction_id": "local-" + datetime.utcnow().strftime("%s"),
                "result": "FRAUD" if pred == 1 else "SAFE",
                "probability": f"{prob:.4f}",
                "reason": reason,
                "features_saved": "Y" if st.session_state.get("consent_use_personal", True) else "N",
            }
            try:
                common.append_audit(entry)
            except Exception as e:
                st.warning(f"Could not write audit log: {e}")
        else:
            st.info("Audit logging disabled by consent settings.")
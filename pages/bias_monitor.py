import streamlit as st
import pandas as pd
from pages import common


def render(model, feature_columns, training_df):
    st.header("Bias Monitoring")

    st.write("Compute fairness statistics over a demographic column and show simple bias checks.")

    if "gender" not in training_df.columns and "region" not in training_df.columns:
        st.info("No demographic column found in sample data (expected 'gender' or 'region').")
        return

    demo_col = st.selectbox("Demographic column", options=[c for c in ["gender", "region"] if c in training_df.columns])

    # Predict on training data to get predicted labels
    X = training_df.drop(columns=["id", "is_fraud"]) if "id" in training_df.columns else training_df.drop(columns=["is_fraud"]) 
    X_proc = pd.get_dummies(X, columns=[c for c in ["location", "merchant", "device_type", "gender"] if c in X.columns])
    # ensure columns
    for c in feature_columns:
        if c not in X_proc.columns:
            X_proc[c] = 0
    X_proc = X_proc[feature_columns]

    preds = model.predict(X_proc)
    dfp = training_df.copy()
    dfp["pred"] = preds

    st.write("Group-level prediction rates:")
    group_rates = dfp.groupby(demo_col)["pred"].mean().rename("pred_rate").reset_index()
    st.dataframe(group_rates)

    # Simple statistical parity difference
    rates = group_rates["pred_rate"].values
    if len(rates) >= 2:
        spd = rates.max() - rates.min()
    else:
        spd = 0.0

    st.write(f"Statistical parity difference (max - min): {spd:.3f}")

    threshold = st.slider("Bias detection threshold", 0.0, 0.5, 0.1)
    if abs(spd) > threshold:
        st.error("Possible bias detected: group prediction rates differ beyond threshold.")
    else:
        st.success("No strong bias detected under selected threshold.")

    st.write("Group counts and basic confusion stats:")
    counts = dfp.groupby(demo_col)["pred"].agg(["count", "sum"]).reset_index()
    counts.columns = [demo_col, "count", "pred_sum"]
    st.dataframe(counts)
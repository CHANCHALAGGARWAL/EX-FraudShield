import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
AUDIT_CSV = DATA_DIR / "audit_log.csv"


def _load_transactions():
    df = pd.read_csv(TRANSACTIONS_CSV)
    return df


def train_model():
    df = _load_transactions()
    # Basic preprocessing: drop id, target
    X = df.drop(columns=["id", "is_fraud"]) if "id" in df.columns else df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    # One-hot encode categorical columns
    X_proc = pd.get_dummies(X, columns=[c for c in ["location", "merchant", "device_type", "gender"] if c in X.columns])

    feature_columns = list(X_proc.columns)

    # small RF model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_proc, y)

    return model, feature_columns, df


def preprocess_input(input_dict, feature_columns):
    # create DF
    df = pd.DataFrame([input_dict])
    df_proc = pd.get_dummies(df, columns=[c for c in ["location", "merchant", "device_type", "gender"] if c in df.columns])
    # align columns
    for c in feature_columns:
        if c not in df_proc.columns:
            df_proc[c] = 0
    df_proc = df_proc[feature_columns]
    return df_proc


def predict_and_explain(model, feature_columns, input_dict):
    X_in = preprocess_input(input_dict, feature_columns)
    prob = model.predict_proba(X_in)[0][1]
    pred = int(prob >= 0.5)

    # Simple human-friendly reason: use top feature importances + feature values
    importances = dict(zip(feature_columns, model.feature_importances_))
    # pick top 3 most important features present in input
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top = []
    for f, _ in sorted_feats:
        # check if this feature is relevant in X_in (>0)
        if X_in.iloc[0].get(f, 0) != 0:
            top.append(f)
        if len(top) >= 3:
            break

    reason = "High risk features: " + ", ".join(top) if top else "Model indicates risk from transaction pattern."

    # Prepare simple shap-like values using permutation of importances * sign of feature value
    # (This is a lightweight proxy to avoid heavy runtime in small prototype.)
    shap_vals = {f: (X_in.iloc[0].get(f, 0) * importances.get(f, 0)) for f in feature_columns}
    # convert to list of (feature, value) sorted
    shap_list = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    return pred, prob, reason, shap_list, X_in


def append_audit(entry: dict):
    # Ensure audit file exists with header
    import csv
    header = ["timestamp", "transaction_id", "result", "probability", "reason", "features_saved"]
    file_path = AUDIT_CSV
    write_header = not file_path.exists()
    with open(file_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(entry)
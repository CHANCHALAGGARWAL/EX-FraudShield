import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
MODEL_FILE = MODEL_DIR / "model.pkl"
AUDIT_CSV = DATA_DIR / "audit_log.csv"


def load_transactions():
    return pd.read_csv(TRANSACTIONS_CSV)


def train_and_save_model(random_state: int = 42):
    df = load_transactions()
    X = df.drop(columns=["id", "is_fraud"]) if "id" in df.columns else df.drop(columns=["is_fraud"]) 
    y = df["is_fraud"]

    cat_cols = [c for c in ["location", "merchant", "device_type", "gender"] if c in X.columns]
    X_proc = pd.get_dummies(X, columns=cat_cols)

    feature_columns = list(X_proc.columns)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_proc, y)

    payload = {"model": model, "feature_columns": feature_columns}
    joblib.dump(payload, MODEL_FILE)
    return model, feature_columns


def load_model():
    if MODEL_FILE.exists():
        payload = joblib.load(MODEL_FILE)
        return payload.get("model"), payload.get("feature_columns")
    return None, None


def predict_and_explain_loaded(model, feature_columns, input_dict):
    import pandas as pd
    df = pd.DataFrame([input_dict])
    cat_cols = [c for c in ["location", "merchant", "device_type", "gender"] if c in df.columns]
    df_proc = pd.get_dummies(df, columns=cat_cols)
    for c in feature_columns:
        if c not in df_proc.columns:
            df_proc[c] = 0
    df_proc = df_proc[feature_columns]

    prob = model.predict_proba(df_proc)[0][1]
    pred = int(prob >= 0.5)

    importances = dict(zip(feature_columns, model.feature_importances_))
    shap_vals = {f: (df_proc.iloc[0].get(f, 0) * importances.get(f, 0)) for f in feature_columns}
    shap_list = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    top_feats = [f for f, v in shap_list if df_proc.iloc[0].get(f, 0) != 0][:3]
    reason = "High risk features: " + ", ".join(top_feats) if top_feats else "Model indicates risk from transaction pattern."

    return pred, prob, reason, shap_list, df_proc


def append_audit(entry: dict):
    import csv
    header = ["timestamp", "transaction_id", "result", "probability", "reason", "features_saved"]
    file_path = AUDIT_CSV
    write_header = not file_path.exists()
    with open(file_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(entry)

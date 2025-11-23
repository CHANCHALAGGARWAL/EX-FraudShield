"""Smoke test: load model and run a sample prediction."""
import joblib
import pandas as pd
from pathlib import Path


def run():
    root = Path(__file__).resolve().parents[1]
    model_file = root / 'model' / 'model.pkl'
    assert model_file.exists(), 'model/model.pkl must exist'
    payload = joblib.load(model_file)
    model = payload.get('model')
    feats = payload.get('feature_columns')
    assert hasattr(model, 'predict'), 'Loaded model must implement predict'

    df = pd.read_csv(root / 'data' / 'transactions.csv')
    sample = df.drop(columns=['id','is_fraud']).iloc[0].to_dict()

    X = pd.DataFrame([sample])
    X_proc = pd.get_dummies(X, columns=[c for c in ['location','merchant','device_type','gender'] if c in X.columns])
    for c in feats:
        if c not in X_proc.columns:
            X_proc[c] = 0
    X_proc = X_proc[feats]

    prob = model.predict_proba(X_proc)[0][1]
    pred = int(prob >= 0.5)
    print('SMOKE_OK', pred, prob)


if __name__ == '__main__':
    run()

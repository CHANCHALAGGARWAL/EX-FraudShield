"""Simple helper to train and save the model used by the Streamlit app.

Usage: python train_model.py
This will call `pages.common_utils.train_and_save_model()` and save a joblib model
at `model/model.pkl`.
"""

from pages import common_utils


def main():
    print("Training model (this may take a moment)...")
    model, feature_columns = common_utils.train_and_save_model()
    print(f"Saved model with {len(feature_columns)} feature columns")


if __name__ == '__main__':
    main()

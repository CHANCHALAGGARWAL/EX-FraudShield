import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALERTS_CSV = ROOT / "data" / "alerts.csv"


def render():
    st.header("Real-Time Alerts")
    st.write("Alerts are read from a local CSV to simulate real-time alerting.")

    try:
        df = pd.read_csv(ALERTS_CSV)
        st.dataframe(df.sort_values(by=["timestamp"], ascending=False))
    except Exception as e:
        st.error(f"Could not load alerts: {e}")
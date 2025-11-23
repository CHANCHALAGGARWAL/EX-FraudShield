import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_CSV = ROOT / "data" / "audit_log.csv"


def render():
    st.header("Audit / Decision Log")
    st.write("Shows recorded decisions and basic filters.")

    try:
        df = pd.read_csv(AUDIT_CSV)
        st.dataframe(df.sort_values(by=["timestamp"], ascending=False))
    except FileNotFoundError:
        st.info("No audit records yet. Predictions will create audit entries if consented.")
    except Exception as e:
        st.error(f"Could not load audit log: {e}")
import streamlit as st


def render():
    st.header("User Consent & Data Control")

    if "consent_use_personal" not in st.session_state:
        st.session_state.consent_use_personal = True
    if "consent_save_audit" not in st.session_state:
        st.session_state.consent_save_audit = True

    st.write("Control how local personal data and logs are used/stored in this prototype.")

    use_personal = st.checkbox("Allow saving personal features with predictions (location, device, merchant)", value=st.session_state.consent_use_personal)
    save_audit = st.checkbox("Allow saving audit logs of predictions (privacy-preserving)", value=st.session_state.consent_save_audit)

    st.session_state.consent_use_personal = use_personal
    st.session_state.consent_save_audit = save_audit

    st.markdown("---")
    st.write("Notes:")
    st.write("- This prototype stores audit logs locally in `data/audit_log.csv`. Toggle controls grant explicit consent for saving personal features and logs.")
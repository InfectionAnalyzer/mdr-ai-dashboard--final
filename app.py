import streamlit as st
import pandas as pd
from predictor import predict_objective
import joblib
import io

st.set_page_config(page_title="MDR AI System", layout="wide")

st.title("📊 National MDR Prediction & Antibiotic Misuse Dashboard")

# -- Sidebar: Navigation
st.sidebar.header("📋 Select Objective")
objective_map = {
    "☑ Phase 1: MDR Prediction": "mdr_model_obj1_retrained.pkl",
    "💊 Phase 1: Dose Error Classification": "dose_error_model_obj2_retrained.pkl",
    "🦠 Phase 1: Resistance Mechanism": "resistance_model_obj3.pkl",
    "🧠 Phase 1: Explainable & Fair AI": "fairness_model_obj4.pkl",
    "⏱ Phase 2: Infection Onset Forecast": "infection_onset_model_obj5.pkl",
    "📈 Phase 2: Response Forecasting": "response_model_obj6.pkl",
    "📉 Phase 2: Resistance Progression": "progression_model_obj7.pkl",
    "📋 Phase 3: Policy Simulation": "policy_model_obj8.pkl",
    "💸 Phase 3: Economic Impact": "economic_model_obj9.pkl",
    "🧬 Phase 3: Cluster Detection": "cluster_model_obj10.pkl",
    "⚖️ Phase 3: Fairness Audit": "fairness_audit_model_obj11.pkl",
    "📱 Phase 3: EMR Feedback": "emr_feedback_model_obj12.pkl"
}

selected_obj = st.sidebar.radio("Choose module", list(objective_map.keys()))
model_file = objective_map[selected_obj]

st.markdown(f"## {selected_obj}")
st.markdown("----")

# -- Columns: Upload (left), Output (right)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload CSV")
    with st.expander("ℹ️ File must match model features"):
        st.info("Upload a .csv file with patient data. Rows = cases, Columns = features.")
    uploaded_file = st.file_uploader("Upload patient data (.csv)", type="csv")

with col2:
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.success(f"✅ {len(data)} rows uploaded.")
        st.markdown("### 👁 Preview")
        st.dataframe(data.head())

        # Predict
        preds = predict_objective(data, model_file)
        st.markdown("### 📊 Prediction Output")
        st.dataframe(preds)

        # Allow CSV download
        buffer = io.StringIO()
        preds.to_csv(buffer, index=False)
        st.download_button("📥 Download Results", buffer.getvalue(), file_name="predictions.csv", mime="text/csv")
    else:
        st.warning("📁 Upload a file to start prediction.")
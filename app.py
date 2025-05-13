import streamlit as st
import pandas as pd
from predictor import predict_objective
import joblib

st.set_page_config(page_title="MDR AI System", layout="wide")

st.title("📊 National MDR Prediction & Antibiotic Misuse Audit System")

tabs = st.tabs([
    "Objective 1: MDR Prediction",
    "Objective 2: Dose Error Classification",
    "Objective 3: Resistance Mechanism",
    "Objective 4: Explainable & Fair AI",
    "Objective 5: Infection Onset Forecast",
    "Objective 6: Response Forecasting",
    "Objective 7: Resistance Progression",
    "Objective 8: Policy Simulation",
    "Objective 9: Economic Impact",
    "Objective 10: Cluster Detection",
    "Objective 11: Fairness Audit",
    "Objective 12: EMR Feedback"
])

model_files = [
    "mdr_model_obj1_retrained.pkl",
    "dose_error_model_obj2_retrained.pkl",
    "resistance_model_obj3.pkl",
    "fairness_model_obj4.pkl",
    "infection_onset_model_obj5.pkl",
    "response_model_obj6.pkl",
    "progression_model_obj7.pkl",
    "policy_model_obj8.pkl",
    "economic_model_obj9.pkl",
    "cluster_model_obj10.pkl",
    "fairness_audit_model_obj11.pkl",
    "emr_feedback_model_obj12.pkl"
]

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(tab._label)
        uploaded_file = st.file_uploader("Upload patient data (.csv) for this objective", type="csv", key=f"upload_{i}")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("📋 Input Preview:", data.head())
            preds = predict_objective(data, model_files[i])
            st.success("✅ Prediction Complete")
            st.dataframe(preds)preds = predict_objective(data, model_files[i])
st.dataframe(preds)  # ✅ Correct

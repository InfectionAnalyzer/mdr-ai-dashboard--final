import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="MDR App Diagnostic Mode", layout="wide")
st.title("🛠️ Diagnostic: File Upload Test (MDR AI System)")

st.sidebar.header("📋 Debug Panel")
selected_option = st.sidebar.radio("Choose Test Objective", ["MDR Prediction"])

st.markdown("### 📤 Upload Your CSV File")
uploaded_file = st.file_uploader("Upload a test CSV file (Max 200MB)", type="csv")

if uploaded_file:
    st.success("✅ File upload detected.")
    st.write("🔍 Uploaded file info:", uploaded_file.name, uploaded_file.size, "bytes")

    try:
        # Attempt to read CSV
        data = pd.read_csv(uploaded_file)
        st.markdown("### 👁 Input Preview (First 5 rows)")
        st.dataframe(data.head())
        st.success("✅ CSV loaded successfully.")

        # Show column names and types
        st.markdown("### 🧾 Column Diagnostics")
        st.write(data.dtypes)

    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
else:
    st.warning("📁 Please upload a file to test CSV reading.")
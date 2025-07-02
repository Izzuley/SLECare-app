# slecare_app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import json
import os

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="SLECare - Ramalan CKD & Remisi", layout="wide")
st.title("🧬 SLECare - Ramalan CKD & Remisi")
st.markdown("---")

# === RADIO TOGGLE: MODEL SELECTION ===
target_option = st.radio("Pilih Model Ramalan:", ["CKD", "Remission"], horizontal=True)

# === LOAD MODEL & FEATURE SET ===
if target_option == "CKD":
    model_path = "catboost_ckd_model.pkl"
    feature_path = "ckd_rfe_features.json"
else:
    model_path = "catboost_remission_model.pkl"
    feature_path = "remission_rfe_features.json"

model = joblib.load(model_path)
with open(feature_path, "r") as f:
    selected_features = json.load(f)

# === DEFINE INPUT RULES ===
binary_dropdown_features = {
    "MSK", "MUCOCUTANEOS", "NPSLE", "GIT", "FIRST OR RELAPSE LN",
    "ANY CR 6 MTH", "AC KIDNEY INJURY INITIAL (AKI)", "CR 12 MTH PRED 7.5",
    "GLOBAL SCLEROSIS_MISSING", "CKD", "CR 6 MTH PRED 10", "CR 6 MTH PRED 7.5", "RB_diffuse",
    "ACE/ARB"
}  # Removed "LA" from here
selectbox_features = {
    "GENDER", "RACE", "APL POSITIVE", "LA"
}  # Added "LA" here

# === LAYOUT COLUMNS ===
left, right = st.columns(2)

# === USER INPUTS ===
user_input = {}
st.subheader("📝 Masukkan Data Pesakit")

for i, feat in enumerate(selected_features):
    col = left if i % 2 == 0 else right
    with col:
        if feat in binary_dropdown_features:
            val = st.selectbox(f"{feat}", options=[0, 1], key=f"{feat}_binary")
            user_input[feat] = int(val)
        elif feat in selectbox_features:
            if feat == "GENDER":
                val = st.selectbox(f"{feat}", options=[0, 1], format_func=lambda x: ["Perempuan", "Lelaki"][x], key=f"{feat}_gender")
            elif feat == "RACE":
                val = st.selectbox(f"{feat}", options=[0, 1, 2], format_func=lambda x: ["Malay", "Chinese", "Indian/Other"][x], key=f"{feat}_race")
            elif feat == "APL POSITIVE":
                val = st.selectbox(f"{feat}", options=[0, 1, 2], key=f"{feat}_apl")
            elif feat == "LA":
                val = st.selectbox(f"{feat}", options=[0, 1, 2], format_func=lambda x: ["Negatif", "Positif", "Tidak Diuji"][x], key=f"{feat}_la")
            user_input[feat] = int(val)
        else:
            val = st.number_input(f"{feat}", step=0.01, format="%0.2f", key=f"{feat}_num")
            user_input[feat] = val

predict_btn = st.button("🔍 Ramal")

# === PREDICTION LOGIC ===
if predict_btn:
    input_df = pd.DataFrame([user_input])
    input_df = input_df[selected_features]  # Ensure correct column order
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    if (target_option == "CKD" and prediction == 1) or (target_option == "Remission" and prediction == 0):
        st.error("❗ Ramalan: Risiko tinggi untuk keadaan ini.")
    else:
        st.info("🔵 Ramalan: Risiko rendah atau tiada keadaan ini.")

    # === SHAP EXPLAINER ===
    st.subheader("📊 Penjelasan Ciri (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # === SHAP WATERFALL PLOT ===
    st.markdown("#### 🔹 Kesan setiap ciri terhadap ramalan (Waterfall Plot)")
    fig_waterfall = plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[0], input_df.iloc[0]
    )
    st.pyplot(fig_waterfall)

    # === SHAP BAR PLOT ===
    st.markdown("#### 🔹 Ciri paling mempengaruhi ramalan")
    fig_bar = plt.figure(figsize=(6, 4))  # Width x Height in inches
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig_bar)

    # === OPTIONAL EXPORT ===
    st.download_button(
        label="📥 Muat Turun Keputusan (CSV)",
        data=input_df.assign(Prediction=prediction).to_csv(index=False).encode("utf-8"),
        file_name=f"slecare_prediction_{target_option.lower()}.csv",
        mime="text/csv"
    )
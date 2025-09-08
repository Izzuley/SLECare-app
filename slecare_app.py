import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import json
from catboost import Pool
from pathlib import Path

st.set_page_config(page_title="SLECare - Ramalan Baseline", layout="wide")
st.title("🧬 SLECare - Ramalan Risiko Baseline")
st.markdown("---")

pilihan_model = st.radio("Pilih Model Ramalan:", ["CKD", "Remission"], horizontal=True, key="pilihan_model_radio")

if pilihan_model == "CKD":
    model_path = "catboost_ckd_model.pkl"
    feature_path = "ckd_rfe_features.json"
    global_barplot_img = "ckd_barplot.png"
    master_cat_features = [
        "GENDER", "MSK", "MUCOCUTANEOS", "NPSLE", "SEROSITIS",
        "ACE/ARB", "AC KIDNEY INJURY INITIAL (AKI)", "GLOBAL SCLEROSIS_MISSING",
        "CRESCENT_MISSING", "RACE", "APL POSITIVE", "LA",
        "ANTIDSDNA PRE TX", "ANTIB2 GP1 IGG", "ACL IGM"
    ]
else:
    model_path = "catboost_remission_model.pkl"
    feature_path = "remission_rfe_features.json"
    global_barplot_img = "remission_barplot.png"
    master_cat_features = [
        "PULM", "GIT", "FIRST OR RELAPSE LN", "INDUCTION CYC", "ACE/ARB",
        "AC KIDNEY INJURY INITIAL (AKI)", "CKD", "CRESCENT_MISSING",
        "GLOBAL SCLEROSIS_MISSING", "RACE", "LA", "ACL IGM", "ANTIB2GP1 IGM",
        "ANTIDSDNA PRE TX"
    ]

model = joblib.load(model_path)
with open(feature_path, "r") as f:
    selected_features = json.load(f)

binary_dropdown_features = set(master_cat_features)
selectbox_features = {"GENDER", "RACE", "FIRST OR RELAPSE LN", "ANTIDSDNA PRE TX", "ANTIB2 GP1 IGG", "ACL IGM", "ANTIB2GP1 IGM", "LA"}

left, right = st.columns(2)
user_input = {}
st.subheader("📝 Masukkan Data Pesakit Baseline")

for i, feat in enumerate(selected_features):
    col = left if i % 2 == 0 else right
    with col:
        if feat in binary_dropdown_features:
            if feat in selectbox_features:
                if feat == "GENDER":
                    val = st.selectbox(f"{feat}", [0, 1], format_func=lambda x: ["Lelaki", "Perempuan"][x], key=f"{feat}_{i}_gender")
                elif feat == "RACE":
                    val = st.selectbox(f"{feat}", [0, 1, 2], format_func=lambda x: ["Melayu", "Cina", "India"][x], key=f"{feat}_{i}_race")
                elif feat == "ANTIDSDNA PRE TX":
                    val = st.selectbox(f"{feat}", [0, 1, 2], format_func=lambda x: ["Negatif", "Positif", "Tidak Diuji"][x], key=f"{feat}_{i}_anti")
                elif feat == "ANTIB2 GP1 IGG":
                    val = st.selectbox(f"{feat}", [0, 1, 2], format_func=lambda x: ["Negatif", "Positif", "Tidak Diuji"][x], key=f"{feat}_{i}_igg")
                elif feat == "LA":
                    val = st.selectbox(f"{feat}", [0, 1, 2], format_func=lambda x: ["Negatif", "Positif", "Tidak Diuji"][x], key=f"{feat}_{i}_la")
                elif feat == "ACL IGM":
                    val = st.selectbox(f"{feat}", [0, 1, 2], format_func=lambda x: ["Negatif", "Positif", "Tidak Diuji"][x], key=f"{feat}_{i}_acl_igm")
                elif feat == "ANTIB2GP1 IGM":
                    val = st.selectbox(f"{feat}", [0, 1, 2], format_func=lambda x: ["Negatif", "Positif", "Tidak Diuji"][x], key=f"{feat}_{i}_antib2gp1_igm")
                elif feat == "FIRST OR RELAPSE LN":
                    val = st.selectbox(f"{feat}", [1, 2], format_func=str, key=f"{feat}_{i}_firstrelapse", help="1 = Pertama kali, 2 = Relapse")
                else:
                    val = st.selectbox(f"{feat}", [0, 1], key=f"{feat}_{i}_binarybox")
                user_input[feat] = int(val)
            else:
                hint = None
                if feat == "GLOBAL SCLEROSIS_MISSING":
                    hint = "1 = Nilai GLOBAL SCLEROSIS tiada, median 0 akan digunakan"
                elif feat == "CRESCENT_MISSING":
                    hint = "1 = Nilai CRESCENT tiada, median 0 akan digunakan"
                val = st.selectbox(f"{feat}", [0, 1], key=f"{feat}_{i}_binary", help=hint)
                user_input[feat] = int(val)
        else:
            val = st.number_input(f"{feat}", step=0.01, format="%0.2f", key=f"{feat}_{i}_num")
            user_input[feat] = val

if user_input.get("GLOBAL SCLEROSIS_MISSING") == 1:
    user_input["GLOBAL SCLEROSIS"] = 0.0
if user_input.get("CRESCENT_MISSING") == 1:
    user_input["CRESCENT"] = 0.0

predict_btn = st.button("🔍 Ramal")

if predict_btn:
    input_df = pd.DataFrame([user_input])
    input_df = input_df[selected_features]
    cat_feats_in_use = [f for f in master_cat_features if f in selected_features]
    for cat in ["ANTIDSDNA PRE TX", "ANTIB2 GP1 IGG", "LA", "ACL IGM", "ANTIB2GP1 IGM"]:
        if cat in cat_feats_in_use:
            input_df[cat] = input_df[cat].astype(str)
    pool = Pool(data=input_df, cat_features=cat_feats_in_use)
    prediction = model.predict(pool)[0]
    st.markdown("---")
    if (pilihan_model == "CKD" and prediction == 1) or (pilihan_model == "Remission" and prediction == 0):
        st.error("❗ Ramalan: Risiko Tinggi.")
    else:
        st.success("✅ Ramalan: Risiko Rendah atau Tiada.")

    st.subheader("📊 Penjelasan SHAP")
    explainer = shap.TreeExplainer(model)
    shap_values_local = explainer.shap_values(input_df)
    shap_input_df = input_df.copy()
    for cat in ["ANTIDSDNA PRE TX", "ANTIB2 GP1 IGG", "LA", "ACL IGM", "ANTIB2GP1 IGM"]:
        if cat in shap_input_df.columns:
            shap_input_df[cat] = shap_input_df[cat].astype(int)

    st.markdown("#### 🔹 Sumbangan Faktor Individu")
    fig_waterfall = plt.figure()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values_local[0], shap_input_df.iloc[0])
    st.pyplot(fig_waterfall)

    st.markdown("#### 🔹 Sumbangan Faktor Keseluruhan")
    st.image(global_barplot_img, caption="Faktor Keseluruhan (Global)")

    # ==== Dependence Plots (Precomputed PNGs) ====
    DEP_DIR = Path("Dependence Plots")
    st.markdown("#### 🔹 Plot Kebergantungan (Precomputed)")
    if pilihan_model == "CKD":
        dep_files = ["CKD_CREAT.png", "CKD_LA.png", "CKD_RACE.png"]
    else:
        dep_files = ["Remission_CKD.png", "Remission_GLOBAL.png", "Remission_MONTH.png", "Remission_RACE.png"]

    cols = st.columns(2)
    for idx, fname in enumerate(dep_files):
        p = DEP_DIR / fname
        with cols[idx % 2]:
            if p.exists():
                st.image(str(p), caption=f"{fname.replace('_',' ').replace('.png','')}")
            else:
                st.info(f"Plot tidak ditemui: {p}")

    st.download_button(
        "📥 Muat Turun Ramalan",
        input_df.assign(Ramalan=prediction).to_csv(index=False).encode("utf-8"),
        file_name=f"slecare_ramalan_{pilihan_model.lower()}.csv",
        mime="text/csv"
    )

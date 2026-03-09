# SLECare

**SLECare** is a machine learning-based clinical decision support application developed to predict two important outcomes in Malaysian **Lupus Nephritis (LN)** patients:

- **Chronic Kidney Disease (CKD) risk**
- **Delayed remission risk**

The system was developed using clinical data from LN patients treated at **Hospital Canselor Tuanku Muhriz (HCTM), UKM** between **2000 and 2020**, and is designed to support more informed and proactive clinical assessment. The final deployed interface uses the **Baseline CatBoost model** and provides **prediction output, local SHAP explanation, global SHAP visualization, and CSV export** functionality.

---

## Project Background

Systemic Lupus Erythematosus (SLE) is a heterogeneous autoimmune disease, and **Lupus Nephritis (LN)** is one of its most serious manifestations because it can lead to **chronic kidney disease (CKD)** and poor long-term renal outcomes. This project, titled **SLECare**, was built to classify LN patient risk using machine learning while keeping the workflow clinically interpretable and reproducible.

The broader study benchmarked multiple machine learning models, but the final application deploys only the **Baseline CatBoost model**, as it achieved the most balanced overall performance for clinical use. The modelling workflow emphasised careful preprocessing, leakage control, feature selection, and explainability through SHAP.

---

## Objectives

This application was built to:

1. Predict the risk of **CKD** in Malaysian LN patients.
2. Predict the risk of **delayed remission**.
3. Provide **interpretable model output** using SHAP.
4. Support clinicians with a lightweight, interactive web-based prediction tool.

---

## Main Features

- **Two prediction modes**
  - CKD
  - Remission
- **Baseline-only deployment**
  - Uses pre-treatment / baseline patient features only
- **CatBoost-based prediction**
  - Chosen as the final deployed model
- **SHAP interpretability**
  - Local waterfall plot for patient-specific explanation
  - Global SHAP bar plot image for overall feature importance
- **Interactive Streamlit interface**
  - User-friendly input form for clinicians or evaluators
- **CSV export**
  - Download prediction results directly from the app

---

## Model Scope

Although the full project benchmarked three modelling scenarios:

- **Baseline** = pre-treatment features only
- **6MTH** = baseline + 3-month features
- **12MTH** = baseline + 3-month + 6-month features

only the **Baseline scenario** is deployed in this application, as it is the most practical for real clinical use and does not depend on future follow-up measurements.

---

## Final Deployed Model

The application uses:

- **CatBoost** for classification
- **JSON feature lists** to ensure consistent model inputs
- **Precomputed SHAP global bar plot images**
- **Joblib-saved trained models**

Two trained models are expected:

- `catboost_ckd_model.pkl`
- `catboost_remission_model.pkl`

and their corresponding selected feature files:

- `ckd_rfe_features.json`
- `remission_rfe_features.json`

---

## App Workflow

1. User selects a prediction target:
   - **CKD**
   - **Remission**
2. User enters baseline patient information.
3. The app loads the correct trained CatBoost model and selected feature list.
4. The app generates a prediction.
5. The result is displayed as either:
   - **High Risk**
   - **Low / No Risk**
6. SHAP-based explanations are shown:
   - **Local explanation** via waterfall plot
   - **Global explanation** via bar plot image
7. The user may download the output as a **CSV file**.

---

## Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **CatBoost**
- **SHAP**
- **Matplotlib**
- **Joblib**
- **JSON**

---

## Repository Contents

A minimal working repository should contain files similar to the following:

```text
.
├── slecare_app.py
├── catboost_ckd_model.pkl
├── catboost_remission_model.pkl
├── ckd_rfe_features.json
├── remission_rfe_features.json
├── ckd_barplot.png
├── remission_barplot.png
├── requirements.txt
└── README.md
```

If your main Streamlit file has a different name, update the run command accordingly.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not yet have a `requirements.txt`, you can install the core dependencies manually:

```bash
pip install streamlit pandas joblib shap matplotlib catboost
```

---

## Running the App

Use Streamlit to launch the application:

```bash
streamlit run slecare_app.py
```

Then open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

---

## Required Input Files

The app depends on the following external model and asset files being present in the same directory as the Streamlit app unless paths are changed in code:

### Model files
- `catboost_ckd_model.pkl`
- `catboost_remission_model.pkl`

### Feature files
- `ckd_rfe_features.json`
- `remission_rfe_features.json`

### SHAP image files
- `ckd_barplot.png`
- `remission_barplot.png`

If these files are missing, the application will not run correctly.

---

## Input Behaviour

The app dynamically builds the input form based on the selected feature list for each model.

It supports:

- **Numeric inputs** via number fields
- **Categorical inputs** via dropdown selection
- **Binary indicators** for selected clinical features
- **Missingness flags** such as:
  - `GLOBAL SCLEROSIS_MISSING`
  - `CRESCENT_MISSING`

Where a missingness flag is set to 1, the app fills the associated original value with a default numeric value before prediction.

---

## Output

For each prediction, the app provides:

- **Predicted risk category**
- **Patient-specific SHAP waterfall plot**
- **Global SHAP feature importance image**
- **Downloadable CSV result**

---

## Clinical Interpretation Note

This application is intended as a **decision support tool**, not as a standalone diagnostic or treatment system. Predictions should always be interpreted alongside:

- clinical judgement,
- patient history,
- laboratory findings,
- and specialist assessment.

---

## Research and Methodology Summary

The wider SLECare project followed a structured machine learning workflow that included:

- removal of irrelevant and leakage-prone features,
- manual missing value handling,
- outlier checks,
- stratified train-test split,
- feature selection using RFE,
- benchmark comparison across CatBoost, Random Forest, MLP, and SVM,
- and SHAP-based interpretability.

While multiple benchmark models were explored, the final deployed interface uses only the **Baseline CatBoost model** for practicality, interpretability, and balanced performance.

---

## Limitations

- The deployed app uses **baseline features only**.
- The system was developed from a **single-centre Malaysian LN cohort**.
- Predictions are constrained by the quality and representativeness of the historical dataset.
- The repository does not itself include retraining code unless explicitly added.

---

## Suggested Future Improvements

- Add model confidence probabilities to the interface
- Add clearer feature descriptions for non-technical users
- Add input validation and error handling for missing artifact files
- Add Docker support for easier deployment
- Add deployment instructions for Streamlit Community Cloud or other hosting platforms
- Add audit logging for clinical testing environments

---

## Acknowledgements

This project was developed as part of a Final Year Project at:

- **Universiti Kebangsaan Malaysia (UKM)**
- **Faculty of Information Science and Technology (FTSM)**

with clinical data and domain support related to Malaysian Lupus Nephritis cases from **Hospital Canselor Tuanku Muhriz (HCTM)**.

---

## Citation

If you use or adapt this project, please cite the corresponding Final Year Project report and acknowledge the original clinical and academic context of SLECare.

---

## License

Add your preferred license here, for example:

```text
MIT License
```

or replace this section with your institution's required licensing terms.

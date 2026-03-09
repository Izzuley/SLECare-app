# SLECare: Baseline Risk Prediction App

SLECare is a Streamlit-based clinical decision support prototype developed as part of a Final Year Project (FYP) in machine learning for systemic lupus erythematosus (SLE).

This application allows a user to enter a patient's baseline clinical data and generate a machine learning prediction for one of two supported outcomes:

- **CKD risk prediction**
- **Remission outcome prediction**

In addition to the prediction result, the app provides **SHAP-based explainability** so users can see which factors contributed most to the model's decision at both the individual and overall level.

---

## Project Overview

SLE is a complex autoimmune disease with highly variable clinical presentations and outcomes. Because of this complexity, conventional analysis may miss important non-linear relationships between patient variables and disease outcomes.

This project applies a **CatBoost-based machine learning workflow** to structured clinical data and turns the trained models into an interactive application for easier demonstration and interpretation.

The app is designed to:

- support **baseline risk estimation** for selected SLE-related outcomes,
- improve understanding of **important predictive features**,
- provide **local and global explainability** through SHAP,
- and demonstrate how machine learning can be translated into a simple clinical-facing tool.

---

## Main Features

### 1. Dual prediction mode
Users can choose between two prediction modules:

- **CKD**
- **Remission**

Each module loads its own:

- trained CatBoost model,
- selected feature list,
- global SHAP importance plot,
- and dependence plots.

### 2. Interactive patient input form
The interface dynamically generates input fields based on the selected model's required features.

It supports:

- numeric clinical inputs,
- binary categorical inputs,
- multi-class categorical inputs such as gender, race, and selected immunological markers,
- and missing-indicator logic for selected biopsy-related variables.

### 3. Prediction result display
After the user enters patient data and clicks the prediction button, the app returns a risk classification:

- **High risk**, or
- **Low / no risk**

### 4. SHAP explainability
To make the prediction interpretable, the app includes:

- **Local explanation** using a SHAP waterfall plot for the current patient
- **Global explanation** using a saved SHAP feature importance bar plot
- **Dependence plots** for selected important variables

### 5. CSV export
Users can download the entered patient data together with the generated prediction as a CSV file.

---

## Tech Stack

- **Python**
- **Streamlit** for the web interface
- **CatBoost** for machine learning prediction
- **SHAP** for model explainability
- **Pandas** for data handling
- **Matplotlib** for plot rendering
- **Joblib / JSON** for loading saved models and selected feature files

---

## Repository Contents

A typical project structure for this app is expected to look like this:

```bash
.
├── slecare_app.py
├── catboost_ckd_model.pkl
├── catboost_remission_model.pkl
├── ckd_rfe_features.json
├── remission_rfe_features.json
├── ckd_barplot.png
├── remission_barplot.png
├── Dependence Plots/
│   ├── CKD_CREAT.png
│   ├── CKD_LA.png
│   ├── CKD_RACE.png
│   ├── Remission_CKD.png
│   ├── Remission_GLOBAL.png
│   ├── Remission_MONTH.png
│   └── Remission_RACE.png
├── requirements.txt
└── README.md
```

> Important: the app depends on these model, JSON, and image assets being present in the correct paths.

---

## How the App Works

### CKD module
When the user selects **CKD**, the app loads:

- `catboost_ckd_model.pkl`
- `ckd_rfe_features.json`
- `ckd_barplot.png`

It then displays the feature input form based on the selected CKD features and generates a CKD risk prediction.

### Remission module
When the user selects **Remission**, the app loads:

- `catboost_remission_model.pkl`
- `remission_rfe_features.json`
- `remission_barplot.png`

It then displays the feature input form based on the selected remission features and generates the corresponding outcome prediction.

### Explainability workflow
After a prediction is made, the app:

1. builds the input record into a dataframe,
2. prepares categorical variables for CatBoost,
3. runs the model prediction,
4. computes SHAP values using `shap.TreeExplainer`,
5. displays a **waterfall plot** for the individual patient,
6. shows a **global feature importance plot**,
7. and shows additional **dependence plots** if the image files exist.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create and activate a virtual environment (recommended)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not yet have a `requirements.txt`, the main packages needed are:

```txt
streamlit
pandas
joblib
shap
matplotlib
catboost
```

---

## Running the App

Start the Streamlit app with:

```bash
streamlit run slecare_app.py
```

Then open the local URL shown in the terminal, usually:

```bash
http://localhost:8501
```

---

## Using the App

1. Launch the app.
2. Choose the prediction mode: **CKD** or **Remission**.
3. Enter the required patient baseline data.
4. Click **Ramal**.
5. Review:
   - the prediction result,
   - the SHAP waterfall plot,
   - the global importance plot,
   - and the dependence plots.
6. Download the result as CSV if needed.

---

## Input Handling Notes

The app includes several interface rules to improve usability:

- Some variables are entered through **dropdowns** instead of free-text input.
- Some immunological variables use encoded categories such as:
  - `0 = Negative`
  - `1 = Positive`
  - `2 = Not Tested`
- Some biopsy-related missingness flags automatically assign default median values when the value is unavailable.

This means the app is designed to work with the same feature format used during model preparation.

---

## Expected Output

For each prediction, the app provides:

- a risk classification,
- local SHAP explanation for the individual patient,
- global SHAP feature importance,
- optional SHAP dependence plots,
- and a downloadable CSV containing the entered record and prediction.

---

## Limitations

This repository is a prototype / academic FYP application and should be interpreted accordingly.

- It is **not a production clinical system**.
- It depends on the availability of the exact trained model files and supporting assets.
- Predictions are limited to the outcomes implemented in the current version of the app.
- The app is intended for **research, demonstration, and educational purposes**.
- Clinical decisions should never rely on this tool alone without expert medical judgement.

---

## Future Improvements

Potential future enhancements include:

- cleaner feature labels for non-technical users,
- better form grouping by clinical category,
- probability score display in addition to class prediction,
- richer patient-level explanation summaries in plain language,
- model performance dashboard,
- deployment to cloud hosting,
- and support for additional SLE-related prediction tasks.

---

## Academic Context

This app was developed as part of an FYP focused on applying machine learning to SLE-related clinical prediction and interpretability. The project combines:

- structured clinical data,
- CatBoost modeling,
- feature selection,
- and SHAP-based explainable AI

into a simple interactive interface that makes the trained models easier to demonstrate and understand.

---

## Disclaimer

This application is intended for **academic and research use only**. It is not a certified medical device and is not intended to replace clinician judgement, diagnosis, or treatment planning.

---

## Author

**Muhammad Izzul Islam Bin Faisal**  
Final Year Project - SLECare


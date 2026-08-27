# Clinical Outcomes Prediction
A machine‑learning dashboard for predicting hypertension risk with patient‑level SHAP interpretability.

## Overview
This project builds an end‑to‑end clinical risk prediction system using a Random Forest model and SHAP interpretability. The dashboard allows users to:

- Upload or select patient data
- Generate a hypertension risk prediction
- View global feature importance
- Explore patient‑specific SHAP explanations
- Interpret model behavior using waterfall and beeswarm plots

The goal is to provide transparent, interpretable clinical ML that supports decision‑making rather than replacing it.

## Live Dashboard
You can access the deployed Streamlit app here:
[Live Streamlit App](https://mw-clinical-outcomes-prediction.streamlit.app/)

## Model
The model is a **Random Forest Classifier** trained on some basic clinical features:
- Blood pressure
- Age
- Race
- Sex

The model outputs a **probability of hypertension** for each patient.

## SHAP Interpretability
SHAP (SHapely Additive exPlanations) is used to interpret both global and individual predictions.

#### Global Interpretability
A SHAP Beeswarm plot illustrates the distribution of feature impacts across all patients.

#### Individual Interpretability
A SHAP Waterfall plot shows how each feature pushes a single patient's risk up or down. This highlights the strongest risk drivers for the selected patient.

These plots help clinicians understand *why* the model made a prediction.


## Screenshots

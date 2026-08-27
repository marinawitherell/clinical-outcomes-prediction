import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle

# Load the trained model
rf = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
df = pd.read_csv("nhanes_hypertension_clean.csv")

features = ['Age', 'Sex', 'Race', 'Systolic_BP', 'Diastolic_BP']

st.title("NHANES Hypertension Risk Dashboard")

# Create a sidebar for patient selection

st.sidebar.header("Select Patient")
patient_index = st.sidebar.slider("Patient Index", min_value=0, max_value=len(df)-1, value=0)
patient = df.iloc[patient_index]

st.write("### Selected Patient Information")
st.write(patient[features])

# Predict risk of hypertension
X = df[features]
# X_scaled = scaler.transform(X)
risk = rf.predict_proba(X)[:, 1]
risk_indiv = rf.predict_proba(X.iloc[patient_index].values.reshape(1, -1))[0, 1]

st.write("### Hypertension Risk Score")
st.write(f"{risk_indiv:.3f}")

# Risk score distribution plot
st.write("### Risk Score Distribution")

fig, ax = plt.subplots()
ax.hist(risk, bins=30, color='steelblue')
ax.axvline(risk_indiv, color='red', linestyle='--', linewidth=2, label='Selected Patient')
ax.set_title("Risk Score Distribution")
st.pyplot(fig)

# SHAP summary plot
X_df = pd.DataFrame(X, columns=features)

explainer = shap.Explainer(rf, X_df)
shap_values = explainer(X_df)

st.write("### SHAP Feature Importance")
plt.figure()
shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
st.pyplot(plt.gcf())

# st.write("### SHAP Summary (Beeswarm Plot)")
# plt.figure()
# shap.summary_plot(shap_values, X_df, show=False)
# st.pyplot(plt.gcf())


# Individual SHAP explanation for the selected patient
st.write("### Individual Patient Explanation")
# i = patient_index
# base_value = explainer.expected_value[0]
# shap_vals = shap_values[i].values
# features_row = X_df.iloc[i].values

# fig_force = shap.force_plot(
#     base_value,
#     shap_vals,
#     features_row,
#     matplotlib=True
# )

# st.pyplot(fig_force)


i = patient_index

# Convert multi-output SHAP explanation to single-output (class 1)
exp = shap_values[i]
exp_single = shap.Explanation(
    values=exp.values[:, 1],
    base_values=exp.base_values[1],
    data=exp.data,
    feature_names=exp.feature_names
)

st.write("### Individual Patient SHAP Explanation (Waterfall Plot)")

plt.figure()
shap.plots.waterfall(exp_single)
fig_waterfall = plt.gcf()
st.pyplot(fig_waterfall)
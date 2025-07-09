import streamlit as st
import pandas as pd
import pickle
import numpy as np

with open("credit_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

status_map = {
    'A11': '< 0 DM',
    'A12': '0 ≤ < 200 DM',
    'A13': '≥ 200 DM or salary assignment',
    'A14': 'No checking account'
}
credit_history_map = {
    'A30': 'No credits/all paid back',
    'A31': 'All paid back duly (this bank)',
    'A32': 'Credits paid back till now',
    'A33': 'Delay in paying off',
    'A34': 'Critical/other credit existing'
}
purpose_map = {
    'A40': 'Car (new)', 'A41': 'Car (used)', 'A42': 'Furniture/equipment',
    'A43': 'Radio/TV', 'A44': 'Domestic appliances', 'A45': 'Repairs',
    'A46': 'Education', 'A48': 'Retraining', 'A49': 'Business', 'A410': 'Others'
}
savings_map = {
    'A61': '< 100 DM', 'A62': '100 ≤ < 500 DM', 'A63': '500 ≤ < 1000 DM',
    'A64': '≥ 1000 DM', 'A65': 'Unknown/no savings'
}
employment_map = {
    'A71': 'Unemployed', 'A72': '< 1 year', 'A73': '1 ≤ < 4 years',
    'A74': '4 ≤ < 7 years', 'A75': '≥ 7 years'
}
personal_status_map = {
    'A91': 'Male-divorced/separated', 'A92': 'Female-div/sep/married',
    'A93': 'Male-single', 'A94': 'Male-married/widowed', 'A95': 'Female-single'
}
debtors_map = {'A101': 'None', 'A102': 'Co-applicant', 'A103': 'Guarantor'}
property_map = {
    'A121': 'Real estate', 'A122': 'Building society/life ins.',
    'A123': 'Car or other', 'A124': 'No property/unknown'
}
installment_plan_map = {'A141': 'Bank', 'A142': 'Stores', 'A143': 'None'}
housing_map = {'A151': 'Rent', 'A152': 'Own', 'A153': 'For free'}
job_map = {
    'A171': 'Unskilled-nonresident', 'A172': 'Unskilled-resident',
    'A173': 'Skilled/official', 'A174': 'Management/self-employed'
}
telephone_map = {'A191': 'None', 'A192': 'Yes (registered)'}
foreign_worker_map = {'A201': 'Yes', 'A202': 'No'}

reverse_maps = {
    "Status": {v: k for k, v in status_map.items()},
    "CreditHistory": {v: k for k, v in credit_history_map.items()},
    "Purpose": {v: k for k, v in purpose_map.items()},
    "Savings": {v: k for k, v in savings_map.items()},
    "EmploymentSince": {v: k for k, v in employment_map.items()},
    "PersonalStatusSex": {v: k for k, v in personal_status_map.items()},
    "Debtors": {v: k for k, v in debtors_map.items()},
    "Property": {v: k for k, v in property_map.items()},
    "InstallmentPlans": {v: k for k, v in installment_plan_map.items()},
    "Housing": {v: k for k, v in housing_map.items()},
    "Job": {v: k for k, v in job_map.items()},
    "Telephone": {v: k for k, v in telephone_map.items()},
    "ForeignWorker": {v: k for k, v in foreign_worker_map.items()}
}

st.set_page_config(page_title="German Credit Predictor", layout="centered")
st.title("💳 German Credit Risk Classifier")

mode = st.radio("Choose Input Mode:", ["Manual Entry", "Upload CSV"])

if mode == "Manual Entry":
    st.subheader("🔎 Enter details for one customer")
    
    input_dict = {
        "Status": st.selectbox("Status of checking account", list(reverse_maps["Status"].keys())),
        "Duration": st.slider("Duration in months", 4, 72, 24),
        "CreditHistory": st.selectbox("Credit History", list(reverse_maps["CreditHistory"].keys())),
        "Purpose": st.selectbox("Purpose", list(reverse_maps["Purpose"].keys())),
        "CreditAmount": st.slider("Credit Amount", 250, 20000, 3000),
        "Savings": st.selectbox("Savings account", list(reverse_maps["Savings"].keys())),
        "EmploymentSince": st.selectbox("Employment since", list(reverse_maps["EmploymentSince"].keys())),
        "InstallmentRate": st.slider("Installment rate (%)", 1, 4, 2),
        "PersonalStatusSex": st.selectbox("Personal status and sex", list(reverse_maps["PersonalStatusSex"].keys())),
        "Debtors": st.selectbox("Other debtors/guarantors", list(reverse_maps["Debtors"].keys())),
        "ResidenceSince": st.slider("Years at current residence", 1, 4, 2),
        "Property": st.selectbox("Property", list(reverse_maps["Property"].keys())),
        "Age": st.slider("Age", 18, 75, 35),
        "InstallmentPlans": st.selectbox("Other installment plans", list(reverse_maps["InstallmentPlans"].keys())),
        "Housing": st.selectbox("Housing", list(reverse_maps["Housing"].keys())),
        "ExistingCredits": st.slider("Existing credits at bank", 1, 4, 1),
        "Job": st.selectbox("Job", list(reverse_maps["Job"].keys())),
        "LiablePeople": st.slider("People supported", 1, 4, 1),
        "Telephone": st.selectbox("Telephone", list(reverse_maps["Telephone"].keys())),
        "ForeignWorker": st.selectbox("Foreign worker", list(reverse_maps["ForeignWorker"].keys()))
    }

    for col in reverse_maps:
        input_dict[col] = reverse_maps[col][input_dict[col]]

    if st.button("Predict"):
        df_input = pd.DataFrame([input_dict])

        for col in df_input.select_dtypes(include='object').columns:
            le = label_encoders[col]
            df_input[col] = le.transform(df_input[col])

        df_input = df_input[feature_columns]
        scaled = scaler.transform(df_input)
        prediction = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        st.markdown("### 🧾 Prediction Result")
        if prediction == 1:
            st.success(f"✅ Status: **Good**\n\n📈 Probability (Good): **{prob:.2f}**")
        else:
            st.error(f"🚨 Status: **Bad**\n\n📉 Probability (Bad): **{1 - prob:.2f}**")

else:
    st.subheader("📤 Upload CSV for bulk predictions")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            for col in input_df.select_dtypes(include='object').columns:
                if col in label_encoders:
                    input_df[col] = label_encoders[col].transform(input_df[col])

            missing_cols = set(feature_columns) - set(input_df.columns)
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            else:
                input_df = input_df[feature_columns]
                scaled_data = scaler.transform(input_df)
                predictions = model.predict(scaled_data)
                probabilities = model.predict_proba(scaled_data)[:, 1]

                result_df = input_df.copy()
                result_df["Predicted Risk"] = ["Good (0)" if p == 1 else "Bad (1)" for p in predictions]
                result_df["Probability (Good)"] = probabilities.round(3)

                st.success("Predictions completed!")
                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="credit_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# 💳 Credit Scoring App - Ensemble Methods

A Streamlit web app that classifies credit risk (Good or Bad) using an ensemble model trained on the [German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).  
This project leverages feature engineering, label encoding, scaling, and ensemble modeling to provide both single and batch credit risk predictions.

---

## 🚀 Features

- 🔎 **Manual Entry**: Enter individual customer details for instant prediction
- 📤 **CSV Upload**: Upload a CSV for batch credit predictions
- 🧠 **Pretrained Model**: Ensemble (e.g., XGBoost) trained with proper encoding and scaling
- 🎯 **Output**: Predicts `Good` or `Bad` with associated probability
- 📥 Download results as CSV (batch mode)

---
## 📦 Repository Structure
├── app.py # Streamlit application
├── code.ipynb # Contains the training, preprocessing and visualization code & results
├── credit_classifier.pkl # Trained ensemble classifier
├── scaler.pkl # StandardScaler for numerical features
├── features.pkl # List of model input features
├── label_encoders.pkl # Saved label encoders for categorical variables
├── README.md # This file

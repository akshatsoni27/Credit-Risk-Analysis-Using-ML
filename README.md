<div align="center">
  <h1 align="center">💳 Credit Risk Analysis System</h1>
  <p align="center">
    <strong>A complete, production-ready machine learning pipeline and interactive dashboard for evaluating credit applications.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/XGBoost-1D9D58?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
    <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  </p>
</div>

<hr />

## 🌟 Overview

This project encompasses a machine learning pipeline that ingests the Statlog (German Credit Data) dataset, trains an **XGBoost** classifier with hyperparameter tuning, and serves real-time predictions through a modern, responsive **Streamlit** dashboard.

## ⚖️ Business Logic & Cost Matrix

A crucial domain factor for this dataset is the **asymmetrical cost of misclassification**:
- Misclassifying a "Bad" risk applicant as "Good" is **5 times worse** than misclassifying a "Good" applicant as "Bad" (cost of default vs. cost of lost opportunity).

We address this heavily in the model training phase by utilizing XGBoost's `scale_pos_weight = 5.0`. This aggressively penalizes False Negatives (predicting "Good" when actual is "Bad"), explicitly optimizing the model for an extremely high **Recall** on high-risk profiles.

## 🚀 Setup Instructions

### 1. Create a Virtual Environment

It is highly recommended to isolate your dependencies using a Python virtual environment:

```powershell
python -m venv .venv
# Activate the environment
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install the required packages using the `requirements.txt` manifest:

```powershell
pip install -r requirements.txt
```

### 3. Bake the Model

Execute the training script to ingest the dataset, run cross-validation grid search to find the optimal hyperparameters, and export the finalized inference pipeline to `model.joblib`:

```powershell
python train.py
```

### 4. Launch the Web Application

Start the Streamlit UI to interactively assess credit risks using the newly baked model:

```powershell
streamlit run app.py
```

## 📊 Features Evaluated

The model actively evaluates the following application parameters:
- **Demographics:** Age, Sex
- **Financials:** Requested credit amount, Savings account tier, Checking account tier
- **Context:** Purpose of credit, Duration of credit (months), Housing status, Job category

*(Note: Credit history and exact Employment History duration were abstracted by the model as they were not present in the base CSV dataset).*

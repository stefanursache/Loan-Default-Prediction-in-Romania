📌 Project Overview

This repository contains two AI-powered loan default prediction models designed for Romanian businesses. The models use real-time economic data, machine learning, and deep learning AI to assess loan risks accurately.

1️⃣ Loan Default Prediction (Basic Model)

Uses Random Forest & Gradient Boosting classifiers.

Incorporates real Romanian economic indicators (inflation, interest rates, GDP growth, etc.).

Evaluates feature importance and key financial metrics.

File: loan_default_prediction_huggingface.py

2️⃣ Loan Default Prediction (Advanced Model)

Adds Hugging Face TabNet for structured data classification.

Implements new financial ratios (Loan-to-Income Ratio, Debt Service Coverage).

Supports multi-model evaluation (Random Forest, Gradient Boosting, TabNet).

Includes SHAP analysis for model explainability.

File: loan_default_prediction_advanced.py

📂 Dataset & APIs

This project relies on real-world financial & economic data:

Romanian Economic Data API: Inflation, GDP Growth, Interest Rates.

Banking APIs (PSD2) for financial transactions.

ANAF API for business creditworthiness (Planned integration).

Google Trends API for economic sentiment analysis (Planned integration).

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/stefan.ursache/loan-default-romania.git
cd loan-default-romania

Install dependencies:

pip install -r requirements.txt

Run the prediction scripts:

python loan_default_prediction_huggingface.py

python loan_default_prediction_advanced.py

📊 Model Performance

Model

Accuracy

ROC AUC Score

F1 Score

Random Forest

XX%

XX

XX

Gradient Boosting

XX%

XX

XX

Hugging Face TabNet

XX%

XX

XX

📈 Feature Importance

The key factors influencing loan defaults:

Credit Score

Loan Amount

Business Age

Debt-to-Income Ratio

Inflation Rate

🏗️ Future Improvements

🔹 Add PSD2 APIs for real banking data.🔹 Deploy as a FastAPI for real-time scoring.🔹 Improve interpretability using SHAP & LIME.🔹 Build an interactive Streamlit dashboard.

📜 License

This project is open-source under the MIT License.

🔗 Contributors: Stefan and 

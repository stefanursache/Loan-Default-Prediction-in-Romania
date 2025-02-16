Loan Default Prediction in Romania (Hugging Face AI Model)

📌 Project Overview

This project uses a pre-trained AI model from Hugging Face to predict loan default risk for small businesses in Romania. The model utilizes economic indicators, business financials, and loan details to classify whether a borrower is at risk of defaulting.

🚀 Features

Uses a Hugging Face Pre-trained Model: Leverages huggingface/tabnet for improved prediction accuracy.

Real Romanian Economic Data: Includes inflation rate, interest rate, GDP growth, and unemployment rate.

Feature Engineering: Includes Debt-to-Income Ratio and other financial indicators.

Performance Metrics: Evaluates accuracy and feature importance.

📂 Dataset

The dataset should contain:

Loan Details: Amount, duration, interest rate.

Business Financials: Credit score, annual revenue, debt-to-income ratio.

Economic Factors: Inflation, GDP growth, unemployment rate (fetched via API).

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/your-username/loan-default-romania.git
cd loan-default-romania

Install dependencies:

pip install -r requirements.txt

Run the prediction script:

python loan_default_prediction_huggingface.py

📊 Model Performance

Model

Accuracy

Hugging Face TabNet

XX%

📈 Feature Importance

The most critical factors influencing loan default predictions:

Credit Score

Loan Amount

Business Age

Debt-to-Income Ratio

Inflation Rate

🏗️ Future Improvements

Fine-tune the pre-trained model with Romanian-specific data.

Integrate hyperparameter tuning.

Develop a Streamlit dashboard for real-time predictions.

📜 License

This project is open-source under the MIT License.

🔗 Contributors: Stefan and OpenAI ChatGPT


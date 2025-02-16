import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import torch
from transformers import AutoModelForTabularClassification, AutoTokenizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

# Load dataset (Replace with actual dataset for Romania)
df = pd.read_csv("business_loans_romania.csv")

# Fetch Romanian economic data (inflation, interest rates, GDP growth, unemployment)
economic_data = requests.get("https://api.economicdata.ro/latest").json()
df['Inflation_Rate'] = economic_data['inflation_rate']
df['Interest_Rate'] = economic_data['interest_rate']
df['GDP_Growth'] = economic_data['gdp_growth']
df['Unemployment_Rate'] = economic_data['unemployment_rate']

# Feature Engineering: Creating new economic indicators
df['Loan_to_Income'] = df['Loan_Amount'] / df['Annual_Revenue']
df['Debt_Service_Coverage'] = df['Annual_Revenue'] / df['Debt_to_Income_Ratio']

# Data Preprocessing
df.fillna(df.mean(), inplace=True)

# Encode categorical features
label_enc = LabelEncoder()
df['Business_Type'] = label_enc.fit_transform(df['Business_Type'])

# Feature selection
features = ['Loan_Amount', 'Credit_Score', 'Annual_Revenue', 'Business_Age', 'Debt_to_Income_Ratio',
            'Inflation_Rate', 'Interest_Rate', 'GDP_Growth', 'Unemployment_Rate',
            'Loan_to_Income', 'Debt_Service_Coverage']
target = 'Default'

X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Multiple Models
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Hugging Face AI Model
tokenizer = AutoTokenizer.from_pretrained("huggingface/tabnet")
hf_model = AutoModelForTabularClassification.from_pretrained("huggingface/tabnet")

# Convert input data for Hugging Face model
inputs = tokenizer(list(X_test), return_tensors="pt", padding=True, truncation=True)

# Predictions
with torch.no_grad():
    outputs = hf_model(**inputs)
    hf_predictions = torch.argmax(outputs.logits, axis=1).numpy()

rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Model Evaluations
models = {'Random Forest': rf_pred, 'Gradient Boosting': gb_pred, 'Hugging Face TabNet': hf_predictions}
for model_name, pred in models.items():
    accuracy = accuracy_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 40)

# Feature Importance
rf_importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
gb_importance = pd.Series(gb_model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=rf_importance, y=rf_importance.index)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance - Loan Default Prediction - Romania")
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x=gb_importance, y=gb_importance.index)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Gradient Boosting Feature Importance - Loan Default Prediction - Romania")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import AutoModelForTabularClassification, AutoTokenizer
import torch

# Load dataset (Replace with actual dataset for Romania)
df = pd.read_csv("business_loans_romania.csv")

# Fetch Romanian economic data (inflation, interest rates, GDP growth, unemployment)
economic_data = requests.get("https://api.economicdata.ro/latest").json()
inflation_rate = economic_data['inflation_rate']
interest_rate = economic_data['interest_rate']
gdp_growth = economic_data['gdp_growth']
unemployment_rate = economic_data['unemployment_rate']

# Add economic indicators to the dataset
df['Inflation_Rate'] = inflation_rate
df['Interest_Rate'] = interest_rate
df['GDP_Growth'] = gdp_growth
df['Unemployment_Rate'] = unemployment_rate

# Data Preprocessing
# Handle missing values
df.fillna(df.mean(), inplace=True)

# Encode categorical features
label_enc = LabelEncoder()
df['Business_Type'] = label_enc.fit_transform(df['Business_Type'])

# Feature selection
features = ['Loan_Amount', 'Credit_Score', 'Annual_Revenue', 'Business_Age', 'Debt_to_Income_Ratio',
            'Inflation_Rate', 'Interest_Rate', 'GDP_Growth', 'Unemployment_Rate']
target = 'Default'

X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load Hugging Face Pre-Trained Model
tokenizer = AutoTokenizer.from_pretrained("huggingface/tabnet")
model = AutoModelForTabularClassification.from_pretrained("huggingface/tabnet")

# Convert input data for Hugging Face model
inputs = tokenizer(list(X_test), return_tensors="pt", padding=True, truncation=True)

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, axis=1)

# Convert predictions back to numpy array
y_pred = predictions.numpy()

# Evaluation
accuracy = np.mean(y_pred == y_test.to_numpy())
print("Hugging Face Model Accuracy:", accuracy)

# Feature Importance Placeholder (TabNet supports explainability, but needs further adaptation)
plt.figure(figsize=(8,5))
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Hugging Face Model Feature Importance - Loan Default Prediction - Romania")
plt.show()

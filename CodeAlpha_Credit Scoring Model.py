# Credit Scoring Model - Machine Learning Project
# Author: Student Project
# Description: Predicts creditworthiness using ML classification algorithms

# ================================
# 1. Import Required Libraries
# ================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# ================================
# 2. Load Dataset
# ================================
# Replace 'credit_data.csv' with your dataset file
# Sample columns expected:
# income, debt, credit_utilization, payment_history, loan_amount, default

data = pd.read_csv('credit_data.csv')

print("Dataset Shape:", data.shape)
print(data.head())

# ================================
# 3. Data Preprocessing
# ================================
# Handle missing values
data.fillna(data.mean(), inplace=True)

# Separate features and target
X = data.drop('default', axis=1)   # Features
y = data['default']                # Target

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 4. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 5. Train Models
# ================================

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ================================
# 6. Model Evaluation Function
# ================================
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} Performance ---")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-Score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================================
# 7. Evaluate All Models
# ================================
evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
evaluate_model(dt_model, X_test, y_test, "Decision Tree")
evaluate_model(rf_model, X_test, y_test, "Random Forest")

# ================================
# 8. Prediction on New Data
# ================================
# Example new applicant data
new_applicant = np.array([[50000, 15000, 0.3, 0.9, 200000]])
new_applicant_scaled = scaler.transform(new_applicant)

prediction = rf_model.predict(new_applicant_scaled)
probability = rf_model.predict_proba(new_applicant_scaled)

if prediction[0] == 1:
    print("\nCredit Approved (Low Risk)")
else:
    print("\nCredit Rejected (High Risk)")

print("Approval Probability:", probability)

# ================================
# End of Project Code
# ================================

"""
==============================================================================
PCCOER, Pune - Department of Computer Engineering
Subject: Machine Learning Laboratory | Academic Year: 2026-27
Lab-7: Naive Bayes Classification Algorithm
Student Name: Ujjwal Chaudhari | Roll No: 56 | Section: A | Batch: A4
GitHub Profile: https://github.com/Ujjwal-Chaudhari-Git
# GitHub Repository: https://github.com/Ujjwal-Chaudhari-Git/Lab-7-Naive-Bayes
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)

# 1. Ingest Data
df = pd.read_csv("bank_notes.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# 2. Extract Features and Target
X = df[['variance', 'skewness', 'curtosis', 'entropy']]
y = df['Target']

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Train Gaussian Naive Bayes Model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 5. Predictions & Posterior Probabilities
y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)[:, 1]

# 6. Evaluation Metrics
print("\n" + "="*50)
print("GAUSSIAN NAIVE BAYES PERFORMANCE REPORT")
print("="*50)
print(f"Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision : {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Recall    : {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1-Score  : {f1_score(y_test, y_pred)*100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Genuine (0)', 'Forged (1)']))

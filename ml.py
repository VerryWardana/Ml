import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

print("=== Memulai Proses ===")

# Load dataset
df = pd.read_csv('marketing_campaign_cleaned.csv')

# Cek missing value
missing_values = df.isnull().sum()
total_missing = missing_values.sum()
if total_missing == 0:
    print("✅ Tidak ada nilai hilang (NA) dalam dataset.")
else:
    print("⚠️ Ada nilai hilang di kolom berikut:")
    print(missing_values[missing_values > 0])

# Pastikan kolom 'Response' ada
if 'Response' not in df.columns:
    raise ValueError("❌ ERROR: Kolom 'Response' tidak ditemukan di dataset.")

# Konversi ke kategori / label encoding
le_education = LabelEncoder()
le_marital = LabelEncoder()
df['Education'] = le_education.fit_transform(df['Education'])
df['Marital_Status'] = le_marital.fit_transform(df['Marital_Status'])

# Pastikan Response hanya 0/1
if not set(df['Response'].unique()).issubset({0,1}):
    raise ValueError("❌ ERROR: Kolom 'Response' mengandung nilai selain 0 dan 1.")

print("\nDistribusi target Response:")
print(df['Response'].value_counts(normalize=True))

# Pisahkan fitur dan label
X = df.drop('Response', axis=1)
y = df['Response']

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nJumlah Data Train: {len(X_train)}")
print(f"Jumlah Data Test : {len(X_test)}")

print("\nProporsi Response di Train:")
print(pd.Series(y_train).value_counts(normalize=True))

print("\nProporsi Response di Test:")
print(pd.Series(y_test).value_counts(normalize=True))

# Training Model Decision Tree (C5.0 mirip DecisionTreeClassifier di sklearn)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("✅ Training selesai.")

# Prediksi & Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy:.4f}")


print("=== Proses Selesai ===")
from sklearn import tree
plt.figure(figsize=(20,10))
tree.plot_tree(model, feature_names=X.columns, class_names=["0", "1"], filled=True)
plt.show()
import rpy2.robjects as ro


r_code = """
library(C50)
data(iris)
model <- C5.0(Species ~ ., data=iris)
print(summary(model))
"""
ro.r(r_code)




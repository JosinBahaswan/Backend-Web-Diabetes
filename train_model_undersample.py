import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
import json
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# -------------------------------
print("\U0001F504 Memuat data dari file Excel...")
DATA_PATH = 'diabetes_dataset.xlsx'
df = pd.read_excel(DATA_PATH, header=4)
print(f"Jumlah total data awal: {len(df)}")

# -------------------------------
df = df.dropna()
print(f"Jumlah data setelah menghapus nilai kosong: {len(df)}")

# -------------------------------
df = df[df['Diabetes_012'].isin(['no diabetes', 'diabetes'])]
print(f"Jumlah data setelah menyaring hanya 2 kelas (no diabetes & diabetes): {len(df)}")

# -------------------------------
print("\U0001F520 Melakukan label encoding pada kolom kategorikal...")
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])
le_diabetes = LabelEncoder()
df['Diabetes_012'] = le_diabetes.fit_transform(df['Diabetes_012'])

# -------------------------------
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# -------------------------------
cols_to_drop = ['NoDocbcCost', 'AnyHealthcare', 'MentHlth', 'PhysHlth']
X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
print(f"\U0001FA9B Kolom yang dihapus: {cols_to_drop}")

# -------------------------------
print("\U0001F4CF Melakukan normalisasi pada fitur numerik...")
num_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
dump(scaler, 'scaler.joblib')
print("✅ Normalisasi selesai dan scaler disimpan.")

# -------------------------------
print("\U0001F500 Membagi data menjadi data latih dan data uji (80:20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -------------------------------
print('\n\U0001F4CA Distribusi kelas sebelum balancing (data latih):')
print(pd.Series(y_train).value_counts())

# -------------------------------
print("\n\U0001F501 Tahap 1: SMOTE oversampling minoritas 2.5x...")
minority_label = y_train.value_counts().idxmin()
majority_label = y_train.value_counts().idxmax()
minority_count = y_train.value_counts()[minority_label]
desired_minority = int(minority_count * 2.5)
sampling_strategy_smote = {minority_label: desired_minority, majority_label: y_train.value_counts()[majority_label]}
smote = SMOTE(sampling_strategy=sampling_strategy_smote, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print('✅ Distribusi kelas setelah SMOTE:')
print(pd.Series(y_train).value_counts())

# -------------------------------
print("\n\U0001F4C9 Tahap 2: Undersampling kelas mayoritas agar sama...")
minority_count_new = y_train.value_counts()[minority_label]
sampling_strategy_under = {majority_label: minority_count_new, minority_label: minority_count_new}
rus = RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)
print('✅ Distribusi kelas setelah undersampling:')
print(pd.Series(y_train).value_counts())

# -------------------------------
print("\n➕ Tahap 3: Menambahkan 20.000 data ke kelas mayoritas...")
desired_majority_final = minority_count_new + 15000
sampling_strategy_final = {majority_label: desired_majority_final, minority_label: minority_count_new}
ros_final = RandomOverSampler(sampling_strategy=sampling_strategy_final, random_state=42)
X_train, y_train = ros_final.fit_resample(X_train, y_train)
print('✅ Distribusi kelas akhir:')
print(pd.Series(y_train).value_counts())

# -------------------------------
print("\n⚖️ Cost-sensitive learning (kelas 1 dihukum 2x)...")
class_weights = {0: 1, 1: 2}
rf = RandomForestClassifier(random_state=42, class_weight=class_weights)

# -------------------------------
print("\n🔁 Validasi silang (StratifiedKFold)...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"Skor F1-macro tiap fold: {scores}")
print(f"Rata-rata F1-macro: {scores.mean():.4f} (+/- {scores.std():.4f})")

# -------------------------------
print("\n🧪 GridSearchCV tuning...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', None],
    'bootstrap': [True]
}

grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, scoring='f1', verbose=2)
grid_search.fit(X_train, y_train)
print(f"✅ Best Params: {grid_search.best_params_}")

best_clf = grid_search.best_estimator_

# -------------------------------
print("\n\U0001F50D Evaluasi model awal...")
y_proba = best_clf.predict_proba(X_test)[:, 1]
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# -------------------------------
print("\n🎯 Tuning threshold untuk macro-F1 dan recall kelas 0...")
thresh_range = np.arange(0.1, 0.91, 0.01)
best_f1 = best_recall_0 = 0
best_thresh_f1 = best_thresh_recall0 = 0.5

for t in thresh_range:
    y_thresh = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, y_thresh, average='macro')
    recall_0 = recall_score(y_test, y_thresh, pos_label=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh_f1 = t
    if recall_0 > best_recall_0:
        best_recall_0 = recall_0
        best_thresh_recall0 = t

print(f"🎯 Threshold terbaik (macro-F1): {best_thresh_f1:.2f} - F1: {best_f1:.4f}")
print(f"🎯 Threshold terbaik (recall kelas 0): {best_thresh_recall0:.2f} - Recall 0: {best_recall_0:.4f}")

# -------------------------------
print("\n📊 Evaluasi dengan threshold macro-F1:")
y_pred_f1 = (y_proba >= best_thresh_f1).astype(int)
print(confusion_matrix(y_test, y_pred_f1))
print(classification_report(y_test, y_pred_f1))

print("\n📊 Evaluasi dengan threshold recall kelas 0:")
y_pred_recall = (y_proba >= best_thresh_recall0).astype(int)
print(confusion_matrix(y_test, y_pred_recall))
print(classification_report(y_test, y_pred_recall))

# -------------------------------
print("\n📌 Feature importance:")
importances = best_clf.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]
for i in range(len(features)):
    print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")

plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# -------------------------------
print("\n💾 Menyimpan model dan metrik...")
metrics = {
    'accuracy': accuracy,
    'confusion_matrix': cm.tolist(),
    'labels': le_diabetes.classes_.tolist(),
    'best_params': grid_search.best_params_,
    'threshold_macro_f1': best_thresh_f1,
    'threshold_recall_0': best_thresh_recall0
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

MODEL_PATH = 'diabetes_rf_model_undersample.joblib'
ENCODER_PATH = 'sex_encoder_undersample.joblib'
LABEL_PATH = 'diabetes_label_encoder_undersample.joblib'
dump(best_clf, MODEL_PATH)
dump(le_sex, ENCODER_PATH)
dump(le_diabetes, LABEL_PATH)

# -------------------------------
print("\n✅ Proses selesai.")
print(f"Akurasi akhir: {accuracy:.4f}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)

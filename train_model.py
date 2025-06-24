import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Ganti path ke file dataset Anda
DATA_PATH = 'diabetes_dataset.xlsx'  # Pastikan file ini ada di folder backend

# Load data
try:
    df = pd.read_excel(DATA_PATH, header=4)
    print('Kolom pada file:', df.columns.tolist())
    # Hapus baris yang memiliki data kosong
    df = df.dropna()
    print(f'Jumlah baris setelah menghapus data kosong: {len(df)}')
    # Filter hanya data 'no diabetes' dan 'diabetes'
    df = df[df['Diabetes_012'].isin(['no diabetes', 'diabetes'])]
    # Tampilkan distribusi kelas sebelum oversampling
    print('Distribusi kelas sebelum oversampling:')
    print(df['Diabetes_012'].value_counts())
except Exception as e:
    print('Gagal membaca file:', e)
    raise

# Label encoding untuk kolom kategorikal
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])
le_diabetes = LabelEncoder()
df['Diabetes_012'] = le_diabetes.fit_transform(df['Diabetes_012'])

# Fitur dan target
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Hapus kolom yang kurang informatif
cols_to_drop = ['NoDocbcCost', 'AnyHealthcare', 'MentHlth', 'PhysHlth']
X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])

# Normalisasi fitur numerik
num_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

dump(scaler, 'scaler.joblib')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Terapkan SMOTE pada data latih
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print('Distribusi kelas setelah oversampling (SMOTE) pada data latih:')
print(pd.Series(y_train).value_counts())

# Hyperparameter tuning dengan GridSearchCV (grid diperkecil)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', None],
    'bootstrap': [True]
}
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(
    rf, param_grid, cv=5, n_jobs=-1, scoring='f1', verbose=2
)
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_

# Evaluasi modelh
accuracy = accuracy_score(y_test, best_clf.predict(X_test))
cm = confusion_matrix(y_test, best_clf.predict(X_test))
report = classification_report(y_test, best_clf.predict(X_test))

# Analisis feature importance
importances = best_clf.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]
print("\nFeature Importance Ranking:")
for i in range(len(features)):
    print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Analisis error pada prediksi kelas "no diabetes" (label 0)
y_pred = best_clf.predict(X_test)
error_idx = (y_test == 0) & (y_pred != 0)
print(f"\nJumlah data 'no diabetes' yang salah diprediksi: {error_idx.sum()} dari {sum(y_test == 0)}")
print("\nContoh data 'no diabetes' yang salah diprediksi:")
print(X_test[error_idx].head())

# Simpan metrik evaluasi ke file
metrics = {
    'accuracy': accuracy,
    'confusion_matrix': cm.tolist(),
    'labels': le_diabetes.classes_.tolist(),
    'best_params': grid_search.best_params_
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

# Simpan model dan encoder
MODEL_PATH = 'diabetes_rf_model.joblib'
ENCODER_PATH = 'sex_encoder.joblib'
LABEL_PATH = 'diabetes_label_encoder.joblib'
dump(best_clf, MODEL_PATH)
dump(le_sex, ENCODER_PATH)
dump(le_diabetes, LABEL_PATH)

print('Model terbaik, encoder, scaler, dan metrik evaluasi berhasil disimpan.')
print(f"Akurasi: {accuracy:.4f}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)
print("Parameter terbaik:")
print(grid_search.best_params_)
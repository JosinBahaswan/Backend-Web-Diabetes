import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
import json
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.inspection import permutation_importance

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
print("\n🔍 Analisis korelasi antar fitur...")
plt.figure(figsize=(12, 10))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Korelasi antar Fitur')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# -------------------------------
print("\n✨ Feature Engineering...")

# 1. Menambahkan interaksi antar fitur penting
print("1️⃣ Membuat fitur interaksi...")
X['BMI_Age'] = X['BMI'] * X['Age']
# Check if Glucose column exists
if 'Glucose' in X.columns:
    X['Glucose_BMI'] = X['Glucose'] * X['BMI']
    X['Glucose_Age'] = X['Glucose'] * X['Age']
else:
    print("⚠️ Kolom 'Glucose' tidak ditemukan, melewati fitur terkait Glucose")
X['GenHlth_BMI'] = X['GenHlth'] * X['BMI']
X['GenHlth_Age'] = X['GenHlth'] * X['Age']

# 2. Menambahkan fitur polinomial untuk fitur penting
print("2️⃣ Membuat fitur polinomial...")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
important_features = ['GenHlth', 'BMI', 'Age']
# Add Glucose if it exists
if 'Glucose' in X.columns:
    important_features.append('Glucose')
X_poly = poly.fit_transform(X[important_features])
poly_feature_names = [f"{important_features[i]}_poly" for i in range(len(important_features))]
X_poly_df = pd.DataFrame(X_poly[:, len(important_features):], 
                        columns=[f"poly_{i}" for i in range(X_poly.shape[1] - len(important_features))])
X = pd.concat([X, X_poly_df], axis=1)

# 3. Menambahkan fitur rasio
print("3️⃣ Membuat fitur rasio...")
X['BMI_to_Age'] = X['BMI'] / (X['Age'] + 1)  # +1 untuk menghindari pembagian dengan nol
if 'Glucose' in X.columns:
    X['Glucose_to_BMI'] = X['Glucose'] / (X['BMI'] + 1)
X['GenHlth_to_BMI'] = X['GenHlth'] / (X['BMI'] + 1)

# 4. Menambahkan fitur binning
print("4️⃣ Membuat fitur binning...")
X['BMI_bin'] = pd.qcut(X['BMI'], q=5, labels=False)
X['Age_bin'] = pd.qcut(X['Age'], q=5, labels=False)
if 'Glucose' in X.columns:
    X['Glucose_bin'] = pd.qcut(X['Glucose'], q=5, labels=False)

print(f"Jumlah fitur setelah feature engineering: {X.shape[1]}")

# Check for NaN values and handle them
print("\n🔍 Checking for NaN values...")
nan_count_before = X.isna().sum().sum()
print(f"Jumlah NaN sebelum handling: {nan_count_before}")

if nan_count_before > 0:
    # Fill NaN values with the median of each column
    for col in X.columns:
        if X[col].isna().sum() > 0:
            print(f"- Mengisi NaN di kolom {col} dengan median")
            X[col] = X[col].fillna(X[col].median())

# Make sure X and y have the same number of samples
print("\n🔄 Menyamakan jumlah sampel X dan y...")
print(f"Jumlah sampel X: {len(X)}, Jumlah sampel y: {len(y)}")

if len(X) != len(y):
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    print(f"Jumlah sampel setelah penyamaan: X={len(X)}, y={len(y)}")

# -------------------------------
print("\U0001F4CF Melakukan normalisasi pada fitur numerik...")
num_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
dump(scaler, 'scaler_balanced.joblib')
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
print("\n➕ Tahap 3: Menambahkan 15.000 data ke kelas mayoritas...")
# desired_majority_final = minority_count_new + 15000
# sampling_strategy_final = {majority_label: desired_majority_final, minority_label: minority_count_new}
# ros_final = RandomOverSampler(sampling_strategy=sampling_strategy_final, random_state=42)
# X_train, y_train = ros_final.fit_resample(X_train, y_train)
print('✅ Distribusi kelas akhir:')
print(pd.Series(y_train).value_counts())

# -------------------------------
print("\n🔍 Feature Selection dengan SelectFromModel...")
# Mengambil subset data untuk feature selection agar lebih cepat
subset_size = min(20000, len(X_train))
X_train_subset = X_train[:subset_size]
y_train_subset = y_train[:subset_size]

# Membuat model awal untuk seleksi fitur
pre_model = RandomForestClassifier(n_estimators=50, random_state=42)
pre_model.fit(X_train_subset, y_train_subset)

# Seleksi fitur berdasarkan importance
selector = SelectFromModel(pre_model, threshold='mean')
selector.fit(X_train_subset, y_train_subset)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Mendapatkan nama fitur yang terpilih
selected_features_mask = selector.get_support()
selected_features = X.columns[selected_features_mask]
print(f"Fitur yang terpilih ({len(selected_features)}):")
for feature in selected_features:
    print(f"- {feature}")

# Menggunakan fitur yang terpilih
X_train = X_train_selected
X_test = X_test_selected

# -------------------------------
print("\n⚖️ Cost-sensitive learning (kelas 1 dihukum 1.5x)...")
class_weights = {0: 1, 1: 1.5}  # Mengurangi penalti untuk kelas 1
rf = RandomForestClassifier(random_state=42, class_weight=class_weights)

# -------------------------------
print("\n🔁 Validasi silang (StratifiedKFold)...")
# Menggunakan subset data untuk validasi silang
subset_size_cv = min(30000, len(X_train))
X_train_subset_cv = X_train[:subset_size_cv]
y_train_subset_cv = y_train[:subset_size_cv]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_train_subset_cv, y_train_subset_cv, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"Skor F1-macro tiap fold: {scores}")
print(f"Rata-rata F1-macro: {scores.mean():.4f} (+/- {scores.std():.4f})")

# -------------------------------
print("\n🔄 Implementing proper cross-validation to prevent data leakage...")
# Define the original data before any resampling
X_original = X.copy()
y_original = y.copy()

# Split into train and test sets
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_original, y_original, test_size=0.2, stratify=y_original, random_state=42)

# Setup cross-validation
cv_proper = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proper_scores = []

# Perform proper cross-validation with preprocessing inside each fold
for train_idx, val_idx in cv_proper.split(X_train_orig, y_train_orig):
    # Get the training and validation data for this fold
    X_fold_train, X_fold_val = X_train_orig.iloc[train_idx], X_train_orig.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_orig.iloc[train_idx], y_train_orig.iloc[val_idx]
    
    # Apply SMOTE and undersampling only to the training data
    # First SMOTE
    minority_label = pd.Series(y_fold_train).value_counts().idxmin()
    majority_label = pd.Series(y_fold_train).value_counts().idxmax()
    minority_count = pd.Series(y_fold_train).value_counts()[minority_label]
    desired_minority = int(minority_count * 2.5)
    sampling_strategy_smote = {minority_label: desired_minority, majority_label: pd.Series(y_fold_train).value_counts()[majority_label]}
    smote_fold = SMOTE(sampling_strategy=sampling_strategy_smote, random_state=42)
    X_fold_train_smote, y_fold_train_smote = smote_fold.fit_resample(X_fold_train, y_fold_train)
    
    # Then undersampling
    minority_count_new = pd.Series(y_fold_train_smote).value_counts()[minority_label]
    sampling_strategy_under = {majority_label: minority_count_new, minority_label: minority_count_new}
    rus_fold = RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=42)
    X_fold_train_balanced, y_fold_train_balanced = rus_fold.fit_resample(X_fold_train_smote, y_fold_train_smote)
    
    # Feature selection on this fold's training data only
    pre_model_fold = RandomForestClassifier(n_estimators=50, random_state=42)
    pre_model_fold.fit(X_fold_train_balanced, y_fold_train_balanced)
    selector_fold = SelectFromModel(pre_model_fold, threshold='mean')
    selector_fold.fit(X_fold_train_balanced, y_fold_train_balanced)
    
    # Apply feature selection to both training and validation data
    X_fold_train_selected = selector_fold.transform(X_fold_train_balanced)
    X_fold_val_selected = selector_fold.transform(X_fold_val)
    
    # Train the model on the processed training data
    rf_fold = RandomForestClassifier(random_state=42, class_weight=class_weights)
    rf_fold.fit(X_fold_train_selected, y_fold_train_balanced)
    
    # Evaluate on the validation data
    y_fold_pred = rf_fold.predict(X_fold_val_selected)
    fold_f1 = f1_score(y_fold_val, y_fold_pred, average='macro')
    proper_scores.append(fold_f1)

print(f"Proper CV F1-macro scores: {proper_scores}")
print(f"Proper CV average F1-macro: {np.mean(proper_scores):.4f} (+/- {np.std(proper_scores):.4f})")

# Continue with the original pipeline for the final model
print("\n🧪 GridSearchCV tuning...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [5],
    'max_features': ['sqrt'],
    'min_samples_leaf': [2],
    'bootstrap': [True]
}

grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='f1', verbose=2)
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
print("\n🧪 Evaluasi model pada data asli (non-resampled)...")
# Prepare the original test data with the same feature selection
X_test_orig_selected = selector.transform(X_test_orig)

# Get predictions on original test data
y_proba_orig = best_clf.predict_proba(X_test_orig_selected)[:, 1]
y_pred_orig = best_clf.predict(X_test_orig_selected)
accuracy_orig = accuracy_score(y_test_orig, y_pred_orig)
cm_orig = confusion_matrix(y_test_orig, y_pred_orig)
report_orig = classification_report(y_test_orig, y_pred_orig)

print("Evaluasi pada data asli:")
print(f"Akurasi: {accuracy_orig:.4f}")
print("Confusion Matrix:")
print(cm_orig)
print("Classification Report:")
print(report_orig)

# Threshold tuning on original data
best_f1_orig = best_recall_0_orig = 0
best_thresh_f1_orig = best_thresh_recall0_orig = 0.5

for t in thresh_range:
    y_thresh_orig = (y_proba_orig >= t).astype(int)
    f1_orig = f1_score(y_test_orig, y_thresh_orig, average='macro')
    recall_0_orig = recall_score(y_test_orig, y_thresh_orig, pos_label=0)
    if f1_orig > best_f1_orig:
        best_f1_orig = f1_orig
        best_thresh_f1_orig = t
    if recall_0_orig > best_recall_0_orig:
        best_recall_0_orig = recall_0_orig
        best_thresh_recall0_orig = t

print(f"Threshold terbaik pada data asli (macro-F1): {best_thresh_f1_orig:.2f} - F1: {best_f1_orig:.4f}")
print(f"Threshold terbaik pada data asli (recall kelas 0): {best_thresh_recall0_orig:.2f} - Recall 0: {best_recall_0_orig:.4f}")

# Use these thresholds for the final metrics
metrics = {
    'accuracy': float(accuracy_orig),
    'confusion_matrix': cm_orig.tolist(),
    'labels': le_diabetes.classes_.tolist(),
    'best_params': grid_search.best_params_,
    'threshold_macro_f1': float(best_thresh_f1_orig),
    'threshold_recall_0': float(best_thresh_recall0_orig),
    'selected_features': list(range(1, X_train.shape[1] + 1)),
    'proper_cv_score': float(np.mean(proper_scores))
}

# -------------------------------
print("\n📌 Feature importance:")
if hasattr(best_clf, "feature_importances_"):
    importances = best_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), range(1, X_train.shape[1] + 1), rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importances_balanced.png')
    plt.close()
    
    print("Top 15 feature importances:")
    for i in range(min(15, X_train.shape[1])):
        print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
else:
    print("Model doesn't have feature_importances_ attribute")

# -------------------------------
print("\n🔄 Permutation Importance (more reliable)...")
perm_importance = permutation_importance(best_clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_indices = perm_importance.importances_mean.argsort()[::-1]

plt.figure(figsize=(12, 8))
plt.title("Permutation Importances")
plt.bar(range(X_train.shape[1]), perm_importance.importances_mean[perm_indices], align="center", yerr=perm_importance.importances_std[perm_indices])
plt.xticks(range(X_train.shape[1]), range(1, X_train.shape[1] + 1), rotation=90)
plt.tight_layout()
plt.savefig('permutation_importances_balanced.png')
plt.close()

print("Top 15 permutation importances:")
for i in range(min(15, X_train.shape[1])):
    print(f"{i+1}. Feature {perm_indices[i]}: {perm_importance.importances_mean[perm_indices[i]]:.4f} ± {perm_importance.importances_std[perm_indices[i]]:.4f}")

# -------------------------------
print("\n💾 Menyimpan model dan metrik...")
with open('metrics_balanced.json', 'w') as f:
    json.dump(metrics, f)

MODEL_PATH = 'diabetes_rf_model_balanced.joblib'
ENCODER_PATH = 'sex_encoder_balanced.joblib'
LABEL_PATH = 'diabetes_label_encoder_balanced.joblib'
SELECTOR_PATH = 'feature_selector_balanced.joblib'
dump(best_clf, MODEL_PATH)
dump(le_sex, ENCODER_PATH)
dump(le_diabetes, LABEL_PATH)
dump(selector, SELECTOR_PATH)

# -------------------------------
print("\n✅ Proses selesai.")
print(f"Akurasi akhir: {accuracy_orig:.4f}")
print("Confusion Matrix:")
print(cm_orig)
print("Classification Report:")
print(report_orig)
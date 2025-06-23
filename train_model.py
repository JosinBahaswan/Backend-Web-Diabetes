import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix
import json

# Ganti path ke file dataset Anda
DATA_PATH = 'diabetes_dataset.xlsx'  # Pastikan file ini ada di folder backend

# Load data
try:
    df = pd.read_excel(DATA_PATH, header=4)
    print('Kolom pada file:', df.columns.tolist())
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluasi model
accuracy = accuracy_score(y_test, clf.predict(X_test))
cm = confusion_matrix(y_test, clf.predict(X_test))

# Simpan metrik evaluasi ke file
metrics = {
    'accuracy': accuracy,
    'confusion_matrix': cm.tolist(),
    'labels': le_diabetes.classes_.tolist()
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

# Simpan model dan encoder
MODEL_PATH = 'diabetes_rf_model.joblib'
ENCODER_PATH = 'sex_encoder.joblib'
LABEL_PATH = 'diabetes_label_encoder.joblib'
dump(clf, MODEL_PATH)
dump(le_sex, ENCODER_PATH)
dump(le_diabetes, LABEL_PATH)

print('Model, encoder, dan metrik evaluasi berhasil disimpan.')

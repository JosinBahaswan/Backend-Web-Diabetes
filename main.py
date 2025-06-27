from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import base64
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

app = FastAPI()

# Load model dan encoder (menggunakan model balanced)
MODEL_PATH = 'diabetes_rf_model_balanced.joblib'
ENCODER_PATH = 'sex_encoder_balanced.joblib'
LABEL_PATH = 'diabetes_label_encoder_balanced.joblib'
SCALER_PATH = 'scaler_balanced.joblib'
SELECTOR_PATH = 'feature_selector_balanced.joblib'

assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} not found"
assert os.path.exists(ENCODER_PATH), f"{ENCODER_PATH} not found"
assert os.path.exists(LABEL_PATH), f"{LABEL_PATH} not found"
assert os.path.exists(SCALER_PATH), f"{SCALER_PATH} not found"
assert os.path.exists(SELECTOR_PATH), f"{SELECTOR_PATH} not found"

model = load(MODEL_PATH)
le_sex = load(ENCODER_PATH)
le_diabetes = load(LABEL_PATH)
scaler = load(SCALER_PATH)
feature_selector = load(SELECTOR_PATH)

# Get feature names in the exact order used during training
FEATURE_NAMES = feature_selector.feature_names_in_
print(f"Feature names from selector: {FEATURE_NAMES}")

# Load metrics untuk mendapatkan threshold optimal dari training
metrics_path = os.path.join(os.path.dirname(__file__), 'metrics_balanced.json')
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    # Ambil threshold dari hasil training
    THRESHOLD_F1 = metrics_data.get('threshold_macro_f1', 0.5)  # Untuk keseimbangan precision & recall
    THRESHOLD_RECALL0 = metrics_data.get('threshold_recall_0', 0.9)  # Untuk memaksimalkan recall kelas 0
    
    # Pilih threshold yang ingin digunakan (F1 atau Recall0)
    # Gunakan THRESHOLD_F1 untuk keseimbangan antara false positive dan false negative
    # Gunakan THRESHOLD_RECALL0 untuk meminimalkan false negative (lebih aman untuk diagnosis)
    THRESHOLD = THRESHOLD_F1  # Pilih threshold yang sesuai kebutuhan
else:
    THRESHOLD_F1 = THRESHOLD_RECALL0 = THRESHOLD = 0.5  # Default jika metrics.json tidak ditemukan

# Daftar fitur yang dipakai oleh model (fitur asli)
ORIGINAL_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'GenHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

class DiabetesInput(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int = 0  # Default value if not provided
    DiffWalk: int = 0  # Default value if not provided
    Sex: str
    Age: int
    Education: int
    Income: int
    GenHlth: int
    threshold_type: str = "f1"  # Hanya menggunakan f1 mode seimbang

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain frontend jika ingin lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/predict')
def predict_diabetes(data: DiabetesInput):
    try:
        input_dict = data.dict()
        
        # Selalu gunakan threshold F1 untuk keseimbangan
        threshold = THRESHOLD_F1
        threshold_type = "f1"  # Selalu f1
        
        # Encode kolom Sex
        input_dict['Sex'] = int(le_sex.transform([input_dict['Sex']])[0])
        
        # Buat DataFrame dengan kolom yang tepat sesuai urutan feature_names_in_
        df = pd.DataFrame([input_dict])
        
        # Feature Engineering (sama dengan yang di train_model_balanced.py)
        # 1. Menambahkan interaksi antar fitur penting
        df['BMI_Age'] = df['BMI'] * df['Age']
        df['GenHlth_BMI'] = df['GenHlth'] * df['BMI']
        df['GenHlth_Age'] = df['GenHlth'] * df['Age']
        
        # 2. Menambahkan fitur polinomial untuk fitur penting
        important_features = ['GenHlth', 'BMI', 'Age']
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_poly = poly.fit_transform(df[important_features])
        poly_feature_names = [f"poly_{i}" for i in range(X_poly.shape[1] - len(important_features))]
        X_poly_df = pd.DataFrame(X_poly[:, len(important_features):], columns=poly_feature_names)
        df = pd.concat([df, X_poly_df], axis=1)
        
        # 3. Menambahkan fitur rasio
        df['BMI_to_Age'] = df['BMI'] / (df['Age'] + 1)  # +1 untuk menghindari pembagian dengan nol
        df['GenHlth_to_BMI'] = df['GenHlth'] / (df['BMI'] + 1)
        
        # 4. Menambahkan fitur binning (menggunakan nilai diskrit untuk Age dan BMI)
        df['BMI_bin'] = pd.cut(df['BMI'], bins=5, labels=False)
        df['Age_bin'] = pd.cut(df['Age'], bins=5, labels=False)
        
        # Pastikan semua kolom yang dibutuhkan ada dalam DataFrame
        for col in FEATURE_NAMES:
            if col not in df.columns:
                print(f"Warning: Column {col} is missing. Adding with default value 0.")
                df[col] = 0
        
        # Pastikan urutan kolom sesuai dengan feature_names_in_
        df = df[FEATURE_NAMES]
        
        # Normalisasi fitur numerik
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = scaler.transform(df[num_cols])
        
        # Apply feature selection
        input_data_selected = feature_selector.transform(df)
        
        # Dapatkan probabilitas prediksi
        proba = model.predict_proba(input_data_selected)[0]
        class_labels = le_diabetes.classes_.tolist()
        
        # Cari indeks kelas "diabetes" (biasanya indeks 0)
        diabetes_idx = class_labels.index("diabetes") if "diabetes" in class_labels else 0
        
        # Gunakan threshold dari training untuk prediksi
        if proba[diabetes_idx] < threshold:
            pred_label = "no diabetes"
            pred_idx = class_labels.index("no diabetes") if "no diabetes" in class_labels else 1
        else:
            pred_label = "diabetes"
            pred_idx = diabetes_idx
        
        # Kembalikan hasil prediksi, label, dan probabilitas
        return {
            'prediction': pred_idx,
            'label': pred_label,
            'probabilities': {str(class_labels[i]): float(prob) for i, prob in enumerate(proba)},
            'class_labels': class_labels,
            'threshold_used': threshold,
            'threshold_type': threshold_type,
            'model_used': 'balanced'
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {'error': str(e)}

@app.get('/')
def root():
    return {'message': 'API Prediksi Diabetes siap digunakan.'}

@app.get('/metrics')
def get_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), 'metrics_balanced.json')
    if not os.path.exists(metrics_path):
        return {'error': 'metrics_balanced.json not found'}
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    # Pastikan confusion matrix dan label ada
    confusion_matrix = metrics.get('confusion_matrix')
    labels = metrics.get('labels')
    accuracy = metrics.get('accuracy')
    threshold_f1 = metrics.get('threshold_macro_f1')
    threshold_recall_0 = metrics.get('threshold_recall_0')
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'labels': labels,
        'threshold_f1': threshold_f1,
        'threshold_recall_0': threshold_recall_0
    }

@app.post('/save-table-image')
def save_table_image(image_base64: str):
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        # Buat nama file unik
        filename = f"saved_images/table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with open(filename, 'wb') as f:
            f.write(image_data)
        return {"message": "Gambar tabel berhasil disimpan.", "filename": filename}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
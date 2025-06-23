from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from fastapi import FastAPI, UploadFile, File
import base64
from datetime import datetime

app = FastAPI()

# Load model dan encoder
MODEL_PATH = 'diabetes_rf_model.joblib'
ENCODER_PATH = 'sex_encoder.joblib'
LABEL_PATH = 'diabetes_label_encoder.joblib'
model = load(MODEL_PATH)
le_sex = load(ENCODER_PATH)
le_diabetes = load(LABEL_PATH)

# Daftar fitur yang diharapkan
FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
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
    HvyAlcoholConsump: int
    AnyHealthcare: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: str
    Age: int
    Education: int
    Income: int

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
        # Encode kolom Sex
        input_dict['Sex'] = int(le_sex.transform([input_dict['Sex']])[0])
        # Urutkan sesuai fitur
        input_data = np.array([input_dict[feat] for feat in FEATURES]).reshape(1, -1)
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        class_labels = le_diabetes.classes_.tolist()
        print("Predicted index:", pred)
        print("Predicted label:", le_diabetes.inverse_transform([pred])[0])
        print("Probabilities:", proba)
        print("Class labels:", class_labels)
        # Ambil prediksi berdasarkan probabilitas tertinggi
        pred_proba_idx = int(np.argmax(proba))
        pred_label = class_labels[pred_proba_idx]
        print("Predicted index (argmax):", pred_proba_idx)
        print("Predicted label (argmax):", pred_label)
        print("Probabilities:", proba)
        print("Class labels:", class_labels)
        # Kembalikan hasil prediksi, label, dan probabilitas
        return {
            'prediction': pred_proba_idx,
            'label': pred_label,
            'probabilities': {str(class_labels[i]): float(prob) for i, prob in enumerate(proba)},
            'class_labels': class_labels
        }
    except Exception as e:
        return {'error': str(e)}

@app.get('/')
def root():
    return {'message': 'API Prediksi Diabetes siap digunakan.'}

@app.get('/metrics')
def get_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), 'metrics.json')
    if not os.path.exists(metrics_path):
        return {'error': 'metrics.json not found'}
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

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
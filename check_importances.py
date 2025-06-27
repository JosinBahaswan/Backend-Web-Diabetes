from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
print("Loading model...")
model = load('diabetes_rf_model_balanced.joblib')

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Define the original feature names
original_features = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", 
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", 
    "Veggies", "HvyAlcoholConsump", "GenHlth", "DiffWalk", 
    "Sex", "Age", "Education", "Income"
]

# Define the engineered features
engineered_features = [
    "BMI_Age", "GenHlth_BMI", "GenHlth_Age", 
    "poly_0", "poly_1", "poly_2", "poly_3", "poly_4", "poly_5",
    "BMI_to_Age", "GenHlth_to_BMI", "BMI_bin", "Age_bin"
]

# Combine all feature names
all_features = original_features + engineered_features

# Print feature importances with names
print("\nFeature importances with names:")
for i in range(len(importances)):
    idx = indices[i]
    if idx < len(all_features):
        feature_name = all_features[idx]
    else:
        feature_name = f"Unknown Feature {idx+1}"
    
    print(f"{i+1}. {feature_name}: {importances[idx]:.4f} ({idx+1})")

# Print top 5 features
print("\nTop 5 most important features:")
for i in range(5):
    idx = indices[i]
    if idx < len(all_features):
        feature_name = all_features[idx]
    else:
        feature_name = f"Unknown Feature {idx+1}"
    
    print(f"{i+1}. {feature_name}: {importances[idx]:.4f} ({importances[idx]*100:.2f}%)")

print("\nDone!") 
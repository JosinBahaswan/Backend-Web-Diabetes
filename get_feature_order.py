from joblib import load
import pandas as pd
import numpy as np
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model and feature selector
print("Loading model and feature selector...")
model_path = os.path.join(current_dir, 'diabetes_rf_model_balanced.joblib')
selector_path = os.path.join(current_dir, 'feature_selector_balanced.joblib')
scaler_path = os.path.join(current_dir, 'scaler_balanced.joblib')

print(f"Model path: {model_path}")
print(f"Selector path: {selector_path}")
print(f"Scaler path: {scaler_path}")

model = load(model_path)
selector = load(selector_path)
scaler = load(scaler_path)

# Try to get feature names from the model
print("\nTrying to get feature names from the model...")
if hasattr(model, 'feature_names_in_'):
    print("Model has feature_names_in_ attribute!")
    print(model.feature_names_in_)
else:
    print("Model does not have feature_names_in_ attribute")

# Try to get feature names from the selector
print("\nTrying to get feature names from the selector...")
if hasattr(selector, 'feature_names_in_'):
    print("Selector has feature_names_in_ attribute!")
    print(selector.feature_names_in_)
else:
    print("Selector does not have feature_names_in_ attribute")

# Try to get feature names from the scaler
print("\nTrying to get feature names from the scaler...")
if hasattr(scaler, 'feature_names_in_'):
    print("Scaler has feature_names_in_ attribute!")
    print(scaler.feature_names_in_)
else:
    print("Scaler does not have feature_names_in_ attribute")

# Print selector properties
print("\nSelector properties:")
for attr in dir(selector):
    if not attr.startswith('_'):
        try:
            value = getattr(selector, attr)
            if not callable(value):
                print(f"{attr}: {value}")
        except:
            pass

# Try to get the selected features
print("\nTrying to get selected features...")
if hasattr(selector, 'get_support'):
    selected_mask = selector.get_support()
    print(f"Selected mask: {selected_mask}")
    print(f"Number of selected features: {sum(selected_mask)}")
else:
    print("Selector does not have get_support method")

print("\nDone!") 
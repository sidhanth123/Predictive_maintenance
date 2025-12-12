import pandas as pd
import joblib

MODEL_PATH = "models/best_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

# Example new data point
new_data = pd.DataFrame([{
    'Air temperature [K]': 300,
    'Process temperature [K]': 310,
    'Rotational speed [rpm]': 1600,
    'Torque [Nm]': 40,
    'Tool wear [min]': 15,
    'Target': 0,  # Will be dropped
    'Type_L': 1,  # Example encoded categories
    'Type_M': 0,
}])

# Ensure same features as training data (drop target)
if 'Target' in new_data.columns:
    new_data = new_data.drop(columns=['Target'])

prediction = model.predict(new_data)[0]
prob = model.predict_proba(new_data)[0][1]

print(f"Prediction: {'Failure' if prediction==1 else 'No Failure'}")
print(f"Failure Probability: {prob:.2f}")

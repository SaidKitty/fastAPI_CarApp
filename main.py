from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

app = FastAPI()

# Load model and encoder
model = joblib.load('xgboost_model_onehot2.pkl')
encoder = joblib.load('onehot_encoder.pkl')

# Request model
class CarInput(BaseModel):
    make: str
    model: str
    year: int
    condition: str
    mileage: float
    fuel_type: str
    volume: float
    color: str
    transmission: str
    drive_unit: str
    segment: str

@app.post("/predict")
async def predict_price(features: CarInput):
    input_data = pd.DataFrame([features.dict()])

    # Rename columns to match the trained model
    input_data.rename(columns={
        "mileage": "mileage(kilometers)",
        "volume": "volume(cm3)"
    }, inplace=True)

    # Ensure all expected columns exist
    expected_columns = ['make', 'model', 'year', 'condition', 'mileage(kilometers)', 
                        'fuel_type', 'volume(cm3)', 'color', 'transmission', 
                        'drive_unit', 'segment']
    
    missing_cols = [col for col in expected_columns if col not in input_data.columns]
    if missing_cols:
        return {"error": f"Missing columns: {missing_cols}"}

    # Apply the trained encoder
    input_encoded = encoder.transform(input_data)  # Ensure `encoder` is loaded correctly

    # Pass the correctly formatted data to the model
    prediction = model.predict(input_encoded)
    
    return {"predicted_price": prediction[0]}

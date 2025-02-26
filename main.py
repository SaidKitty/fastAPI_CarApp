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
def predict_price(data: CarInput):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Encode categorical features
    input_encoded = encoder.transform(input_data)
    
    # Predict price
    prediction = model.predict(input_encoded)[0]
    
    return {"predicted_price": round(prediction, 2)}

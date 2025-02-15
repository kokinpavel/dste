import pickle
from typing import Any, Dict

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load CatBoost model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load preprocess model
with open("models/preprocess_model.pkl", "rb") as f:
    preprocess_model = pickle.load(f)


def preprocess(input_data: pd.DataFrame):
    return preprocess_model.transform(input_data)


# Define input schema
class InputData(BaseModel):
    features: Dict[str, Any]  # JSON


@app.post("/predict/")
def predict(input_data: InputData):
    # Convert input to DataFrame
    input = pd.DataFrame([input_data.features])
    input = preprocess(input)

    # Run inference
    prediction = model.predict_proba(input)[:, 1]
    return {"prediction": prediction[0]}


# Run API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)

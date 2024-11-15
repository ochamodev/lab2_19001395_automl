from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np
import uvicorn

app = FastAPI()

global model

class DataRequest(BaseModel):
    data: list

@app.post("/predict")
async def predict(request: DataRequest):
    data = np.array(request.data)
    predictions = model.predict_proba(data)
    results = [{"Clase" + str(i): prob for i, prob in enumerate(pred)} for pred in predictions]
    return {"predictions": results}

def start_api(port=8000):
    model = load("model/trained_model.joblib")
    uvicorn.run(app, host="0.0.0.0", port=port)

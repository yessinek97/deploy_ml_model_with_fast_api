"""Implementation of the fast api"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.model import inference
from model.data import process_data

app = FastAPI()

class Value(BaseModel):
  value: int

class InferenceInput(BaseModel):
  model: float
  data: str

@app.get('/')
async def welcome_message():
  return {
    "This is a welcome message to the API",
}

@app.post("/predict/")
async def ingest_data(inference_input: InferenceInput):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, _, _, _ = process_data(
        inference_input.data,
        categorical_features=cat_features,
        training=False,
    )
    preds = inference_input.model.predict(X)
    return preds

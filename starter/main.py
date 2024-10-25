"""Implementation of the fast api"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.model import inference
from model.data import process_data

app = FastAPI()

class Value(BaseModel):
  value: int

class Data(BaseModel):
  feature_1: float
  feature_2: str

@app.get('/')
async def welcome_message():
  return {
    "This is a welcome message to the API",
}

@app.post("/data/")
async def ingest_data(data, model):
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
        data,
        categorical_features=cat_features,
        training=False,
    )
    preds = model.predict(X)
    return preds

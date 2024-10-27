"""Implementation of the fast api"""
from fastapi import FastAPI
from pydantic import BaseModel
from model.model import inference
from model.data import process_data
import pandas as pd
from sklearn.model_selection import train_test_split
from model.model import train_model
from typing import Any
import os
from pathlib import Path

data_path = os.path.join(
    os.path.abspath("."),
    Path(__file__).parent,
    "data/census.csv",
)
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
model = train_model(X_train, y_train)

app = FastAPI()


class InferenceInput(BaseModel):
    arbitrary_types_allowed: bool = True
    model: Any
    data: Any


@app.get('/')
async def welcome_message():
    return {
        "This is a welcome message to the API",
    }


@app.post("/predict/{sinppet_size}")
async def predict(snippet_size: int = 10):
    inference_input = InferenceInput(
        model=model,
        data=test,
    )
    X, _, _, _ = process_data(
        inference_input.data,
        categorical_features=cat_features,
        training=False,
    )
    preds = inference(model, X)
    print("here is a sinppet of the predicted output:\n")
    print({
        i: prediction 
        for i, prediction in enumerate(preds[:snippet_size])
    })
    return {'preds: '}

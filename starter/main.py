"""Implementation of the fast api"""
from fastapi import FastAPI
from pydantic import BaseModel
from model.model import inference
from model.data import process_data
import pandas as pd
from sklearn.model_selection import train_test_split
from model.model import train_model
from typing import Any

data_path = "~/data/census.csv"
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


@app.post("/predict/")
async def predict(inference_input: InferenceInput):
    X, _, _, _ = process_data(
        inference_input.data,
        categorical_features=cat_features,
        training=False,
    )
    preds = inference(model, X)
    return {'preds: ', preds}

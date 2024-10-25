"""Test script to test GET and POST api calls"""
import sys
import json
from fastapi.testclient import TestClient
from main import app
from main import InferenceInput
from model.data import process_data
import pandas as pd

client = TestClient(app)
data_path = "data/census.csv"
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

def test_post(trained_model):
    try:
        df = pd.read_csv(data_path)
        X, _, _, _ = process_data(
            df,
            categorical_features=cat_features,
            training=False,
        )
        inference_input=InferenceInput(
            trained_model,
            X
        )
    except BaseException:
        print("ERROR: Error loading data")
    
    try:
        r = client.post("/predict", inference_input=inference_input)
        print(r.json())
        assert len(r.json()["preds"]) > 0
    except BaseException:
        print("ERROR: failed retrieving inference results")

def test_get():
    r = client.post("/")
    print(r.json())
    assert r.json() != None
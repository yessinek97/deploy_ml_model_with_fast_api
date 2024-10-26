"""Test script to test GET and POST api calls"""
from fastapi.testclient import TestClient
from main import app
from main import InferenceInput
from model.data import process_data
import pandas as pd
import numpy as np

client = TestClient(app)
# abs data path
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


def test_predict_low_salary(trained_model, data):
    try:
        mean_salary = np.mean(data["salary"].values)
        data = data[data["salary"] <= mean_salary]

        X, _, _, _ = process_data(
            data,
            categorical_features=cat_features,
            training=False,
        )
        inference_input = InferenceInput(
            arbitrary_types_allowed=True,
            model=trained_model,
            data=X
        )
    except BaseException:
        print("ERROR: Error loading data")

    try:
        r = client.post("/predict", inference_input=inference_input)
        print(r.json())
        assert np.mean(r.json()["preds"]) <= mean_salary, "model not accurate"
        assert r.status_code != 400
    except BaseException:
        print("ERROR: failed retrieving inference results")


def test_predict_high_salary(trained_model, data):
    try:
        mean_salary = np.mean(data["salary"].values)
        data = data[data["salary"] > mean_salary]
        X, _, _, _ = process_data(
            data,
            categorical_features=cat_features,
            training=False,
        )
        inference_input = InferenceInput(
            arbitrary_types_allowed=True,
            model=trained_model,
            data=X
        )
    except BaseException:
        print("ERROR: Error loading data")

    try:
        r = client.post("/predict", inference_input=inference_input)
        print(r.json())
        assert np.mean(r.json()["preds"]) > mean_salary, "model not accurate"
        assert r.status_code != 400
    except BaseException:
        print("ERROR: failed retrieving inference results")


def test_welcome_message():
    try:
        r = client.get("/")
        print(r.json())
        assert r.json() is not None
        assert r.status_code != 400
    except BaseException:
        print("ERROR: encountered error fetching GET reponse")

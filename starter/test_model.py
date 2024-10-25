"""test script to test model training and inference"""
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score


def test_compute_model_metrics(model, process_data):
    X, y, _, _ = process_data
    try:
        preds = model.predict(X)
        _ = fbeta_score(y, preds, beta=1, zero_division=1)
        _ = precision_score(y, preds, zero_division=1)
        _ = recall_score(y, preds, zero_division=1)
    except BaseException:
        print('ERROR: encountered error during evaluation')


def test_inference(model, process_data):
    X, _, _, _ = process_data
    try:
        X = np.random_choice(X, len(X) // 10)
        _ = model.predict(X)
    except BaseException:
        print("ERROR: encountered error during inference")

"""Script for model training and inference"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from model.data import process_data
import numpy as np


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # instanciate model and fit it
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Trained Random Forest model.
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def eval_model_on_slices(model, data):
    """Computes model performance on slices of data.

    Args:
        model (_type_): Trained_model.
        data (_type_): Data used in training.
    """
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
    print("Sliced Model Evaluation: \n")
    for feature in cat_features:
        if data[feature].unique() == np.array([0, 1]):
            slices = [data[data[feature] == value]
                      for value in np.array([0, 1])]
        else:
            feature_mean = np.mean(data[feature].values)
            slices = [
                data[data[feature] >= feature_mean],
                data[data[feature] < feature_mean]
            ]

        for slice_idx, slice in enumerate(slices):
            X, y, _, _ = process_data(
                slice,
                categorical_features=cat_features,
                label="salary",
                training=False,
            )
            preds = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            print(f"\nFeature {feature} Slice {slice_idx}:\n")
            print(
                f"precision :{precision} - recall: {recall} - fbeta: {fbeta}")

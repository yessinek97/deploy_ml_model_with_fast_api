import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pytest
from ml.model import train_model

def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--label", action="store")
    
@pytest.fixture(scope="session")
def data(request):
    try:
        data_path = request.config.option.csv
        df = pd.read_csv(data_path)
        assert len(df) > 1, 'ERROR: data frame has no rows'
    except:
        print('ERROR: problem loading dataset')
    return df

@pytest.fixture(scope="session")
def process_data(data, request):
    categorical_features = [
        "workclass",
        "education",
        "education-num",
        "martial-status",
        "occupation",
        "race",
        "sex",
        "relatioship",
        "native-country",
    ]
    try:
        label = request.config.option.label
        y = data[label]
        X = data.drop([label], axis=1)

        X_categorical = X[categorical_features].values
        X_continuous = X.drop(*[categorical_features], axis=1)

        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()

        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()

        assert len(X_categorical)>0, 'there has to be at least one categorical feature'
        assert len(X_continuous)>0, 'there has to be at least one continuous feature'

    except:
        print('ERROR: encountered problem processing data')
    
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return (X, y, encoder, lb)

@pytest.fixture(scope="session")  
def test_train_model(process_data):
    X, y, _, _ = process_data
    try:
        model = train_model(X, y)
    except:
        print('ERROR: Encountered problem when trying to train the model')
    return model
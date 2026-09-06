import os
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def sample_data():
    """To provide sample DataFrame mimicking the census data."""
    data = {
        'age': [39, 50, 38, 53, 28],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private'],
        'fnlgt': [77516, 83311, 215646, 234721, 338409],
        'education': [
            'Bachelors',
            'Bachelors',
            'HS-grad',
            '11th',
            'Bachelors',
        ],
        'education-num': [13, 13, 9, 7, 13],
        'marital-status': [
            'Never-married',
            'Married-civ-spouse',
            'Divorced',
            'Married-civ-spouse',
            'Married-civ-spouse',
        ],
        'occupation': [
            'Adm-clerical',
            'Exec-managerial',
            'Handlers-cleaners',
            'Handlers-cleaners',
            'Prof-specialty',
        ],
        'relationship': [
            'Not-in-family',
            'Husband',
            'Not-in-family',
            'Husband',
            'Wife',
        ],
        'race': [
            'White',
            'White',
            'White',
            'Black',
            'Black',
        ],
        'sex': ['Male', 'Male', 'Male', 'Male', 'Female'],
        'capital-gain': [2174, 0, 0, 0, 0],
        'capital-loss': [0, 0, 0, 0, 0],
        'hours-per-week': [40, 13, 40, 40, 40],
        'native-country': [
            'United-States',
            'United-States',
            'United-States',
            'United-States',
            'Cuba',
        ],
        'salary': ['<=50K', '<=50K', '<=50K', '<=50K', '<=50K'],
    }
    return pd.DataFrame(data)

def test_process_data(sample_data):
    """Test that process_data returns expected types and processed matrices."""
    categorical_features = [
        "workclass", "education", "marital-status", 
        "occupation", "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(
        sample_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=True
    )
    
    # Check that features and labels are returned as expected arrays/matrices
    assert len(X) == len(sample_data)
    assert len(y) == len(sample_data)
    assert encoder is not None
    assert lb is not None

def test_train_model(sample_data):
    """Test that train_model returns trained RandomForestClassifier instance."""
    categorical_features = [
        "workclass", "education", "marital-status", 
        "occupation", "relationship", "race", "sex", "native-country"
    ]
    X, y, _, _ = process_data(
        sample_data, 
        categorical_features=categorical_features, 
        label="salary", 
        training=True
    )
    
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics():
    """Test that compute_model_metrics returns precision, recall, and f1 scores within [0, 1]."""
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
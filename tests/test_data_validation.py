from sklearn.datasets import load_iris
import numpy as np


def load_data():
    X, y = load_iris(return_X_y=True)
    return X, y


def test_no_nulls():
    X, y = load_data()
    assert not np.isnan(X).any(), "Features contain null values"
    assert not np.isnan(y).any(), "Labels contain null values"


def test_feature_count():
    X, _ = load_data()
    assert X.shape[1] == 4, f"Expected 4 features, got {X.shape[1]}"


def test_label_values():
    _, y = load_data()
    unique = set(np.unique(y))
    assert unique == {0, 1, 2}, f"Unexpected labels: {unique}"


def test_no_empty_dataset():
    X, y = load_data()
    assert len(X) > 0, "Dataset is empty"
    assert len(X) == len(y), "Feature and label count mismatch"


def test_feature_ranges():
    X, _ = load_data()
    assert (X >= 0).all(), "Negative values found in features"
    assert (X <= 10).all(), "Unreasonably large values found"
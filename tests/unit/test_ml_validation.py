"""
Test ML pipeline validation as described in copilot instructions.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def test_ml_pipeline():
    """Test basic ML pipeline functionality."""
    # Create sample data and test ML pipeline
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Basic validation
    assert model.score(X_test, y_test) > 0.5, "Model accuracy too low"
    print("✅ ML pipeline validation passed")


def test_data_structures():
    """Test basic data structure operations."""
    # Test NumPy
    arr = np.array([1, 2, 3, 4, 5])
    assert np.mean(arr) == 3.0, "NumPy mean calculation failed"

    # Test Pandas
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    assert df.shape == (3, 2), "DataFrame shape incorrect"

    print("✅ Data structure validation passed")


if __name__ == "__main__":
    test_ml_pipeline()
    test_data_structures()

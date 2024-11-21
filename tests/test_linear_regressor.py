import pytest
import numpy as np
import sys
import os

# Adding src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from linear_regressor import compute_cost, compute_gradients

@pytest.mark.parametrize(
    "X, y, w, b, lambda_, expected_cost", [
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, 2, 3]), np.array([0.5, 0.5]), 0.5, 0.1, 2.34166667),
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, 2, 3]), np.zeros(2), 0, 0.1, 2.333333333333333),
        (np.array([[1, 2]]), np.array([1]), np.array([0.5, 0.5]), 0.5, 0.1, 0.525),
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, 2, 3]), np.array([0.5, 0.5]), 0.5, 0.0, 2.333333333333333),
        (np.array([[1000, 2000], [3000, 4000], [5000, 6000]]), np.array([1000, 2000, 3000]), np.array([0.5, 0.5]), 500, 0.1, 2333333.341666665),
    ]
)
def test_compute_cost(X, y, w, b, lambda_, expected_cost):
    actual_cost = compute_cost(X, y, w, b, lambda_)
    assert np.isclose(actual_cost, expected_cost), f"Expected {expected_cost}, but got {actual_cost}"

def test_compute_cost_empty_dataset():
    X = np.array([[]])
    y = np.array([])
    w = np.array([0.5, 0.5])
    b = 0.5
    lambda_ = 0.1
    with pytest.raises(ValueError, match="Training set is empty. Please provide a non-empty dataset."):
        compute_cost(X, y, w, b, lambda_)

@pytest.mark.parametrize(
    "X, y, w, b, lambda_, expected_dj_dw, expected_dj_db", [
        # Standard Case
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, 2, 3]), 
         np.array([0.5, 0.5]), 0.5, 0.1, 
         np.array([7.35, 9.35]), 2),
        
        # Zero Weights and Bias
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, 2, 3]), 
         np.zeros(2), 0, 0.1, 
         np.array([-7.33333333, -9.33333333]), -2),
        
        # Single Data Point
        (np.array([[1, 2]]), np.array([1]), 
         np.array([0.5, 0.5]), 0.5, 0.1, 
         np.array([1.05, 2.05]), 1),
        
        # No Regularization
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, 2, 3]), 
         np.array([0.5, 0.5]), 0.5, 0.0, 
         np.array([7.33333333, 9.33333333]), 2),
        
        # Large Values
        (np.array([[1000, 2000], [3000, 4000], [5000, 6000]]), np.array([1000, 2000, 3000]), 
         np.array([0.5, 0.5]), 500, 0.1, 
         np.array([7333333.35, 9333333.35]), 2000),
    ]
)
def test_compute_gradients(X, y, w, b, lambda_, expected_dj_dw, expected_dj_db):
    actual_dj_dw, actual_dj_db = compute_gradients(X, y, w, b, lambda_)
    
    # Assert the values are close
    assert np.allclose(actual_dj_dw, expected_dj_dw), f"Expected dj_dw {expected_dj_dw}, but got {actual_dj_dw}"
    assert np.isclose(actual_dj_db, expected_dj_db), f"Expected dj_db {expected_dj_db}, but got {actual_dj_db}"

def test_compute_gradients_empty_dataset():
    X = np.array([[]])
    y = np.array([])
    w = np.array([0.5, 0.5])
    b = 0.5
    lambda_ = 0.1
    with pytest.raises(ValueError, match="Training set is empty. Please provide a non-empty dataset."):
        compute_gradients(X, y, w, b, lambda_)
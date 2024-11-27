import pytest
import numpy as np
import sys
import os

# Adding src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from logistic_regressor import sigmoid, compute_logistic_cost, compute_logistic_gradients, evaluate_model_performance

def test_sigmoid(): 
    # Test sigmoid function 
    z = np.array([0, 2, -2]) 
    expected = np.array([0.5, 0.8808, 0.1192]) 
    np.testing.assert_almost_equal(sigmoid(z), expected, decimal=4)

@pytest.mark.parametrize(
    "X, y, w, b, lambda_, expected_cost", [
        # Standard case
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 1]), 
         np.array([0.5, 0.5]), 0.5, 0.1, 
         0.72418482),
        
        # Zero weights and bias
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 1]), 
         np.zeros(2), 0, 0.1, 
         0.69314718),
        
        # Single Data Point
        (np.array([[1, 2]]), np.array([1]), 
         np.array([0.5, 0.5]), 0.5, 0.1, 
         0.15192801),
        
        # No regularization
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 1]), 
         np.array([0.5, 0.5]), 0.5, 0.0, 
         0.71585149),
        
        # Large Values
        (np.array([[1000, 2000], [3000, 4000], [5000, 6000]]), np.array([0, 1, 1]), 
         np.array([0.5, 0.5]), 500, 0.1, 
         11.52152533),
    ]
)
def test_compute_logistic_cost(X, y, w, b, lambda_, expected_cost):
    actual_cost = compute_logistic_cost(X, y, w, b, lambda_)
    assert np.isclose(actual_cost, expected_cost), f"Expected {expected_cost}, but got {actual_cost}"

def test_compute_logistic_cost_empty_dataset():
    X = np.array([[]])
    y = np.array([])
    w = np.array([0.5, 0.5])
    b = 0.5
    lambda_ = 0.1
    with pytest.raises(ValueError, match="Training set is empty. Please provide a non-empty dataset."):
        compute_logistic_cost(X, y, w, b, lambda_)

@pytest.mark.parametrize(
    "X, y, w, b, lambda_, expected_dj_dw, expected_dj_db", [
        # Standard Case
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 1]), 
         np.array([0.5, 0.5]), 0.5, 0.1, 
         np.array([0.28815845, 0.57493787]), 0.28677942),
        
        # Zero Weights and Bias
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 1]), 
         np.zeros(2), 0, 0.1, 
         np.array([-1.16666667, -1.33333333]), -0.16666667),
        
        # Single Data Point
        (np.array([[1, 2]]), np.array([1]), 
         np.array([0.5, 0.5]), 0.5, 0.1, 
         np.array([-0.0692029, -0.1884058]), -0.1192029),
        
        # No Regularization
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 1]), 
         np.array([0.5, 0.5]), 0.5, 0.0, 
         np.array([0.27149178, 0.55827120]), 0.28677942),
        
        # Large Values
        (np.array([[1000, 2000], [3000, 4000], [5000, 6000]]), np.array([0, 1, 1]), 
         np.array([0.5, 0.5]), 500, 0.1, 
         np.array([333.350, 666.683]), 0.33333333),
    ]
)
def test_compute_logistic_gradients(X, y, w, b, lambda_, expected_dj_dw, expected_dj_db):
    actual_dj_dw, actual_dj_db = compute_logistic_gradients(X, y, w, b, lambda_)
    
    # Assert the values are close
    assert np.allclose(actual_dj_dw, expected_dj_dw), f"Expected dj_dw {expected_dj_dw}, but got {actual_dj_dw}"
    assert np.isclose(actual_dj_db, expected_dj_db), f"Expected dj_db {expected_dj_db}, but got {actual_dj_db}"

def test_compute_logistic_gradients_empty_dataset():
    X = np.array([[]])
    y = np.array([])
    w = np.array([0.5, 0.5])
    b = 0.5
    lambda_ = 0.1
    with pytest.raises(ValueError, match="Training set is empty. Please provide a non-empty dataset."):
        compute_logistic_gradients(X, y, w, b, lambda_)

def test_evaluate_model_performance():
    # Sample data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])
    w = np.array([0.5, 0.5])
    b = 0.5
    expected_accuracy = 2/3
     
    # Evaluate model performance
    accuracy = evaluate_model_performance(X, y, w, b)
    
    assert accuracy == expected_accuracy, f"Expected {expected_accuracy}, but got {accuracy}"

if __name__ == "__main__": 
    # Run the tests when the script is executed directly
    pytest.main()
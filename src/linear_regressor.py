"""
Linear Regression Module

This module provides functions to perform linear regression including
the computation of cost, gradients, and gradient descent optimization.
"""

import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def compute_cost(X, y, w, b, lambda_=0.0):
    """
    Compute the cost function for linear regression with L2 regularization.
    
    Args:
    X : ndarray, shape [m, n] 
        Training dataset of m examples and n features.
    y: ndarray, shape [m,]    
        Target values.
    w: ndarray, shape [n,]    
        Model parameters (weights).
    b: scalar  
        Model parameter (bias).
    lambda_ : scalar
        Regularization factor.
    
    Returns:
    total_cost: scalar 
        The total cost including the squared error cost and regularization cost.
    """
    m, n = X.shape
    if m == 0 or n == 0:
        raise ValueError("Training set is empty. Please provide a non-empty dataset.")
    
    unreg_cost = np.sum(((np.dot(X, w) + b) - y)**2) / (2 * m)
    reg_cost = (lambda_ / (2 * m)) * np.sum(w**2)
    
    return unreg_cost + reg_cost

def compute_gradients(X, y, w, b, lambda_=0.0):
    """
    Compute the gradients for linear regression with L2 regularization.
    
    Args:
    X : ndarray, shape [m, n] 
        Training dataset of m examples and n features.
    y: ndarray, shape [m,]    
        Target values.
    w: ndarray, shape [n,]    
        Model parameters (weights).
    b: scalar  
        Model parameter (bias).
    lambda_ : scalar
        Regularization factor.

    Returns:
    dj_dw: ndarray, shape (n,)
        Gradients of cost function with respect to weight parameters.
    dj_db: scalar
        Gradient of cost function with respect to bias.
    """
    m, n = X.shape
    if m == 0 or n == 0:
        raise ValueError("Training set is empty. Please provide a non-empty dataset.")

    errors = (np.dot(X, w) + b) - y
    dj_dw = (1 / m) * (X.T @ errors + lambda_ * w)
    dj_db = np.sum(errors) / m
    
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0.0):
    """
    Perform gradient descent to learn the parameters for linear regression.
    
    Args:
    X : ndarray, shape [m, n] 
        Training dataset of m examples and n features.
    y: ndarray, shape [m,]    
        Target values.
    w_in: ndarray, shape [n,]    
        Initial model parameters (weights).
    b_in: scalar  
        Initial model parameter (bias).
    alpha: float
        Learning rate.
    num_iters: scalar
        Number of iterations for gradient descent.
    lambda_ : scalar
        Regularization factor.

    Returns:
    w: ndarray, shape (n,)
        Optimized model parameters (weights).
    b: scalar
        Optimized model parameter (bias).
    J_history: ndarray
        Cost values over iterations.
    """
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradients(X, y, w, b, lambda_)
        b -= alpha * dj_db
        w -= alpha * dj_dw

        if i % (num_iters // 10) == 0:
            J_history.append(compute_cost(X, y, w, b, lambda_))
            print(f"Iteration: {i}, Weights: {w}, Bias: {b}, Cost: {J_history[-1]}")

    return w, b, np.array(J_history)

class LinearRegressor:
    def __init__(self, alpha=0.01, num_iters=1000, lambda_=0.0):
        self.alpha = alpha
        self.num_iters = num_iters
        self.lambda_ = lambda_
        self.w = None
        self.b = None
        self.J_history = None

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0
        J_hist = []

        for i in range(self.num_iters):
            dj_dw, dj_db = compute_gradients(X, y, self.w, self.b, self.lambda_)
            self.b -= self.alpha * dj_db
            self.w -= self.alpha * dj_dw
            J_hist.append(compute_cost(X, y, self.w, self.b, self.lambda_))

        self.J_history = np.array(J_hist)
        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b

def main():
    # Example usage
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = LinearRegressor(alpha=0.01, num_iters=1000, lambda_=0.1)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    # Compute Mean Squared Error
    mse = mean_squared_error(y_test, predictions)

    print("Mean Squared Error:", mse)
    print("Cost history:", regressor.J_history)

if __name__ == "__main__":
    main()

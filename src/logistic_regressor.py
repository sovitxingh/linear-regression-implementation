"""
Logistic Regression Module

This module provides functions to perform logistic regression including
the computation of cost, gradients, gradient descent optimization and various performance metrics.
"""

import numpy as np
import copy
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from utils import z_score_normalize

def sigmoid(z):
    """ 
    Compute the sigmoid of z. 
    
    Args: 
    z (array-like): Input array or scalar. 
    
    Returns: 
    array-like: Sigmoid of input. 
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
    
def compute_logistic_cost(X, y, w, b, lambda_=0.0):
    """
    Compute the cost function for logistic regression with L2 regularization.
    
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
        The total cost including the loss and regularization cost.
    """
    m, n = X.shape
    if m == 0 or n == 0:
        raise ValueError("Training set is empty. Please provide a non-empty dataset.")
    
    # Small epsilon value to avoid log(0) 
    epsilon = 1e-15
    f_wb = sigmoid(np.dot(X, w) + b)
    f_wb = np.clip(f_wb, epsilon, 1 - epsilon) # Ensure values are within (epsilon, 1-epsilon)
    
    loss = - (1 / m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    regularization_cost = (lambda_ / (2*m)) * np.sum(w**2)
    total_cost = loss + regularization_cost
    return total_cost

def compute_logistic_gradients(X, y, w, b, lambda_=0.0):
    """
    Compute the gradients for logistic regression with L2 regularization.
    
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

    errors = sigmoid(np.dot(X, w) + b) - y
    dj_dw = (1 / m) * (X.T @ errors + lambda_ * w)
    dj_db = np.sum(errors) / m
    
    return dj_dw, dj_db

def logistic_gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0.0):
    """
    Perform gradient descent to learn the parameters for logistic regression.
    
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
        dj_dw, dj_db = compute_logistic_gradients(X, y, w, b, lambda_)
        b -= alpha * dj_db
        w -= alpha * dj_dw

        if i % (num_iters // 10) == 0:
            J_history.append(compute_logistic_cost(X, y, w, b, lambda_))
            print(f"Iteration: {i}, Weights: {w}, Bias: {b}, Cost: {J_history[-1]}")

    return w, b, np.array(J_history)
    
def evaluate_model_performance(X, y, w, b):
    """
    Compute and display the accuracy of the model on the training data.

    Args:
    X : ndarray, shape (m, n) - Feature matrix
    y : ndarray, shape (m, ) - Actual labels
    w : ndarray, shape (n, ) - Model weights
    b : scalar - Model bias

    Returns:
    accuracy : float
        Model's accuracy on the provided test dataset.
    """
    predictions = predict(X, w, b)
    accuracy = np.mean(predictions == y)

    # True labels
    true_labels = y

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:\n", conf_matrix)

    # Precision, Recall, and F1 Score
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    # Display metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return accuracy

def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters.
    
    Args:
    X : ndarray, shape (m, n)
        Data of m examples and n features.
    w : ndarray, shape (n,)
        Model parameters (weights).
    b : float
        Model parameter (bias).

    Returns:
    y_pred : ndarray, shape (m,)
        Predictions on data.
    """
    m, n = X.shape
    if m==0 or n==0:
        return np.array([])
        
    y_pred = np.zeros(m)

    probabilities = sigmoid(np.dot(X, w) + b)
    y_pred = (probabilities >= 0.5).astype(int)
    return y_pred

def main():
    # Dummy dataset
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0,0,1,1])
    
    X_scaled = z_score_normalize(X)
    n = X_scaled.shape[1]
    
    initial_w = np.zeros(n)
    initial_b = 0
    alpha = 0.001
    num_iters = 1000
    lambda_ = 0.1

    w, b, J_history = logistic_gradient_descent(X_scaled, y, initial_w, initial_b, alpha, num_iters, lambda_)
    evaluate_model_performance(X_scaled, y, w, b)

if __name__ == "__main__":
    main()

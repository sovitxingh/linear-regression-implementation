import numpy as np

def z_score_normalize(X):
    """
    Args:
    X : ndarray, shape [m, n] 
        Training Dataset of m examples and n features.

    Returns:
    X_norm : ndarray, shape [m, n] 
        Normalized training Dataset of m examples and n features.
    """

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    X_norm = (X - mu) / (sigma + epsilon)
    return X_norm


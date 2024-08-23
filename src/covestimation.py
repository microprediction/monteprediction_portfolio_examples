
# Placeholder
import numpy as np

def cov_estimation(X):
    """
    Compute the covariance matrix of the dataset X.

    Parameters:
        X (ndarray): A 2D numpy array where each row is an observation and each column is a variable.

    Returns:
        ndarray: The covariance matrix of X.
    """
    # Ensure X is a numpy array
    X = np.array(X)

    # Compute the mean of each column
    means = np.mean(X, axis=0)

    # Center the data by subtracting the mean
    centered_X = X - means

    # Compute the covariance matrix
    cov_matrix = np.dot(centered_X.T, centered_X) / (X.shape[0] - 1)

    return cov_matrix
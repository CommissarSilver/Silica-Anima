import numpy as np
import math


def create_dataset(
    number_of_samples: int = 1000,
    number_of_features: int = 2,
    noise: float = 0.1,
):
    """
    Create a dataset with two classes for binary classification.

    Args:
        number_of_samples (int, optional): number of samples to create. Defaults to 1000.
        number_of_features (int, optional): number of features for each datapoint. Defaults to 2.
        noise (float, optional): noise to add to the linearity of the funciton. Defaults to 0.1.
    """
    X = np.random.uniform(-1, 1, size=(number_of_samples, number_of_features))
    # add some noise to the linearity of the function
    # X[:, 1] = X[:, 1] + np.random.normal(0, noise, size=(number_of_samples,))
    # create a linearity
    Y = np.zeros(number_of_samples)
    if np.sum(X[:, 1] > 0) > np.sum(X[:, 1] < 0):
        Y[X[:, 1] > 0] = 1
    else:
        Y[X[:, 1] < 0] = 1

    return X, Y


def sigmoid(x: float):
    """
    Sigmoid function.

    Args:
        x (float): function input

    Returns:
        float: application of the sigmoid function to x
    """
    return 1 / (1 + math.exp(-x))


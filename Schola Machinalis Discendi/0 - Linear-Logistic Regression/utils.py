import numpy as np
import math
import matplotlib.pyplot as plt


def binary_dataset(num_samples: int = 100):
    """
    Generates a binary dataset of size num_samples.
    """
    # Generate random points in the unit square
    points = np.random.rand(num_samples, 2)
    # Assign a label to each point
    labels = np.array([1 if (x[0] > x[1]) else -1 for x in points])
    return points, labels


X, Y = binary_dataset(200)

weights = np.zeros(2)
bias = np.zeros(1)

learning_rate = 0.001
losses = []
epochs = 10000


def sigmoid(z: np.array):
    return 1 / (1 + np.exp(-z))


j = sigmoid(0)

for epoch in range(epochs):
    y_hat = sigmoid(np.dot(X, weights) + bias)
    error = y_hat - Y
    
    delta_w_1 = np.dot(X[:, 0], error) / len(X)
    delta_w_2 = np.dot(X[:, 1], error) / len(X)
    delta_b = np.sum(error) / len(X)

    weights[0] -= learning_rate * delta_w_1
    weights[1] -= learning_rate * delta_w_2
    bias -= learning_rate * delta_b

    loss = np.sum(-1 * ((Y * np.log(y_hat)) + ((1 - Y) * np.log(1 - y_hat)))) / len(X)
    losses.append(loss)

plt.plot(losses)
plt.show()

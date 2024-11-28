import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


inputs = np.array([0, 0, 1])  # 3 inputs  — 3 neurons
weights = np.array([10, 0, -5])  # 3 weights — 3 synapses
outputs = sigmoid(np.dot(inputs, weights))
print(outputs)

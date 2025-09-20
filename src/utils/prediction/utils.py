import numpy as np

def d_logistic(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2

def d_relu(x):
    return x > 0

def relu(x):
    return np.maximum(x, 0)

def logistic(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
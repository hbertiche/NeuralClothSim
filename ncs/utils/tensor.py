import numpy as np
import tensorflow as tf
from scipy.special import binom


def tf_shape(tensor):
    return [(size or -1) for size in tensor.get_shape()]


@tf.function
def compute_nth_derivative(X, n, dt):
    for i in range(n):
        X = (X[:, 1:] - X[:, :-1]) / dt
    return X

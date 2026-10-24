"""Parameter updates implemented directly with NumPy."""
import numpy as np


class Adam:
    """Track exponential averages of gradients and squared gradients per parameter."""

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.steps = 0
        self.first_moments = None
        self.second_moments = None

    def update(self, parameters, gradients, learning_rate):
        if self.first_moments is None:
            self.first_moments = [np.zeros_like(p) for p in parameters]
            self.second_moments = [np.zeros_like(p) for p in parameters]
        self.steps += 1
        for parameter, gradient, first, second in zip(
                parameters, gradients, self.first_moments, self.second_moments):
            first *= self.beta1
            first += (1 - self.beta1) * gradient
            second *= self.beta2
            second += (1 - self.beta2) * gradient ** 2
            corrected_first = first / (1 - self.beta1 ** self.steps)
            corrected_second = second / (1 - self.beta2 ** self.steps)
            parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + self.epsilon)

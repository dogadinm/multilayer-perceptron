"""Monitor validation loss and retain independent copies of the best
parameters.
"""

import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.reference_loss = float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.best_parameters = None

    def observe(self, model, validation_loss, epoch):
        if not np.isfinite(validation_loss):
            raise ValueError('validation loss must be finite')
        # Always retain the actual lowest loss, even for improvements below
        # min_delta.
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.best_epoch = epoch
            self.best_parameters = (
                [w.copy() for w in model.weights],
                [b.copy() for b in model.biases],
            )
        if validation_loss < self.reference_loss - self.min_delta:
            self.reference_loss = validation_loss
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience

    def restore(self, model):
        if self.best_parameters is None:
            raise ValueError('no best parameters have been recorded')
        weights, biases = self.best_parameters
        model.weights = [w.copy() for w in weights]
        model.biases = [b.copy() for b in biases]

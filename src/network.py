"""Dense neural network: explicit forward, backward and SGD."""
from pathlib import Path
import numpy as np


def linear(inputs, weights, bias):
    """Weighted sum: each column of weights corresponds to one neuron."""
    return inputs @ weights + bias


def relu(z):
    """Hidden-layer activation: keep positive values and replace the rest with zero."""
    return np.maximum(0, z)


def relu_derivative(activations):
    """Use cached ReLU outputs: the derivative is 1 for outputs > 0, otherwise 0."""
    return (activations > 0).astype(float)


def softmax(logits):
    """Convert output scores to probabilities that sum to 1 for each row."""
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted_logits)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_from_logits(y, logits):
    """Compute mean cross-entropy from logits for numerical stability."""
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted_logits).sum(axis=1))
    correct_class_logits = shifted_logits[np.arange(len(y)), y]
    return float(np.mean(log_normalizer - correct_class_logits))


def binary_cross_entropy(y, probabilities):
    p = np.clip(probabilities[:, 1], 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def accuracy_score(y, probabilities):
    predicted_classes = probabilities.argmax(axis=1)
    return float(np.mean(predicted_classes == y))


def softmax_cross_entropy_gradient(y, probabilities):
    """Gradient of mean loss with respect to logits: (P - one_hot(y)) / batch_size."""
    gradient = probabilities.copy()
    gradient[np.arange(len(y)), y] -= 1
    return gradient / len(y)


class Network:
    def __init__(self, layers=(30, 24, 24, 2), seed=42):
        if len(layers) < 3 or layers[0] != 30 or layers[-1] != 2 or any(n <= 0 for n in layers):
            raise ValueError('architecture must be 30 -> positive hidden sizes -> 2')
        self.layers = list(layers)
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(0, np.sqrt(2 / a), (a, b))
                        for a, b in zip(layers[:-1], layers[1:])]
        self.biases = [np.zeros((1, b)) for b in layers[1:]]

    def forward(self, x):
        """Input -> hidden layers with ReLU -> output logits -> softmax.

        Cache activations for the backward pass.
        """
        activations = [x]
        for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = linear(activations[-1], weights, bias)
            hidden_output = relu(z)
            activations.append(hidden_output)

        logits = linear(activations[-1], self.weights[-1], self.biases[-1])
        probabilities = softmax(logits)
        return activations, logits, probabilities

    def backward(self, activations, probabilities, y):
        """Propagate the output loss gradient backward through each layer."""
        output_gradient = softmax_cross_entropy_gradient(y, probabilities)
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.weights)

        for layer in reversed(range(len(self.weights))):
            layer_inputs = activations[layer]
            weight_gradients[layer] = layer_inputs.T @ output_gradient
            bias_gradients[layer] = output_gradient.sum(axis=0, keepdims=True)

            if layer > 0:
                input_gradient = output_gradient @ self.weights[layer].T
                output_gradient = input_gradient * relu_derivative(layer_inputs)

        return weight_gradients, bias_gradients

    def gradients(self, x, y):
        """Compute predictions first, then compute loss gradients."""
        activations, _, probabilities = self.forward(x)
        return self.backward(activations, probabilities, y)

    def update_parameters(self, weight_gradients, bias_gradients, learning_rate):
        """Gradient descent step: move parameters against the gradient."""
        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * weight_gradients[layer]
            self.biases[layer] -= learning_rate * bias_gradients[layer]

    def step(self, x, y, learning_rate):
        """Process one batch: forward + backward, then update parameters."""
        weight_gradients, bias_gradients = self.gradients(x, y)
        self.update_parameters(weight_gradients, bias_gradients, learning_rate)

    def predict_probabilities(self, x):
        """Run a forward pass for prediction without updating weights."""
        _, _, probabilities = self.forward(x)
        return probabilities

    def metrics(self, x, y):
        _, logits, probabilities = self.forward(x)
        loss = cross_entropy_from_logits(y, logits)
        accuracy = accuracy_score(y, probabilities)
        return loss, accuracy

    def save(self, path, mean, scale):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = dict(version=np.array(1), layers=np.array(self.layers), mean=mean,
                       scale=scale, classes=np.array(['B', 'M']))
        payload.update({f'w{i}': w for i, w in enumerate(self.weights)})
        payload.update({f'b{i}': b for i, b in enumerate(self.biases)})
        with open(path, 'wb') as stream:
            np.savez_compressed(stream, **payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            if archive['version'].item() != 1 or list(archive['classes']) != ['B', 'M']:
                raise ValueError('unsupported model format')
            model = cls(archive['layers'].tolist())
            for i, (a, b) in enumerate(zip(model.layers[:-1], model.layers[1:])):
                w, bias = archive[f'w{i}'], archive[f'b{i}']
                if w.shape != (a, b) or bias.shape != (1, b):
                    raise ValueError('invalid parameter shape')
                if not np.all(np.isfinite(w)) or not np.all(np.isfinite(bias)):
                    raise ValueError('non-finite model parameters')
                model.weights[i], model.biases[i] = w, bias
            mean, scale = archive['mean'], archive['scale']
            if mean.shape != (30,) or scale.shape != (30,) or not np.all(scale > 0):
                raise ValueError('invalid scaler')
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
                raise ValueError('non-finite scaler')
        return model, mean, scale

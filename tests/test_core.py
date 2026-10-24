import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.data import fit_scaler, split_indices, load_csv, write_csv
from src.network import Network, softmax, binary_cross_entropy


class CoreTests(unittest.TestCase):
    def test_gradients_against_finite_differences(self):
        rng = np.random.default_rng(13)
        x, y = rng.normal(size=(5, 30)), np.array([0, 1, 1, 0, 1])
        net = Network([30, 3, 3, 2], seed=7)
        dw, db = net.compute_gradients(x, y)
        for parameters, gradients in [(net.weights, dw), (net.biases, db)]:
            for parameter, gradient in zip(parameters, gradients):
                for index in np.ndindex(parameter.shape):
                    original = parameter[index]
                    parameter[index] = original + 1e-6
                    plus = net.evaluate(x, y)[0]
                    parameter[index] = original - 1e-6
                    minus = net.evaluate(x, y)[0]
                    parameter[index] = original
                    self.assertAlmostEqual(
                        (plus - minus) / 2e-6, gradient[index], places=6
                    )

    def test_save_load_preserves_predictions(self):
        x = np.random.default_rng(4).normal(size=(10, 30))
        mean, scale = fit_scaler(x)
        net = Network()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.npz'
            net.save(path, mean, scale)
            restored, m, s = Network.load(path)
            np.testing.assert_array_equal(
                net.forward((x - mean) / scale)[2],
                restored.forward((x - m) / s)[2],
            )

    def test_softmax_and_loss(self):
        p = softmax(np.array([[10000.0, 10001.0], [-10000.0, -10001.0]]))
        np.testing.assert_allclose(p.sum(axis=1), 1)
        self.assertTrue(np.isfinite(binary_cross_entropy(np.array([1, 0]), p)))
        net = Network()
        x, y = np.ones((2, 30)), np.array([0, 1])
        self.assertAlmostEqual(
            net.evaluate(x, y)[0], binary_cross_entropy(y, net.forward(x)[2])
        )

    def test_split(self):
        y = np.array([0] * 40 + [1] * 20)
        train, valid = split_indices(y, 0.8, 42)
        self.assertFalse(set(train) & set(valid))
        self.assertEqual(len(set(train) | set(valid)), len(y))
        self.assertEqual(np.sum(y[train]), 16)
        np.testing.assert_array_equal(train, split_indices(y, 0.8, 42)[0])

    def test_scaler_and_csv(self):
        x = np.ones((3, 30))
        mean, scale = fit_scaler(x)
        np.testing.assert_array_equal((x - mean) / scale, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            write_csv(
                path,
                [
                    ['id', 'diagnosis', *['feature'] * 30],
                    ['1', 'M', *['1'] * 30],
                ],
            )
            _, features, y = load_csv(path)
            self.assertEqual(features.shape, (1, 30))
            self.assertEqual(y[0], 1)
            write_csv(path, [['1', 'M', *['nan'] * 30]])
            with self.assertRaises(ValueError):
                load_csv(path)

    def test_learning_reduces_loss(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(100, 30))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        net = Network(seed=42)
        initial = net.evaluate(x, y)[0]
        for _ in range(300):
            net.train_batch(x, y, 0.05)
        loss, accuracy = net.evaluate(x, y)
        self.assertLess(loss, initial * 0.3)
        self.assertGreater(accuracy, 0.95)


if __name__ == '__main__':
    unittest.main()

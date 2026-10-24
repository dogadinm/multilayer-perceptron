"""Check optimizer math, early stopping, metrics, and the bonus CLI."""

import contextlib
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.optimizers import Adam
from src.early_stopping import EarlyStopping
from src.history import save_history, load_history
from src.metrics import classification_metrics
from src.network import Network
from src.data import write_csv, load_csv, standardize
from src.metrics import evaluate_model
from train import train_model, create_parser

ROOT = Path(__file__).resolve().parents[1]


class BonusTests(unittest.TestCase):
    def test_adam_bias_correction_and_persistent_moments(self):
        weights, bias = np.array([1.0]), np.array([2.0])
        optimizer = Adam()
        optimizer.update(
            [weights, bias], [np.array([2.0]), np.array([-3.0])], 0.1
        )
        self.assertAlmostEqual(weights[0], 1 - 0.1 * 2 / (2 + 1e-8))
        self.assertAlmostEqual(bias[0], 2 + 0.1 * 3 / (3 + 1e-8))
        previous = weights[0]
        optimizer.update(
            [weights, bias], [np.array([-1.0]), np.array([0.0])], 0.1
        )
        # For gradients 2 then -1: m2=.08 and v2=.004996 before correction.
        expected_update = (
            0.1
            * (0.08 / (1 - 0.9**2))
            / (math.sqrt(0.004996 / (1 - 0.999**2)) + 1e-8)
        )
        self.assertAlmostEqual(weights[0], previous - expected_update)
        self.assertEqual(optimizer.steps, 2)

    def test_early_stopping_patience_min_delta_and_snapshot(self):
        model = Network()
        monitor = EarlyStopping(patience=2, min_delta=0.1)
        self.assertFalse(monitor.observe(model, 1.0, 1))
        model.weights[0].fill(2.0)
        self.assertFalse(monitor.observe(model, 0.8, 2))
        model.weights[0].fill(3.0)
        self.assertFalse(monitor.observe(model, 0.75, 3))
        model.weights[0].fill(4.0)
        self.assertTrue(monitor.observe(model, 0.76, 4))
        monitor.restore(model)
        self.assertEqual(monitor.best_epoch, 3)
        np.testing.assert_array_equal(model.weights[0], 3.0)
        model.weights[0].fill(9.0)
        monitor.restore(model)
        np.testing.assert_array_equal(model.weights[0], 3.0)

    def test_train_loop_restores_best_epoch(self):
        model = Network()
        args = create_parser().parse_args(
            [
                '--train',
                'a',
                '--valid',
                'b',
                '--early-stopping',
                '--patience',
                '2',
                '--epochs',
                '10',
            ]
        )
        x, y = np.ones((2, 30)), np.array([0, 1])

        def epoch_update(*unused):
            model.weights[0] += 1

        metrics = classification_metrics(y, np.array([[1.0, 0.0], [0.0, 1.0]]))
        evaluations = []
        for loss in [0.5, 0.3, 0.4, 0.6]:
            evaluations.extend(
                [dict(loss=loss, **metrics), dict(loss=loss, **metrics)]
            )
        original = model.weights[0].copy()
        with (
            patch('train.train_one_epoch', side_effect=epoch_update),
            patch('train.evaluate_model', side_effect=evaluations),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            history = train_model(model, x, y, x, y, args)
        self.assertEqual(len(history), 4)
        np.testing.assert_allclose(model.weights[0], original + 2)

    def test_metrics_confusion_and_zero_division(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        predicted = np.array([0, 0, 1, 0, 1, 1])
        metrics = classification_metrics(y, np.eye(2)[predicted])
        self.assertEqual(
            [metrics[k] for k in ('tn', 'fp', 'fn', 'tp')], [2, 1, 1, 2]
        )
        for name in ('accuracy', 'precision', 'recall', 'f1'):
            self.assertAlmostEqual(metrics[name], 2 / 3)
        metrics = classification_metrics(
            np.array([0, 0]), np.array([[1.0, 0.0], [1.0, 0.0]])
        )
        self.assertEqual(
            [metrics[k] for k in ('precision', 'recall', 'f1')],
            [0.0, 0.0, 0.0],
        )

    def test_history_round_trip_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.json'
            metrics = dict(
                loss=0.3, accuracy=0.8, precision=0.7, recall=0.9, f1=0.78
            )
            records = [dict(epoch=1, train=metrics, validation=metrics)]
            save_history(
                path, records, {'optimizer': 'adam'}, {'saved_epoch': 1}
            )
            self.assertEqual(load_history(path)['epochs'], records)
            path.write_text('{"version": 1, "epochs": []}')
            with self.assertRaises(ValueError):
                load_history(path)

    def test_bonus_cli_pipeline_and_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(9)
            x = rng.normal(size=(80, 30))
            labels = np.where(x[:, 0] > 0, 'M', 'B')
            write_csv(
                root / 'input.csv',
                [
                    [str(i), label, *row]
                    for i, (row, label) in enumerate(zip(x, labels))
                ],
            )

            def run(*command):
                return subprocess.run(
                    [sys.executable, *command],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            run(
                'split.py',
                str(root / 'input.csv'),
                '--train',
                str(root / 'train.csv'),
                '--valid',
                str(root / 'valid.csv'),
            )
            for name in ('sgd', 'adam'):
                run(
                    'train.py',
                    '--train',
                    str(root / 'train.csv'),
                    '--valid',
                    str(root / 'valid.csv'),
                    '--optimizer',
                    name,
                    '--epochs',
                    '5',
                    '--history',
                    str(root / f'{name}.json'),
                    '--model',
                    str(root / f'{name}.npz'),
                    '--plots',
                    str(root / name),
                    '--no-show',
                    '--early-stopping',
                    '--patience',
                    '2',
                )
                payload = load_history(root / f'{name}.json')
                self.assertEqual(payload['config']['optimizer'], name)
                self.assertGreaterEqual(len(payload['epochs']), 1)
                loaded, mean, scale = Network.load(root / f'{name}.npz')
                _, valid_x, valid_y = load_csv(root / 'valid.csv')
                actual_metrics = evaluate_model(
                    loaded, standardize(valid_x, mean, scale), valid_y
                )
                self.assertEqual(
                    actual_metrics, payload['summary']['validation']
                )
                result = run(
                    'predict.py',
                    '--model',
                    str(root / f'{name}.npz'),
                    '--data',
                    str(root / 'valid.csv'),
                    '--output',
                    str(root / f'{name}.csv'),
                )
                self.assertIn('Confusion matrix', result.stdout)
                self.assertIn('Recall:', result.stdout)
            run(
                'compare.py',
                str(root / 'sgd.json'),
                str(root / 'adam.json'),
                '--output',
                str(root / 'comparison'),
                '--no-show',
            )
            for metric in ('loss', 'accuracy', 'precision', 'recall', 'f1'):
                self.assertGreater(
                    (root / 'comparison' / f'{metric}.png').stat().st_size,
                    1000,
                )


if __name__ == '__main__':
    unittest.main()

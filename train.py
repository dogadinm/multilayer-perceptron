"""Train a configurable multilayer perceptron using mini-batch SGD or Adam."""

import argparse
from pathlib import Path
import numpy as np
from src.data import load_csv, fit_scaler, standardize
from src.network import Network
from src.visualization import plot_curves
from src.optimizers import Adam
from src.early_stopping import EarlyStopping
from src.history import save_history
from src.metrics import evaluate_model, print_classification_metrics


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--valid', required=True)
    parser.add_argument(
        '--layers', '--layer', type=int, nargs='+', default=[24, 24]
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=16)
    parser.add_argument(
        '--learning-rate', '--learning_rate', type=float, default=0.01
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default='models/model.npz')
    parser.add_argument('--plots', default='plots')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='restore the lowest-validation-loss weights',
    )
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument(
        '--min-delta',
        type=float,
        default=0.0,
        help='minimum improvement to reset patience',
    )
    parser.add_argument(
        '--history',
        default='history/run.json',
        help='save all epoch metrics and run settings',
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='save graphs without opening windows',
    )
    return parser


def validate_training_options(args):
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        raise ValueError(
            'epochs, batch size and learning rate must be positive and finite'
        )
    if (
        args.patience <= 0
        or not np.isfinite(args.min_delta)
        or args.min_delta < 0
    ):
        raise ValueError(
            'patience must be positive and min delta must be '
            'nonnegative and finite'
        )
    outputs = [Path(p).resolve() for p in (args.model, args.history)]
    inputs = {Path(p).resolve() for p in (args.train, args.valid)}
    plot_paths = {
        Path(args.plots).resolve() / f'{metric}.png'
        for metric in ('loss', 'accuracy')
    }
    if (
        len(set(outputs)) != 2
        or set(outputs) & (inputs | plot_paths)
        or inputs & plot_paths
    ):
        raise ValueError(
            'model, history, plots and input paths must be distinct'
        )


def prepare_training_data(train_path, valid_path):
    """Load -> validate partitions -> fit the scaler on training data ->
    standardize.
    """
    train_rows, x_train, y_train = load_csv(train_path)
    valid_rows, x_valid, y_valid = load_csv(valid_path)
    if set(train_rows[:, 0]) & set(valid_rows[:, 0]):
        raise ValueError('train and validation contain overlapping IDs')
    if len(np.unique(y_train)) != 2:
        raise ValueError('training data must contain both B and M')

    mean, scale = fit_scaler(x_train)
    x_train = standardize(x_train, mean, scale)
    x_valid = standardize(x_valid, mean, scale)
    return x_train, y_train, x_valid, y_valid, mean, scale


def shuffled_batches(x, y, batch_size, rng):
    """Shuffle indices each epoch and yield batches, including the final
    partial batch.
    """
    order = rng.permutation(len(y))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        yield x[indices], y[indices]


def train_one_epoch(
    model, x, y, batch_size, learning_rate, rng, optimizer=None
):
    for x_batch, y_batch in shuffled_batches(x, y, batch_size, rng):
        model.train_batch(x_batch, y_batch, learning_rate, optimizer)


def print_epoch_metrics(epoch, epochs, record):
    train = record['train']
    valid = record['validation']
    print(
        f"epoch {epoch:03d}/{epochs} - loss: {train['loss']:.6f}"
        f" - accuracy: {train['accuracy']:.4f}"
        f" - val_loss: {valid['loss']:.6f}"
        f" - val_accuracy: {valid['accuracy']:.4f}"
    )
    for name, metrics in [('train', train), ('validation', valid)]:
        print(
            f"  {name}: precision: {metrics['precision']:.4f}"
            f" - recall: {metrics['recall']:.4f}"
            f" - f1: {metrics['f1']:.4f}"
        )


def train_model(model, x_train, y_train, x_valid, y_valid, args):
    """Update parameters on training data; monitor validation without taking
    gradients.
    """
    rng = np.random.default_rng(args.seed)
    optimizer = Adam() if args.optimizer == 'adam' else None
    stopping = (
        EarlyStopping(args.patience, args.min_delta)
        if args.early_stopping
        else None
    )
    history = []
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(
                model,
                x_train,
                y_train,
                args.batch_size,
                args.learning_rate,
                rng,
                optimizer,
            )
            record = dict(
                epoch=epoch,
                train=evaluate_model(model, x_train, y_train),
                validation=evaluate_model(model, x_valid, y_valid),
            )
            history.append(record)
            print_epoch_metrics(epoch, args.epochs, record)
            if stopping and stopping.observe(
                model, record['validation']['loss'], epoch
            ):
                print(
                    f'Early stopping at epoch {epoch}: '
                    f'no sufficient improvement for {args.patience} epochs'
                )
                break
    if stopping:
        stopping.restore(model)
        print(
            f'Restored best weights from epoch {stopping.best_epoch} '
            f'(val_loss: {stopping.best_loss:.6f})'
        )
    return history


def main():
    parser = create_parser()
    args = parser.parse_args()
    try:
        validate_training_options(args)
        x_train, y_train, x_valid, y_valid, mean, scale = (
            prepare_training_data(args.train, args.valid)
        )
        print(
            f'x_train shape: {x_train.shape}\nx_valid shape: {x_valid.shape}'
        )

        model = Network([30, *args.layers, 2], args.seed)
        history = train_model(model, x_train, y_train, x_valid, y_valid, args)

        saved_epoch = (
            min(history, key=lambda record: record['validation']['loss'])[
                'epoch'
            ]
            if args.early_stopping
            else history[-1]['epoch']
        )
        summary = dict(
            epochs_completed=len(history),
            saved_epoch=saved_epoch,
            stopped_early=len(history) < args.epochs,
            train=evaluate_model(model, x_train, y_train),
            validation=evaluate_model(model, x_valid, y_valid),
        )
        config = dict(vars(args), layers=model.layers, positive_class='M')
        model.save(args.model, mean, scale)
        save_history(args.history, history, config, summary)
        print(f'History saved: {args.history}')
        print('Saved model validation metrics:')
        print_classification_metrics(summary['validation'])
        print(f'Model saved: {args.model}')
        plot_curves(history, args.plots, not args.no_show)
        print(
            f'Graphs saved: {args.plots}/loss.png, {args.plots}/accuracy.png'
        )
    except (ValueError, OSError, FloatingPointError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

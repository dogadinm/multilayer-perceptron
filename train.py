"""Train a configurable multilayer perceptron using mini-batch SGD."""
import argparse
import numpy as np
from src.data import load_csv, fit_scaler, standardize
from src.network import Network
from src.visualization import plot_curves


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--valid', required=True)
    parser.add_argument('--layers', '--layer', type=int, nargs='+', default=[24, 24])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=16)
    parser.add_argument('--learning-rate', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default='models/model.npz')
    parser.add_argument('--plots', default='plots')
    parser.add_argument('--no-show', action='store_true', help='save graphs without opening windows')
    return parser


def validate_training_options(args):
    if args.epochs <= 0 or args.batch_size <= 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('epochs, batch size and learning rate must be positive and finite')


def prepare_training_data(train_path, valid_path):
    """Load -> validate partitions -> fit the scaler on training data -> standardize."""
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
    """Shuffle indices each epoch and yield batches, including the final partial batch."""
    order = rng.permutation(len(y))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        yield x[indices], y[indices]


def train_one_epoch(model, x, y, batch_size, learning_rate, rng):
    for x_batch, y_batch in shuffled_batches(x, y, batch_size, rng):
        model.step(x_batch, y_batch, learning_rate)


def print_epoch_metrics(epoch, epochs, metrics):
    loss, accuracy, val_loss, val_accuracy = metrics
    print(f'epoch {epoch:03d}/{epochs} - loss: {loss:.6f} - accuracy: {accuracy:.4f}'
          f' - val_loss: {val_loss:.6f} - val_accuracy: {val_accuracy:.4f}')


def train_model(model, x_train, y_train, x_valid, y_valid, args):
    """Train on the training partition; evaluate both partitions after each epoch."""
    rng = np.random.default_rng(args.seed)
    history = []
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, x_train, y_train, args.batch_size, args.learning_rate, rng)
            train_metrics = model.metrics(x_train, y_train)
            valid_metrics = model.metrics(x_valid, y_valid)
            epoch_metrics = (*train_metrics, *valid_metrics)
            history.append(epoch_metrics)
            print_epoch_metrics(epoch, args.epochs, epoch_metrics)
    return history


def main():
    parser = create_parser()
    args = parser.parse_args()
    try:
        validate_training_options(args)
        x_train, y_train, x_valid, y_valid, mean, scale = prepare_training_data(args.train, args.valid)
        print(f'x_train shape: {x_train.shape}\nx_valid shape: {x_valid.shape}')

        model = Network([30, *args.layers, 2], args.seed)
        history = train_model(model, x_train, y_train, x_valid, y_valid, args)

        model.save(args.model, mean, scale)
        print(f'Model saved: {args.model}')
        plot_curves(history, args.plots, not args.no_show)
        print(f'Graphs saved: {args.plots}/loss.png, {args.plots}/accuracy.png')
    except (ValueError, OSError, FloatingPointError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

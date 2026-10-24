"""Load a saved model, predict labeled CSV rows and evaluate BCE."""
import argparse
from pathlib import Path
import numpy as np
from src.data import load_csv, write_csv, standardize
from src.network import Network, binary_cross_entropy
from src.metrics import classification_metrics, print_classification_metrics


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/model.npz')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='data/predictions.csv')
    return parser


def validate_output_path(data_path, model_path, output_path):
    if Path(output_path).resolve() in (Path(data_path).resolve(), Path(model_path).resolve()):
        raise ValueError('output must not overwrite the dataset or model')


def predict_dataset(model_path, x):
    """Load the model -> apply saved scaling -> compute probabilities with forward."""
    model, mean, scale = Network.load(model_path)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        normalized_x = standardize(x, mean, scale)
        return model.predict_probabilities(normalized_x)


def save_predictions(path, rows, probabilities):
    labels = np.array(['B', 'M'])[probabilities.argmax(axis=1)]
    result = [['id', 'actual', 'predicted', 'probability_B', 'probability_M']]
    for row, label, probability in zip(rows, labels, probabilities):
        result.append([row[0], row[1], label,
                       f'{probability[0]:.10f}', f'{probability[1]:.10f}'])
    write_csv(path, result)


def main():
    parser = create_parser()
    args = parser.parse_args()
    try:
        validate_output_path(args.data, args.model, args.output)
        rows, x, y = load_csv(args.data)
        probabilities = predict_dataset(args.model, x)
        save_predictions(args.output, rows, probabilities)
        print(f'Binary cross-entropy: {binary_cross_entropy(y, probabilities):.6f}')
        print_classification_metrics(classification_metrics(y, probabilities))
        print(f'Predictions: {args.output}')
    except (ValueError, OSError, KeyError, FloatingPointError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

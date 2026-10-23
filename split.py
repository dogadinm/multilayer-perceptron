"""Split the labeled dataset into reproducible stratified partitions."""
import argparse
from pathlib import Path
from src.data import load_csv, split_indices, write_csv


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--ratio', type=float, default=0.8, help='training fraction (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train', default='data/train.csv')
    parser.add_argument('--valid', default='data/validation.csv')
    return parser


def split_dataset(dataset_path, train_path, valid_path, ratio, seed):
    """Validate paths -> load data -> select indices -> write both partitions."""
    paths = [Path(p).resolve() for p in (dataset_path, train_path, valid_path)]
    if len(set(paths)) != 3:
        raise ValueError('input, train and validation paths must be different')
    rows, _, y = load_csv(dataset_path)
    train_indices, valid_indices = split_indices(y, ratio, seed)
    write_csv(train_path, rows[train_indices])
    write_csv(valid_path, rows[valid_indices])
    return len(train_indices), len(valid_indices)


def main():
    parser = create_parser()
    args = parser.parse_args()
    try:
        train_size, valid_size = split_dataset(args.dataset, args.train, args.valid, args.ratio, args.seed)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f'Train: {train_size} rows -> {args.train}')
    print(f'Validation: {valid_size} rows -> {args.valid}')


if __name__ == '__main__':
    main()

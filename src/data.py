"""CSV handling and preprocessing, without machine-learning libraries."""
import csv
from pathlib import Path

import numpy as np


def validate_data_row(row, path, line):
    """Validate one row before converting the full dataset to arrays."""
    if len(row) != 32:
        raise ValueError(f'{path}:{line}: expected 32 columns, got {len(row)}')
    if row[1] not in ('B', 'M'):
        raise ValueError(f'{path}:{line}: diagnosis must be B or M')
    try:
        features = np.asarray(row[2:], dtype=float)
    except ValueError as error:
        raise ValueError(f'{path}:{line}: features must be numeric') from error
    if not np.all(np.isfinite(features)):
        raise ValueError(f'{path}:{line}: missing or non-finite features')


def load_csv(path):
    """Read Wisconsin format: ID, diagnosis (B/M), 30 numeric features.

    An optional header and one empty trailing CSV column are accepted.
    """
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or not any(cell.strip() for cell in row):
                continue
            row = [cell.strip() for cell in row]
            if row[-1] == '':
                row = row[:-1]
            if not rows and len(row) > 1 and row[1].lower() in ('diagnosis', 'label'):
                continue
            validate_data_row(row, path, line)
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no data rows')
    raw = np.asarray(rows)
    return raw, raw[:, 2:].astype(float), (raw[:, 1] == 'M').astype(int)


def split_indices(y, ratio, seed):
    if not 0 < ratio < 1:
        raise ValueError('ratio must be between 0 and 1')
    rng = np.random.default_rng(seed)
    train, valid = [], []
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == label))
        if len(indices) < 2:
            raise ValueError('each class needs at least two samples for stratified splitting')
        cut = min(len(indices) - 1, max(1, int(len(indices) * ratio)))
        train.extend(indices[:cut])
        valid.extend(indices[cut:])
    return rng.permutation(train), rng.permutation(valid)


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        csv.writer(stream).writerows(rows)


def fit_scaler(x):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def standardize(x, mean, scale):
    """Apply fitted scaling parameters during both training and prediction."""
    return (x - mean) / scale

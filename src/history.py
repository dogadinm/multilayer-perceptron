"""Persist epoch metrics and run settings as portable JSON."""
import json
from pathlib import Path
import numpy as np

METRICS = ('loss', 'accuracy', 'precision', 'recall', 'f1')


def save_history(path, records, config, summary):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=1, config=config, summary=summary, epochs=records)
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_history(path):
    with open(path, encoding='utf-8') as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or payload.get('version') != 1:
        raise ValueError(f'{path}: unsupported history format')
    records = payload.get('epochs')
    if not isinstance(records, list) or not records:
        raise ValueError(f'{path}: history has no epochs')
    for epoch, record in enumerate(records, 1):
        if not isinstance(record, dict) or record.get('epoch') != epoch:
            raise ValueError(f'{path}: invalid epoch sequence')
        for partition in ('train', 'validation'):
            metrics = record.get(partition)
            if not isinstance(metrics, dict):
                raise ValueError(f'{path}: missing {partition} metrics')
            for name in METRICS:
                value = metrics.get(name)
                if not isinstance(value, (float, int)) or not np.isfinite(value):
                    raise ValueError(f'{path}: invalid {partition} {name}')
    return payload

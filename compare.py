"""Compare learning curves from saved JSON histories."""

import argparse
from pathlib import Path
from src.history import load_history, METRICS
from src.visualization import plot_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'histories', nargs='+', help='at least two JSON histories'
    )
    parser.add_argument('--labels', nargs='+', help='one label per history')
    parser.add_argument('--output', default='plots/comparison')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    try:
        if len(args.histories) < 2:
            raise ValueError('provide at least two histories')
        labels = (
            args.labels
            if args.labels is not None
            else [Path(p).stem for p in args.histories]
        )
        if len(labels) != len(args.histories) or len(set(labels)) != len(
            labels
        ):
            raise ValueError('provide one unique label per history')
        runs = [
            (label, load_history(path)['epochs'])
            for label, path in zip(labels, args.histories)
        ]
        plot_runs(runs, args.output, not args.no_show, METRICS)
        print(f'Comparison graphs saved: {args.output}')
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

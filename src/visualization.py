"""Plot learning curves from one run or compare several saved runs."""
import os
from pathlib import Path


def prepare_plotting(show):
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.mplconfig').resolve()))
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_runs(runs, directory, show, metrics=('loss', 'accuracy')):
    """Each run is a (label, epoch-records) pair; allow different run lengths."""
    plt = prepare_plotting(show)
    Path(directory).mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        fig, ax = plt.subplots()
        for index, (label, records) in enumerate(runs):
            epochs = [record['epoch'] for record in records]
            color = f'C{index % 10}'
            for partition, style in [('train', '--'), ('validation', '-')]:
                values = [record[partition][metric] for record in records]
                ax.plot(epochs, values, linestyle=style, color=color, label=f'{label}: {partition}')
        ax.set(xlabel='Epoch', ylabel=metric.capitalize(), title=f'{metric.capitalize()} during training')
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(directory) / f'{metric}.png', dpi=150)
    if show:
        plt.show()
    plt.close('all')


def plot_curves(history, directory, show):
    plot_runs([('Model', history)], directory, show)

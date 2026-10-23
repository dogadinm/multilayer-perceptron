"""Plot training and validation curves after training."""
import os
from pathlib import Path
import numpy as np


def plot_curves(history, directory, show):
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.mplconfig').resolve()))
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Path(directory).mkdir(parents=True, exist_ok=True)
    values = np.asarray(history)
    for name, train_col, valid_col in [('loss', 0, 2), ('accuracy', 1, 3)]:
        fig, ax = plt.subplots()
        epochs = np.arange(1, len(history) + 1)
        ax.plot(epochs, values[:, train_col], label='Train')
        ax.plot(epochs, values[:, valid_col], label='Validation')
        ax.set(xlabel='Epoch', ylabel=name.capitalize(), title=f'{name.capitalize()} during training')
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(directory) / f'{name}.png', dpi=150)
    if show:
        plt.show()
    plt.close('all')


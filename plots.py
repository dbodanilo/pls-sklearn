import os

import matplotlib.pyplot as plt

from util import get_globs, get_paths


def plot_components(name, title, xords, yords, xlabels, ylabels, X, Y, nrows=2, ncols=3, show=False):
    path = name + "-components"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    if not any(os.path.exists(path) for path in globs) or show:
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 4 * nrows), layout="constrained")

        for i, (x_ax, ordinal) in enumerate(zip(axes[0], xords)):
            x_ax.bar(xlabels, X[:, i])
            x_ax.set_ylim((-1, 1))
            x_ax.grid(True, axis="y")
            x_ax.set(title=f"{ordinal} {title} component")

        for j, (y_ax, ordinal) in enumerate(zip(axes[1], yords)):
            y_ax.bar(ylabels, Y[:, j])
            y_ax.set_ylim((-1, 1))
            y_ax.grid(True, axis="y")
            y_ax.set(title=f"{ordinal} {title} component")

        if show:
            fig.show()
        else:
            for path in paths:
                fig.savefig(path)


def plot_predictions(name, xlabels, ylabels, X, Y_true, Y_pred, R2, iter_x=True, iter_y=True, nrows=1, ncols=2, show=False):
    path = name + "-predictions"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    # Only generate it once.
    if not any(os.path.exists(path) for path in globs) or show:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4 * nrows), layout="constrained")

        # preview accuracy on first components.
        for i, ax in enumerate(axes):
            j = i if iter_y else 0
            i = i if iter_x else 0

            ax.scatter(X[:, i], Y_true[:, j], alpha=0.3,
                       label="ground truth")
            ax.scatter(X[:, i], Y_pred[:, j], alpha=0.3,
                       label="predictions")
            ax.set(xlabel=xlabels[i],
                   ylabel=ylabels[j],
                   title=f"{xlabels[i]} vs. {ylabels[j]}, $R^2 = {R2[j]:.3f}$")
            ax.legend()

        if show:
            fig.show()
        else:
            for path in paths:
                fig.savefig(path)

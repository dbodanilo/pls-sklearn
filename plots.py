import matplotlib.pyplot as plt
import numpy as np

from model import load_leme
from util import latexify


def plot_components(X, Y, xtitle, xords, xlabels, ytitle=None, yords=None, ylabels=None, nrows=2, ncols=3):
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10 * ncols, 4 * nrows), layout="constrained")
    if ytitle is None:
        ytitle = xtitle
    if yords is None:
        yords = xords
    if ylabels is None:
        ylabels = xlabels

    n_xlabels = len(xlabels)
    n_ylabels = len(ylabels)

    x_mean_x = np.linspace(-1, n_xlabels, n_xlabels)
    y_mean_x = np.linspace(-1, n_ylabels, n_ylabels)

    # j index of the maximum X[i, j] value.
    xj_max = np.absolute(X).max(axis=1).argmax()
    yj_max = xj_max
    if X.shape != Y.shape:
        yj_max = np.absolute(Y).max(axis=1).argmax()

    for i, (x_ax, ordinal) in enumerate(zip(axes.flat[:ncols], xords)):
        xi = X[:, i]
        # force largest variable to be positive.
        xi *= np.sign(xi[xj_max])

        x_ax.bar(xlabels, xi)
        x_ax.set_ylim((-1, 1))
        x_ax.grid(True, axis="y")
        x_ax.set(title=f"{ordinal} {xtitle} component")

        # reference: https://github.com/matplotlib/matplotlib/issues/13774#issuecomment-478250353
        plt.setp(x_ax.get_xticklabels(), rotation=45,
                 ha="right", rotation_mode="anchor")

        xi_abs_mean = np.absolute(xi).mean()
        xi_abs_means = np.repeat(xi_abs_mean, n_xlabels)

        x_ax.plot(x_mean_x,  xi_abs_means, color="g", label="$+$|mean|")
        x_ax.plot(x_mean_x, -xi_abs_means, color="r", label="$-$|mean|")
        x_ax.legend()
        x_ax.set_ylim((-1, 1))

    for i, (y_ax, ordinal) in enumerate(zip(axes.flat[ncols:], yords)):
        yi = Y[:, i]
        yi *= np.sign(yi[yj_max])

        y_ax.bar(ylabels, yi)
        y_ax.set_ylim((-1, 1))
        y_ax.grid(True, axis="y")
        y_ax.set(title=f"{ordinal} {ytitle} component")

        plt.setp(y_ax.get_xticklabels(), rotation=45,
                 ha="right", rotation_mode="anchor")

        yi_abs_mean = np.absolute(yi).mean()
        yi_abs_means = np.repeat(yi_abs_mean, n_ylabels)

        y_ax.plot(y_mean_x,  yi_abs_means, color="g", label="$+$|mean|")
        y_ax.plot(y_mean_x, -yi_abs_means, color="r", label="$-$|mean|")
        y_ax.legend()
        y_ax.set_ylim((-1, 1))

    return fig


def plot_correlations():
    """Reference: https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas"""

    leme_data = load_leme(split=False)
    leme_corr = leme_data.corr(method="spearman").iloc[7:, 2:7] ** 2

    xlabels = latexify(leme_data.iloc[:, 7:].columns)
    ylabels = latexify(leme_data.iloc[:, 2:7].columns)

    fig = plt.figure(figsize=(19, 15), layout="constrained")

    plt.matshow(leme_corr, fignum=fig.number)

    # Y has less variables, so I place it in x-axis.
    plt.xticks(range(len(ylabels)), ylabels, fontsize=14, rotation=45)
    plt.yticks(range(len(xlabels)), xlabels, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)

    plt.title("Correlation Matrix", fontsize=16)

    plt.show()


def plot_predictions(xlabels, ylabels, X, Y_true, Y_pred, R2, iter_x=True, iter_y=True, nrows=1, ncols=2):
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), layout="constrained")

    # preview accuracy on first components.
    for i, ax in enumerate(axes):
        xi = i if iter_x else 0
        yi = i if iter_y else 0

        ax.scatter(X[:, xi], Y_true[:, yi], alpha=0.3,
                   label="ground truth")
        ax.scatter(X[:, xi], Y_pred[:, yi], alpha=0.3,
                   label="predictions")
        ax.set(xlabel=xlabels[xi],
               ylabel=ylabels[yi],
               title=f"{xlabels[xi]} vs. {ylabels[yi]}, $R^2 = {R2[yi]:.3f}$")
        ax.legend()

    return fig

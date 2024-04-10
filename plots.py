import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import load_leme
from util import latexify


def plot_components(X, Y, xtitles, xlabels, ytitles=None, ylabels=None, meanlabel="mean", nrows=2, ncols=3, sort=None):
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10 * ncols, 4 * nrows), layout="constrained")
    is_pandas = type(X) == pd.DataFrame
    sort_asc = None if sort is None else sort == "asc"

    if ytitles is None:
        ytitles = xtitles
    if ylabels is None:
        ylabels = xlabels

    n_xlabels = len(xlabels)
    n_ylabels = len(ylabels)

    x_mean_x = np.linspace(-1, n_xlabels, n_xlabels)
    y_mean_x = np.linspace(-1, n_ylabels, n_ylabels)

    x0 = X.iloc[:, 0] if is_pandas else X[:, 0]
    # No sort.
    x_abs_argsort = np.arange(x0.shape[0])
    y_abs_argsort = x_abs_argsort
    if X.shape != Y.shape:
        y0 = Y.iloc[:, 0] if is_pandas else Y[:, 0]
        y_abs_argsort = np.arange(y0.shape[0])

    for i, (x_ax, xtitle) in enumerate(zip(axes.flat[:ncols], xtitles)):
        xi = X.iloc[:, i] if is_pandas else X[:, i]

        xi_abs_argsort = x_abs_argsort
        if sort_asc is not None:
            xi_abs_argsort = np.absolute(xi).argsort()
            if not sort_asc:
                xi_abs_argsort = np.flip(xi_abs_argsort)

        xi = xi.iloc[xi_abs_argsort] if is_pandas else xi[xi_abs_argsort]
        xlabels_i = xlabels[xi_abs_argsort]

        # Force first variable to be positive.
        xi0 = xi.iat[0] if is_pandas else xi[0]
        xi *= np.sign(xi0)

        x_ax.bar(xlabels_i, xi)
        x_ax.set_ylim((-1, 1))
        x_ax.grid(True, axis="y")
        x_ax.set(title=xtitle)

        # Reference: https://github.com/matplotlib/matplotlib/issues/13774#issuecomment-478250353
        plt.setp(x_ax.get_xticklabels(), rotation=45,
                 ha="right", rotation_mode="anchor")

        xi_abs_mean = np.absolute(xi).mean()
        xi_abs_means = np.repeat(xi_abs_mean, n_xlabels)

        x_ax.plot(x_mean_x,  xi_abs_means, color="g", label=f"$+$|{meanlabel}|")
        x_ax.plot(x_mean_x, -xi_abs_means, color="r", label=f"$-$|{meanlabel}|")
        x_ax.legend()
        x_ax.set_ylim((-1, 1))

    for i, (y_ax, ytitle) in enumerate(zip(axes.flat[ncols:], ytitles)):
        yi = Y.iloc[:, i] if is_pandas else Y[:, i]

        yi_abs_argsort = y_abs_argsort
        if sort_asc is not None:
            yi_abs_argsort = np.absolute(yi).argsort()
            if not sort_asc:
                yi_abs_argsort = np.flip(yi_abs_argsort)

        yi = yi.iloc[yi_abs_argsort] if is_pandas else yi[yi_abs_argsort]
        ylabels_i = ylabels[yi_abs_argsort]

        yi0 = yi.iat[0] if is_pandas else yi[0]
        yi *= np.sign(yi0)

        y_ax.bar(ylabels_i, yi)
        y_ax.set_ylim((-1, 1))
        y_ax.grid(True, axis="y")
        y_ax.set(title=ytitle)

        plt.setp(y_ax.get_xticklabels(), rotation=45,
                 ha="right", rotation_mode="anchor")

        yi_abs_mean = np.absolute(yi).mean()
        yi_abs_means = np.repeat(yi_abs_mean, n_ylabels)

        y_ax.plot(y_mean_x,  yi_abs_means, color="g", label=f"$+$|{meanlabel}|")
        y_ax.plot(y_mean_x, -yi_abs_means, color="r", label=f"$-$|{meanlabel}|")
        y_ax.legend()
        y_ax.set_ylim((-1, 1))

    return fig


def plot_correlations():
    """Reference: https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas"""

    leme_data, _ = load_leme(split=False)
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
    is_pandas = type(X) == pd.DataFrame
    n_max = min(X.shape[1], Y_true.shape[1])

    # preview accuracy on first components.
    for i, ax in enumerate(axes.flat[:n_max]):
        xi = i if iter_x else 0
        yi = i if iter_y else 0

        X_i = X.iloc[:, xi] if is_pandas else X[:, xi]
        Y_true_i = Y_true.iloc[:, yi] if is_pandas else Y_true[:, yi]
        Y_pred_i = Y_pred.iloc[:, yi] if is_pandas else Y_pred[:, yi]

        ax.scatter(X_i, Y_true_i, alpha=0.3,
                   label="ground truth")
        ax.scatter(X_i, Y_pred_i, alpha=0.3,
                   label="predictions")
        ax.set(xlabel=xlabels[xi],
               ylabel=ylabels[yi],
               title=f"{xlabels[xi]} vs. {ylabels[yi]}, $R^2 = {R2[yi]:.3f}$")
        ax.legend()

    return fig

def plot_regression(Y_true, Y_pred, xlabels, ylabels, titles, R2, nrows=1, ncols=2):
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), layout="constrained")
    is_pandas = type(Y_true) == pd.DataFrame
    n_samples = Y_true.shape[0]
    n_features = Y_true.shape[1]

    for i, ax in enumerate(axes.flat[:n_features]):
        Y_true_i = Y_true.iloc[:, i] if is_pandas else Y_true[:, i]
        Y_pred_i = Y_pred.iloc[:, i] if is_pandas else Y_pred[:, i]

        y_min_i = min(Y_true_i.min(), Y_pred_i.min())
        y_max_i = max(Y_true_i.max(), Y_pred_i.max())
        y_limits_i = np.linspace(y_min_i, y_max_i, n_samples)

        # Identity: Y_pred = Y_true
        ax.plot(y_limits_i, y_limits_i)
        ax.scatter(Y_true_i, Y_pred_i)
        ax.set(xlabel=xlabels[i], ylabel=ylabels[i],
               title=f"{titles[i]}, $R^2 = {R2[i]:.3f}$")

    return fig

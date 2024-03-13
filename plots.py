import matplotlib.pyplot as plt


def plot_components(title, xords, yords, xlabels, ylabels, X, Y, nrows=2, ncols=3):
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10 * ncols, 4 * nrows), layout="constrained")

    for i, (x_ax, ordinal) in enumerate(zip(axes.flat[:ncols], xords)):
        x_ax.bar(xlabels, X[:, i])
        x_ax.set_ylim((-1, 1))
        x_ax.grid(True, axis="y")
        x_ax.set(title=f"{ordinal} {title} component")

    for j, (y_ax, ordinal) in enumerate(zip(axes.flat[ncols:], yords)):
        y_ax.bar(ylabels, Y[:, j])
        y_ax.set_ylim((-1, 1))
        y_ax.grid(True, axis="y")
        y_ax.set(title=f"{ordinal} {title} component")

    return fig


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

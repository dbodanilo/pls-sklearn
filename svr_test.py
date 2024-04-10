import os

import matplotlib.pyplot as plt
import numpy as np

from decomposition import ScalerPCA
from model import load_leme, train_test_seed_split
from svr import ScalerSVR
from util import get_globs, get_paths


_SHOW = True
_PAUSE = True

X, Y = load_leme()

X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y)

n_samples = X_train.shape[0]
n_features = X_train.shape[1]
n_targets = Y_train.shape[1]

x_pca = ScalerPCA(n_components=n_features).fit(X_train)

y_pca = ScalerPCA(n_components=n_targets).fit(Y_train)


# === SVR ===

x_train = x_pca.transform(X_train)[:, 0]
x_test = x_pca.transform(X_test)[:, 0]

y_train = y_pca.transform(Y_train)[:, 0]
y_test = y_pca.transform(Y_test)[:, 0]

svr_rbf = ScalerSVR(kernel="rbf").fit(X_train, y_train)
svr_lin = ScalerSVR(kernel="linear").fit(X_train, y_train)
svr_poly = ScalerSVR(kernel="poly", coef0=1, degree=2).fit(X_train, y_train)

lw = 2
svrs = [svr_rbf, svr_lin, svr_poly]
svr_steps = [svr.named_steps["svr"] for svr in svrs]

kernel_labels = ["RBF", "Linear", "Polynomial"]
model_colors = ["m", "c", "g"]

path = "svr-regressions"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs) or _SHOW:
    fig, axes = plt.subplots(1, 3, figsize=(15, 10), layout="constrained",
                             sharey=True)

    for ax, svr, step, label, color in zip(axes, svrs, svr_steps, kernel_labels, model_colors):
        ax.plot(
            x_test,
            svr.predict(X_test),
            color=color,
            lw=lw,
            label=f"{label} model",
        )
        ax.scatter(
            x_train[step.support_],
            y_train[step.support_],
            facecolor="none",
            edgecolor=color,
            s=50,
            label=f"{label} support vectors",
        )
        ax.scatter(
            x_train[np.setdiff1d(np.arange(len(x_train)), step.support_)],
            y_train[np.setdiff1d(np.arange(len(y_train)), step.support_)],
            facecolor="none",
            edgecolor="k",
            s=50,
            label="other training data",
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=1,
            fancybox=True,
            shadow=True,
        )

    fig.text(0.5, 0.04, "data", ha="center", va="center")
    fig.text(0.06, 0.5, "target", ha="center",
             va="center", rotation="vertical")
    fig.suptitle("Support Vector Regression", fontsize=14)

    if _SHOW:
        fig.show()
    else:
        for path in paths:
            fig.savefig(path)

    if _PAUSE:
        input("Press Enter to continue...")

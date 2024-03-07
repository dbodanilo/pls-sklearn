import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize

from decomposition import ScalerPCA, ScalerPCR, ScalerSVR
from model import load_leme, train_test_seed_split
from plots import plot_predictions, plot_components
from util import fit_predict_try_transform, get_globs, get_paths, latexify


NOW = datetime.now()
TODAY = NOW.strftime("%Y-%m-%d")

X, Y = load_leme()

descriptors = latexify(X.columns.drop(["N.", "Semente"]))

targets = latexify(Y.columns.drop(["N.", "Semente"]))

X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y)

n_samples = X_train.shape[0]
n_features = X_train.shape[1]
n_targets = Y_train.shape[1]

n_test = X_test.shape[0]


# === PCA ===

x_pca = ScalerPCA(n_components=n_features).fit(X_train)
x_pca_step: PCA = x_pca.named_steps["pca"]

y_pca = ScalerPCA(n_components=n_targets).fit(Y_train)
y_pca_step: PCA = y_pca.named_steps["pca"]

print("PCA\n===")
print("X and Y\n-------")
for n in range(1, n_targets + 1):
    # TODO: as with the R-square at the end:
    # aggregate metric over all seeds (mean)
    x_explained = x_pca_step.explained_variance_ratio_[:n].sum()
    y_explained = y_pca_step.explained_variance_ratio_[:n].sum()
    print("\nn =", n)
    print(f"{100*x_explained:.2f}% of X's variance explained")
    print(f"{100*y_explained:.2f}% of Y's variance explained")

# Until comparable to that of Y's first two PCA components
# (scaled and on worse train/test split).
# TODO: go back to exploring n > 5, at least for X.
print("\nOnly X\n------")
for n in range(n_targets + 1, n_features + 1):
    x_explained = x_pca_step.explained_variance_ratio_[:n].sum()
    print("\nn =", n)
    print(f"{100*x_explained:.2f}% of X's variance explained")


# === PCR ===

# n > n_targets will only throw an error if using
# PLSCanonical, not PLSRegression.
# for PLSCanonical: min(n_samples, n_features, n_targets)
# for PLSRegression: min(n_samples, n_features)
n_max = min(n_samples, n_features, n_targets)

pcr = ScalerPCR(n_components=n_max).fit(X_train, Y_train)

X_test_pca = x_pca.transform(X_test)

Y_pred_pcr = pcr.predict(X_test)

# Last three targets are the most important (Av0, fT, Pwr).
pcr_predictions = {
    "name": "pcr",
    "xlabels": ["X's PCA " + str(i) for i in range(1, 4)],
    "ylabels": targets[-3:],
    "X": X_test_pca,
    "Y_true": Y_test[:, -3:],
    "Y_pred": Y_pred_pcr[:, -3:],
    "R2": r2_score(Y_test[:, -3:], Y_pred_pcr[:, -3:], multioutput="raw_values"),
    "iter_x": False,
    "ncols": 3,
}

plot_predictions(**pcr_predictions)

r_pcr = ScalerPCR(n_components=n_max).fit(Y_train, X_train)

Y_test_pca = y_pca.transform(Y_test)

X_pred_pcr = r_pcr.predict(Y_test)
X_pred_pcr_t = x_pca.transform(X_pred_pcr)

R2_X_pcr_t = r2_score(X_test_pca, X_pred_pcr_t, multioutput="raw_values")

path = "pcr-predictions_reversed"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

    # X's Principal Components
    for i, (ax, ord) in enumerate(zip(axes, ["1st", "2nd", "3rd"])):
        ax.scatter(Y_test_pca[:, 0], X_test_pca[:, i], alpha=0.3,
                   label="ground truth")
        ax.scatter(Y_test_pca[:, 0], X_pred_pcr_t[:, i], alpha=0.3,
                   label="predictions")
        ax.set(xlabel="Projected Y onto 1st PCA component",
               ylabel=f"Projected X onto {ord} PCA component",
               title=f"Y's PCA 1 vs. X's PCA {i + 1}, $R^2 = {R2_X_pcr_t[i]:.3f}$")
        ax.legend()

    for path in paths:
        fig.savefig(path)

Y_pred_pcr_t = y_pca.transform(Y_pred_pcr)

R2_Y_pcr_t = r2_score(Y_test_pca, Y_pred_pcr_t, multioutput="raw_values")

path = "pcr-predictions_transformed"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    axes[0].scatter(X_test_pca[:, 0], Y_test_pca[:, 0], alpha=0.3,
                    label="ground truth")
    axes[0].scatter(X_test_pca[:, 0], Y_pred_pcr_t[:, 0], alpha=0.3,
                    label="predictions")
    axes[0].set(xlabel="Projected X onto 1st PCA component",
                ylabel="Projected Y onto 1st PCA component",
                title=f"X's PCA 1 vs. Y's PCA 1, $R^2 = {R2_Y_pcr_t[0]:.3f}$")
    axes[0].legend()

    axes[1].scatter(X_test_pca[:, 0], Y_test_pca[:, 1], alpha=0.3,
                    label="ground truth")
    axes[1].scatter(X_test_pca[:, 0], Y_pred_pcr_t[:, 1], alpha=0.3,
                    label="predictions")
    axes[1].set(xlabel="Projected X onto 1st PCA component",
                ylabel="Projected Y onto 2nd PCA component",
                title=f"X's PCA 1 vs. Y's PCA 2, $R^2 = {R2_Y_pcr_t[1]:.3f}$")
    axes[0].legend()

    for path in paths:
        fig.savefig(path)


# === PLSR ===

plsr = PLSRegression(n_components=n_max).fit(X_train, Y_train)

X_test_pls, Y_test_pls = plsr.transform(X_test, Y_test)

Y_pred_plsr = plsr.predict(X_test)

R2_Y_plsr = r2_score(Y_test, Y_pred_plsr, multioutput="raw_values")

path = "plsr-predictions"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

    # targets: Av0, fT, Pwr
    indices = list(enumerate(targets))[2:]
    for ax, (i, target) in zip(axes, indices):
        # TODO: transform X_test via PCA for direct comparison with PCR,
        # or any other neutral dimensionality reduction.
        ax.scatter(X_test_pls[:, 0], Y_test[:, i], alpha=0.3,
                   label="ground truth")
        ax.scatter(X_test_pls[:, 0], Y_pred_plsr[:, i], alpha=0.3,
                   label="predictions")
        ax.set(xlabel="Projected X onto 1st PLS component",
               ylabel=target,
               title=f"X's 1st PLS component vs. {target}, $R^2 = {R2_Y_plsr[i]:.3f}$")
        ax.legend()

    for path in paths:
        fig.savefig(path)

r_plsr = PLSRegression(n_components=n_max).fit(Y_train, X_train)

Y_test_pls, X_test_pls = r_plsr.transform(Y_test, X_test)

X_pred_plsr = r_plsr.predict(Y_test)

_, X_pred_plsr_t = r_plsr.transform(Y_test, X_pred_plsr)

R2_X_plsr_t = r2_score(X_test_pls, X_pred_plsr_t, multioutput="raw_values")

path = "plsr-predictions_reversed"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

    # X's principal components.
    for i, (ax, ord) in enumerate(zip(axes, ["1st", "2nd", "3rd"])):
        ax.scatter(Y_test_pls[:, i], X_test_pls[:, i], alpha=0.3,
                   label="ground truth")
        ax.scatter(Y_test_pls[:, i], X_pred_plsr_t[:, i], alpha=0.3,
                   label="predictions")
        ax.set(xlabel=f"Projected Y onto {ord} PLS component",
               ylabel=f"Projected X onto {ord} PLS component",
               title=f"PLS {i + 1}, $R^2 = {R2_X_plsr_t[i]:.3f}$")
        ax.legend()

    for path in paths:
        fig.savefig(path)

_, Y_pred_plsr_t = plsr.transform(X_test, Y_pred_plsr)

R2_Y_plsr_t = r2_score(Y_test_pls, Y_pred_plsr_t, multioutput="raw_values")

path = "plsr-predictions_transformed"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    axes[0].scatter(X_test_pls[:, 0], Y_test_pls[:, 0], alpha=0.3,
                    label="ground truth")
    axes[0].scatter(X_test_pls[:, 0], Y_pred_plsr_t[:, 0], alpha=0.3,
                    label="predictions")
    axes[0].set(xlabel="Projected X onto 1st PLS component",
                ylabel="Projected Y onto 1st PLS component",
                title=f"PLS 1, $R^2 = {R2_Y_plsr_t[0]:.3f}$")
    axes[0].legend()

    axes[1].scatter(X_test_pls[:, 1], Y_test_pls[:, 1], alpha=0.3,
                    label="ground truth")
    axes[1].scatter(X_test_pls[:, 1], Y_pred_plsr_t[:, 1], alpha=0.3,
                    label="predictions")
    axes[1].set(xlabel="Projected X onto 2nd PLS component",
                ylabel="Projected Y onto 2nd PLS component",
                title=f"PLS 2, $R^2 = {R2_Y_plsr_t[1]:.3f}$")
    axes[1].legend()

    axes[2].scatter(X_test_pls[:, 2], Y_test_pls[:, 2], alpha=0.3,
                    label="ground truth")
    axes[2].scatter(X_test_pls[:, 2], Y_pred_plsr_t[:, 2], alpha=0.3,
                    label="predictions")
    axes[2].set(xlabel="Projected X onto 3rd PLS component",
                ylabel="Projected Y onto 3rd PLS component",
                title=f"PLS 3, $R^2 = {R2_Y_plsr_t[2]:.3f}$")
    axes[2].legend()

    for path in paths:
        fig.savefig(path)

x_plsr_components = normalize(plsr.x_rotations_, axis=0)
y_plsr_components = normalize(plsr.y_rotations_, axis=0)

path = "pls-first_components"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")

    axes[0].bar(descriptors, x_plsr_components[:, 0])
    axes[0].set_ylim((-1, 1))
    axes[0].grid(True, axis="y")
    axes[0].set(title="1st PLS components")

    axes[1].bar(targets, y_plsr_components[:, 0])
    axes[1].set_ylim((-1, 1))
    axes[1].grid(True, axis="y")

    for path in paths:
        fig.savefig(path)


ords = ["1st", "2nd", "3rd"]  # TODO: use itertools for /.*th/ ords.

pls_components = {
    "name": "pls",
    "title": "PLS",
    "xords": ords,
    "yords": ords,
    "xlabels": descriptors,
    "ylabels": targets,
    "X": x_plsr_components,
    "Y": y_plsr_components,
}

plot_components(**pls_components)


plsr_all = PLSRegression(n_components=1).fit(X.drop(columns=["N.", "Semente"]), Y.drop(columns=["N.", "Semente"]))

x_all_plsr_components = normalize(plsr_all.x_rotations_, axis=0)
y_all_plsr_components = normalize(plsr_all.y_rotations_, axis=0)

for seed in range(1241, 1246):
    X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y, seed=seed)
    plsr = PLSRegression(n_components=1).fit(X_train, Y_train)

    x_all_plsr_components = np.append(x_all_plsr_components, normalize(plsr.x_rotations_, axis=0), axis=1)
    y_all_plsr_components = np.append(y_all_plsr_components, normalize(plsr.y_rotations_, axis=0), axis=1)


pls_all_components = {
    "name": "pls_all",
    "title": "PLS",
    "xords": ["X's seed: " + str(seed) for seed in ["all", *range(1241, 1246)] ],
    "yords": ["Y's seed: " + str(seed) for seed in ["all", *range(1241, 1246)] ],
    "xlabels": descriptors,
    "ylabels": targets,
    "X": x_all_plsr_components,
    "Y": y_all_plsr_components,
    "ncols": 6,
}

plot_components(**pls_all_components)


# seed=1241: best seed for `X = predict(Y)` and second-best
# for `Y = predict(X)` (based on r2_score).
X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y, seed=1241)

plsr_all_targets = PLSRegression(n_components=1).fit(X_train, Y_train)

plsr_each_components = normalize(plsr_all_targets.x_rotations_, axis=0)

for target, Y_train_target in zip(targets, Y_train.T):
    plsr_each_target = PLSRegression(n_components=1).fit(X_train, Y_train_target)

    plsr_each_components = np.append(plsr_each_components, normalize(plsr_each_target.x_rotations_, axis=0), axis=1)

pls_targets_components = {
    "name": "pls_targets",
    "title": "PLS",
    "xords": [ "X's " + t for t in ["all", *targets[:-3]] ],
    "yords": [ "X's " + t for t in targets[-3:] ],
    "xlabels": descriptors,
    "ylabels": descriptors,
    "X": plsr_each_components[:, :3],
    "Y": plsr_each_components[:, 3:],
}

plot_components(**pls_targets_components)


# === PCR vs. PLSR ===

path = "pcr_vs_plsr-predictions"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    # preview accuracy on first components.
    axes[0].scatter(X_test_pca[:, 0], Y_test_pca[:, 0], alpha=0.3,
                    label="ground truth")
    axes[0].scatter(X_test_pca[:, 0], Y_pred_pcr_t[:, 0], alpha=0.3,
                    label="predictions")
    axes[0].set(
        xlabel="Projected X onto 1st PCA component",
        ylabel="Projected Y onto 1st PCA component",
        title=f"PCA Regression, $R^2 = {R2_Y_pcr_t[0]:.3f}$"
    )
    axes[0].legend()

    axes[1].scatter(X_test_pls[:, 0], Y_test_pls[:, 0], alpha=0.3,
                    label="ground truth")
    axes[1].scatter(X_test_pls[:, 0], Y_pred_plsr_t[:, 0], alpha=0.3,
                    label="predictions")
    axes[1].set(xlabel="Projected X onto 1st PLS component",
                ylabel="Projected Y onto 1st PLS component",
                title=f"PLS Regression, $R^2 = {R2_Y_plsr_t[0]:.3f}$")
    axes[1].legend()

    for path in paths:
        fig.savefig(path)

path = "pca_vs_pls-first_components"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    axes[0].bar(targets, y_pca_step.components_[0])
    axes[0].set_ylim((-1, 1))
    axes[0].grid(True, axis="y")
    axes[0].set(title="1st PCA component")

    axes[1].bar(targets, y_plsr_components[:, 0])
    axes[1].set_ylim((-1, 1))
    axes[1].grid(True, axis="y")
    axes[1].set(title="1st PLS component")

    for path in paths:
        fig.savefig(path)

path = "pca_vs_pls-components"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    axes[0, 0].bar(targets, y_pca_step.components_[0])
    axes[0, 0].set_ylim((-1, 1))
    axes[0, 0].grid(True, axis="y")
    axes[0, 0].set(title="1st PCA component")

    axes[0, 1].bar(targets, y_pca_step.components_[1])
    axes[0, 1].set_ylim((-1, 1))
    axes[0, 1].grid(True, axis="y")
    axes[0, 1].set(title="2nd PCA component")

    axes[0, 2].bar(targets, y_pca_step.components_[2])
    axes[0, 2].set_ylim((-1, 1))
    axes[0, 2].grid(True, axis="y")
    axes[0, 2].set(title="3rd PCA component")

    axes[1, 0].bar(targets, y_plsr_components[:, 0])
    axes[1, 0].set_ylim((-1, 1))
    axes[1, 0].grid(True, axis="y")
    axes[1, 0].set(title="1st PLS component")

    axes[1, 1].bar(targets, y_plsr_components[:, 1])
    axes[1, 1].set_ylim((-1, 1))
    axes[1, 1].grid(True, axis="y")
    axes[1, 1].set(title="2nd PLS component")

    axes[1, 2].bar(targets, y_plsr_components[:, 2])
    axes[1, 2].set_ylim((-1, 1))
    axes[1, 2].grid(True, axis="y")
    axes[1, 2].set(title="3rd PLS component")

    for path in paths:
        fig.savefig(path)

# seed=1241 was the best for the ratio of rPLSR's r2_score
# over rPCR's.
# TODO: print seed used when outputting plots and scores.
X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y, seed=1241)

X_test_pca, X_pred_pcr, X_pred_pcr_t, Y_test_pca, Y_pred_pcr, Y_pred_pcr_t = fit_predict_try_transform(
    ScalerPCR, X_train, X_test, Y_train, Y_test)

X_test_pls, X_pred_plsr, X_pred_plsr_t, Y_test_pls, Y_pred_plsr, Y_pred_plsr_t = fit_predict_try_transform(
    PLSRegression, X_train, X_test, Y_train, Y_test)

y_test_pca_min = min(Y_test_pca[:, 0].min(), Y_pred_pcr_t[:, 0].min())
y_test_pca_max = max(Y_test_pca[:, 0].max(), Y_pred_pcr_t[:, 0].max())
y_limits_pca = np.linspace(y_test_pca_min, y_test_pca_max, n_test)

y_test_pls_min = min(Y_test_pls[:, 0].min(), Y_pred_plsr_t[:, 0].min())
y_test_pls_max = max(Y_test_pls[:, 0].max(), Y_pred_plsr_t[:, 0].max())
y_limits_pls = np.linspace(y_test_pls_min, y_test_pls_max, n_test)

# NOTE: display R-squared for the prediction of each
# component.
R2_Y_pcr_t = r2_score(Y_test_pca, Y_pred_pcr_t, multioutput="raw_values")
R2_Y_plsr_t = r2_score(Y_test_pls, Y_pred_plsr_t, multioutput="raw_values")

path = "pcr_vs_plsr-regression-best_ratio"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

    axes[0].plot(y_limits_pca, y_limits_pca)
    axes[0].scatter(Y_test_pca[:, 0], Y_pred_pcr_t[:, 0])
    axes[0].set(xlabel="Actual Y projected onto 1st PCA component",
                ylabel="Predicted Y projected onto 1st PCA component",
                title=f"PCA Regression, $R^2 = {R2_Y_pcr_t[0]:.3f}$")

    axes[1].plot(y_limits_pls, y_limits_pls)
    axes[1].scatter(Y_test_pls[:, 0], Y_pred_plsr_t[:, 0])
    axes[1].set(xlabel="Actual Y projected onto 1st PLS component",
                ylabel="Predicted Y projected onto 1st PLS component",
                title=f"PLS Regression, $R^2 = {R2_Y_plsr_t[0]:.3f}$")

    for path in paths:
        fig.savefig(path)

x_test_pca_min = min(X_test_pca[:, 0].min(), X_pred_pcr_t[:, 0].min())
x_test_pca_max = max(X_test_pca[:, 0].max(), X_pred_pcr_t[:, 0].max())
x_limits_pca = np.linspace(x_test_pca_min, x_test_pca_max, n_test)

x_test_pls_min = min(X_test_pls[:, 0].min(), X_pred_plsr_t[:, 0].min())
x_test_pls_max = max(X_test_pls[:, 0].max(), X_pred_plsr_t[:, 0].max())
x_limits_pls = np.linspace(x_test_pls_min, x_test_pls_max, n_test)

R2_X_pcr_t = r2_score(X_test_pca, X_pred_pcr_t, multioutput="raw_values")
R2_X_plsr_t = r2_score(X_test_pls, X_pred_plsr_t, multioutput="raw_values")

path = "pcr_vs_plsr-regression_reversed-best_ratio"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

    axes[0].plot(x_limits_pca, x_limits_pca)
    axes[0].scatter(X_test_pca[:, 0], X_pred_pcr_t[:, 0])
    axes[0].set(xlabel="Actual X projected onto 1st PCA component",
                ylabel="Predicted X projected onto 1st PCA component",
                title=f"PCA Regression, $R^2 = {R2_X_pcr_t[0]:.3f}$")

    axes[1].plot(x_limits_pls, x_limits_pls)
    axes[1].scatter(X_test_pls[:, 0], X_pred_plsr_t[:, 0])
    axes[1].set(xlabel="Actual X projected onto 1st PLS component",
                ylabel="Predicted X projected onto 1st PLS component",
                title=f"PLS Regression, $R^2 = {R2_X_plsr_t[0]:.3f}$")
    for path in paths:
        fig.savefig(path)

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
if not any(os.path.exists(path) for path in globs):
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

    for path in paths:
        fig.savefig(path)


# DOING: train only multi-output regressors.
model_labels = ["PCR", "PLSR"]  # , "SVR"
r_labels = ["r" + label for label in model_labels]

models = [ScalerPCR, PLSRegression]  # , ScalerSVR

algos = np.empty(400, dtype="U5")
n_useds = np.empty(400, dtype=int)
ns = np.empty(400, dtype=int)
r2s = np.empty(400, dtype=float)
seeds = np.empty(400, dtype=int)
ts = np.empty(400, dtype=bool)
i = 0
for seed in range(1241, 1246):
    X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y, seed)

    for n in range(1, n_max + 1):
        for label, model in zip(model_labels, models):
            r_label = "r" + label

            X_test_t, X_pred, X_pred_t, Y_test_t, Y_pred, Y_pred_t = fit_predict_try_transform(
                model, X_train, X_test, Y_train, Y_test, n_components=n)

            for n_used in range(1, n + 1):
                # TODO: review imposing a lower bound
                # (-1 or 0) to avoid skewing the mean, as
                # the r2's upper bound is positive 1, but
                # there's no lower bound.
                # 0 seems to be more mathematically sound
                # than -1. But -1 retains the information on
                # whether the regressor performs worse than
                # DummyRegressor(strategy='mean').
                # r2 = max(r2_score(Y_test, Y_pred), 0)
                r2_m = r2_score(Y_test_t[:, :n_used], Y_pred_t[:, :n_used])
                r2_rm = r2_score(X_test_t[:, :n_used], X_pred_t[:, :n_used])

                # A numpy.ndarray for each column.
                for algo, r2 in zip([label, r_label], [r2_m, r2_rm]):
                    algos[i] = algo
                    n_useds[i] = n_used
                    ns[i] = n
                    r2s[i] = r2
                    seeds[i] = seed
                    ts[i] = True
                    i += 1

                # Compare predictions in original feature
                # space, this confirms that calculating the
                # score on the transformed feature space
                # makes it easier for lower `n` to achieve
                # higher `r2_score`, which makes it harder
                # to observe the expected increase in
                # `r2_score` as `n` increases.
                if n_used == n:
                    r2_m = r2_score(Y_test, Y_pred)
                    r2_rm = r2_score(X_test, X_pred)

                    # TODO: explicitly aggregate r2_score
                    # over each variable, i.e., use its
                    # `multioutput` parameter and extract
                    # mean and std over variables.
                    for algo, r2 in zip([label, r_label], [r2_m, r2_rm]):
                        algos[i] = algo
                        n_useds[i] = n_used
                        ns[i] = n
                        r2s[i] = r2
                        seeds[i] = seed
                        ts[i] = False
                        i += 1


# NOTE: avoid aggregation outside pandas!
# e.g., mean values over all seeds;
# and avoid conversion back to python list,
# pandas.DataFrame seems not to work with numpy.array[dict]
r2s_df = pd.DataFrame({
    "algo": algos,
    "n_used": n_useds,
    "n": ns,
    "r2": r2s,
    "seed": seeds,
    "t": ts
})

print("\nPCA vs. PLS vs. SVR", end="")
print("\n===================")
print("(mean, std) over all five seeds")
print("-------------------------------")
for n in range(1, n_max + 1):
    # Print results on original feature space only.
    print("\nn == n_used =", n)
    r2s_df_n = r2s_df[(r2s_df["n"] == n) & (r2s_df["n_used"] == n)]

    for label in model_labels:
        r2s_df_n_algo = r2s_df_n[r2s_df_n["algo"] == label]["r2"]
        # TODO: remove outliers based on distance from mean,
        # not just the min and max samples.
        # r2_min = r2s_df_n_algo["r2"].min()
        # r2_max = r2s_df_n_algo["r2"].max()
        # mean = (r2s_df_n_algo["r2"].sum() - (r2_min + r2_max)) / 3
        mean = r2s_df_n_algo.mean()
        print(label, "R-squared:", f"({mean:.3f}, ", end="")
        print(f"{r2s_df_n_algo.std():.3f})")

print("\n(mean, std) over all five ns", end="")
print("\n----------------------------")
for seed in range(1241, 1246):
    print("\nseed =", seed)
    r2s_df_seed = r2s_df[r2s_df["seed"] == seed]

    for label in model_labels:
        r2s_df_seed_algo = r2s_df_seed[r2s_df_seed["algo"] == label]["r2"]
        mean = r2s_df_seed_algo.mean()
        print(label, "R-squared:", f"({mean:.3f}, ", end="")
        print(f"{r2s_df_seed_algo.std():.3f})")

print("\nPCA vs. PLS vs. SVR (reverse, X = model.predict(Y))", end="")
print("\n===================================================")
print("(mean, std) over all five seeds")
print("-------------------------------")
for n in range(1, n_max + 1):
    print("\nn == n_used =", n)
    r2s_df_n = r2s_df[(r2s_df["n"] == n) & (r2s_df["n_used"] == n)]

    for label in r_labels:
        r2s_df_n_algo = r2s_df_n[r2s_df_n["algo"] == label]["r2"]
        mean = r2s_df_n_algo.mean()
        print(label, "R-squared:", f"({mean:.3f}, ", end="")
        print(f"{r2s_df_n_algo.std():.3f})")

print("\n(mean, std) over all five ns", end="")
print("\n----------------------------")
for seed in range(1241, 1246):
    print("\nseed =", seed)
    r2s_df_seed = r2s_df[r2s_df["seed"] == seed]

    for label in r_labels:
        r2s_df_seed_algo = r2s_df_seed[r2s_df_seed["algo"] == label]["r2"]
        mean = r2s_df_seed_algo.mean()
        print(label, "R-squared:", f"({mean:.3f}, ", end="")
        print(f"{r2s_df_seed_algo.std():.3f})")

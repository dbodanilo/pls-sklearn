import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize

from decomposition import ScalerPCA, ScalerPCR, ScalerSVR
from evol import Evol, UV_EVOL
from model import load_leme, train_test_seed_split
from util import fig_paths


X, Y = load_leme()

descriptors = X.columns.drop(["N.", "Semente"])
targets = Y.columns.drop(["N.", "Semente"])

X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y)

n_samples = X_train.shape[0]
n_features = X_train.shape[1]
n_targets = Y_train.shape[1]

# n > n_targets will only throw an error if using
# PLSCanonical, not PLSRegression.
# for PLSCanonical: min(n_samples, n_features, n_targets)
# for PLSRegression: min(n_samples, n_features)
n_max = min(n_samples, n_features, n_targets)


# === PCA ===

x_pca = ScalerPCA(n_components=n_max).fit(X_train)
x_pca_step: PCA = x_pca.named_steps["pca"]

y_pca = ScalerPCA(n_components=n_max).fit(Y_train)
y_pca_step: PCA = y_pca.named_steps["pca"]

print("PCA\n===")
print("X and Y\n-------")
for n in range(1, n_max + 1):
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
while x_explained < 0.9236 and n < n_max:
    n += 1
    x_explained = x_pca_step.explained_variance_ratio_[:n].sum()
    print("\nn =", n)
    print(f"{100*x_explained:.2f}% of X's variance explained")


# === PCR ===

pcr = ScalerPCR(n_components=n_max).fit(X_train, Y_train)

X_test_pca = x_pca.transform(X_test)

Y_pred_pcr = pcr.predict(X_test)

paths = fig_paths("pcr-predictions")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

    # targets: Av0, fT, Pwr
    indices = list(enumerate(targets))[2:]
    for ax, (i, target) in zip(axes, indices):
        ax.scatter(X_test_pca[:, 0], Y_test[:, i], alpha=0.3,
                   label="ground truth")
        ax.scatter(X_test_pca[:, 0], Y_pred_pcr[:, i], alpha=0.3,
                   label="predictions")
        ax.set(xlabel="Projected X onto 1st PCA component",
               ylabel=target,
               title="X's 1st PCA component vs. " + target)
        ax.legend()

    for path in paths:
        fig.savefig(path)


r_pcr = ScalerPCR(n_components=n_max).fit(Y_train, X_train)

Y_test_pca = y_pca.transform(Y_test)

X_pred_pcr = r_pcr.predict(Y_test)
X_pred_pcr_t = x_pca.transform(X_pred_pcr)

paths = fig_paths("pcr-predictions_reversed")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

    # X's Principal Components
    for i, (ax, ord) in enumerate(zip(axes, ["1st", "2nd", "3rd"])):
        ax.scatter(Y_test_pca[:, 0], X_test_pca[:, i], alpha=0.3,
                   label="ground truth")
        ax.scatter(Y_test_pca[:, 0], X_pred_pcr_t[:, i], alpha=0.3,
                   label="predictions")
        ax.set(xlabel="Projected Y onto 1st PCA component",
               ylabel=f"Projected X onto {ord} PCA component",
               title=f"Y's 1st PCA component vs. X's {ord} PCA component")
        ax.legend()

    for path in paths:
        fig.savefig(path)

Y_pred_pcr_t = y_pca.transform(Y_pred_pcr)

paths = fig_paths("pcr-predictions_transformed")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    axes[0].scatter(X_test_pca[:, 0], Y_test_pca[:, 0], alpha=0.3,
                    label="ground truth")
    axes[0].scatter(X_test_pca[:, 0], Y_pred_pcr_t[:, 0], alpha=0.3,
                    label="predictions")
    axes[0].set(xlabel="Projected X onto 1st PCA component",
                ylabel="Projected Y onto 1st PCA component",
                title="X's 1st PCA component vs. Y's 1st PCA component")
    axes[0].legend()

    axes[1].scatter(X_test_pca[:, 1], Y_test_pca[:, 0], alpha=0.3,
                    label="ground truth")
    axes[1].scatter(X_test_pca[:, 1], Y_pred_pcr_t[:, 0], alpha=0.3,
                    label="predictions")
    axes[1].set(xlabel="Projected X onto 2nd PCA component",
                ylabel="Projected Y onto 1st PCA component",
                title="X's 1st PCA component vs. Y's 1st PCA component")
    axes[0].legend()

    for path in paths:
        fig.savefig(path)


# === PLSR ===

plsr = PLSRegression(n_components=n_max).fit(X_train, Y_train)

X_test_pls, Y_test_pls = plsr.transform(X_test, Y_test)

Y_pred_plsr = plsr.predict(X_test)

paths = fig_paths("plsr-predictions")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
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
               title="X's 1st PLS component vs. " + target)
        ax.legend()

    for path in paths:
        fig.savefig(path)

r_plsr = PLSRegression(n_components=n_max).fit(Y_train, X_train)

Y_test_pls, X_test_pls = r_plsr.transform(Y_test, X_test)

X_pred_plsr = r_plsr.predict(Y_test)

_, X_pred_plsr_t = r_plsr.transform(Y_test, X_pred_plsr)

paths = fig_paths("plsr-predictions_reversed")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

    # X's principal components.
    for i, (ax, ord) in enumerate(zip(axes, ["1st", "2nd", "3rd"])):
        ax.scatter(Y_test_pls[:, i], X_test_pls[:, i], alpha=0.3,
                   label="ground truth")
        ax.scatter(Y_test_pls[:, i], X_pred_plsr_t[:, i], alpha=0.3,
                   label="predictions")
        ax.set(xlabel=f"Projected Y onto {ord} PLS component",
               ylabel=f"Projected X onto {ord} PLS component",
               title=f"Y's {ord} PLS component vs. X's {ord} PLS component")
        ax.legend()

    for path in paths:
        fig.savefig(path)

_, Y_pred_plsr_t = plsr.transform(X_test, Y_pred_plsr)

paths = fig_paths("plsr-predictions_transformed")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    axes[0].scatter(X_test_pls[:, 0], Y_test_pls[:, 0], alpha=0.3,
                    label="ground truth")
    axes[0].scatter(X_test_pls[:, 0], Y_pred_plsr_t[:, 0], alpha=0.3,
                    label="predictions")
    axes[0].set(xlabel="Projected X onto 1st PLS component",
                ylabel="Projected Y onto 1st PLS component",
                title="PLS 1")
    axes[0].legend()

    axes[1].scatter(X_test_pls[:, 1], Y_test_pls[:, 1], alpha=0.3,
                    label="ground truth")
    axes[1].scatter(X_test_pls[:, 1], Y_pred_plsr_t[:, 1], alpha=0.3,
                    label="predictions")
    axes[1].set(xlabel="Projected X onto 2nd PLS component",
                ylabel="Projected Y onto 2nd PLS component",
                title="PLS 2")
    axes[1].legend()

    axes[2].scatter(X_test_pls[:, 2], Y_test_pls[:, 2], alpha=0.3,
                    label="ground truth")
    axes[2].scatter(X_test_pls[:, 2], Y_pred_plsr_t[:, 2], alpha=0.3,
                    label="predictions")
    axes[2].set(xlabel="Projected X onto 3rd PLS component",
                ylabel="Projected Y onto 3rd PLS component",
                title="PLS 3")
    axes[2].legend()

    for path in paths:
        fig.savefig(path)

x_plsr_components = normalize(plsr.x_rotations_, axis=0)
y_plsr_components = normalize(plsr.y_rotations_, axis=0)

paths = fig_paths("pls-first_components")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
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

paths = fig_paths("pls-components")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(2, 3, figsize=(30, 8), layout="constrained")

    axes[0, 0].bar(descriptors, x_plsr_components[:, 0])
    axes[0, 0].set_ylim((-1, 1))
    axes[0, 0].grid(True, axis="y")
    axes[0, 0].set(title="1st PLS components")

    axes[0, 1].bar(descriptors, x_plsr_components[:, 1])
    axes[0, 1].set_ylim((-1, 1))
    axes[0, 1].grid(True, axis="y")
    axes[0, 1].set(title="2nd PLS components")

    axes[0, 2].bar(descriptors, x_plsr_components[:, 2])
    axes[0, 2].set_ylim((-1, 1))
    axes[0, 2].grid(True, axis="y")
    axes[0, 2].set(title="3rd PLS components")

    axes[1, 0].bar(targets, y_plsr_components[:, 0])
    axes[1, 0].set_ylim((-1, 1))
    axes[1, 0].grid(True, axis="y")

    axes[1, 1].bar(targets, y_plsr_components[:, 1])
    axes[1, 1].set_ylim((-1, 1))
    axes[1, 1].grid(True, axis="y")

    axes[1, 2].bar(targets, y_plsr_components[:, 2])
    axes[1, 2].set_ylim((-1, 1))
    axes[1, 2].grid(True, axis="y")

    for path in paths:
        fig.savefig(path)


# === PCR vs. PLSR ===

paths = fig_paths("pcr_vs_plsr-predictions")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    # preview accuracy on first components.
    axes[0].scatter(X_test_pca[:, 0], Y_test_pca[:, 0], alpha=0.3,
                    label="ground truth")
    axes[0].scatter(X_test_pca[:, 0], Y_pred_pcr_t[:, 0], alpha=0.3,
                    label="predictions")
    axes[0].set(
        xlabel="Projected X onto 1st PCA component",
        ylabel="Projected Y onto 1st PCA component",
        title="PCA Regression"
    )
    axes[0].legend()

    axes[1].scatter(X_test_pls[:, 0], Y_test_pls[:, 0], alpha=0.3,
                    label="ground truth")
    axes[1].scatter(X_test_pls[:, 0], Y_pred_plsr_t[:, 0], alpha=0.3,
                    label="predictions")
    axes[1].set(xlabel="Projected X onto 1st PLS component",
                ylabel="Projected Y onto 1st PLS component",
                title="PLS Regression")
    axes[1].legend()

    for path in paths:
        fig.savefig(path)

paths = fig_paths("pca_vs_pls-first_components")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
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

paths = fig_paths("pca_vs_pls-components")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
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

y_test_pls_min = min(Y_test_pls[:, 0].min(), Y_pred_plsr_t[:, 0].min())
y_test_pls_max = max(Y_test_pls[:, 0].max(), Y_pred_plsr_t[:, 0].max())
limits_pls = np.arange(y_test_pls_min, y_test_pls_max, 0.01)

y_test_pca_min = min(Y_test_pca[:, 0].min(), Y_pred_pcr_t[:, 0].min())
y_test_pca_max = max(Y_test_pca[:, 0].max(), Y_pred_pcr_t[:, 0].max())
limits_pca = np.arange(y_test_pca_min, y_test_pca_max, 0.01)

# TODO: display R-squared for the prediction of the first components.
# r2_score(Y_test_pca[:, 0], Y_pred_pcr_t[:, 0])  # ~0.24
# r2_score(Y_test_pls[:, 0], Y_pred_plsr_t[:, 0])  # ~0.65
paths = fig_paths("pcr_vs_plsr-regression")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

    axes[0].plot(limits_pca, limits_pca)
    axes[0].scatter(Y_test_pca[:, 0], Y_pred_pcr_t[:, 0])
    axes[0].set(xlabel="Actual Y projected onto 1st PCA component",
                ylabel="Predicted Y projected onto 1st PCA component",
                title="PCA Regression")

    axes[1].plot(limits_pls, limits_pls)
    axes[1].scatter(Y_test_pls[:, 0], Y_pred_plsr_t[:, 0])
    axes[1].set(xlabel="Actual Y projected onto 1st PLS component",
                ylabel="Predicted Y projected onto 1st PLS component",
                title="PLS Regression")

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

paths = fig_paths("svr-regressions")

if not all(os.path.exists(path) for path in paths):
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


# === Evol ===

n_evol = 3
uv_pls = np.concatenate(
    [plsr.x_weights_[:, :n_evol], plsr.y_weights_[:, :n_evol]], axis=None)

print("\nEvol Test\n=========")
print("R-squared\n---------")
print("\nn =", n_evol)

for algo, uv in [("PLS U,V", uv_pls), ("Evol U,V", UV_EVOL)]:
    evol = Evol(n_components=n_evol).fit(X_train, Y_train, uv)

    Y_pred_evol = evol.predict(X_test)
    r2 = r2_score(Y_test, Y_pred_evol)
    print(algo + " R-squared:", round(r2, 3))

# Just to make sure it's my implementation's fault.
n_plsr = plsr.x_weights_.shape[1]
print(f"\nregular PLS (n = {n_plsr}):",
      round(plsr.score(X_test, Y_test), 3))

# np.dot(X.T, Y) ** 2: squared cross-covariance matrix.
print("\ncross covariance\n----------------")
for name, model in [("plsr", plsr), ("evol", evol)]:
    C2 = np.dot(model.x_scores_.T, model.y_scores_) ** 2

    print(f"diag(C_{name}^2) = {np.round(np.diag(C2), 3)}")
    print(f"|C_{name}^2| = {np.linalg.norm(C2):.3f}")

X_test_evol, Y_test_evol = evol.transform(X_test, Y_test)

x_test_evol0 = X_test_evol[:, 0]
x_test_evol1 = X_test_evol[:, 1]
x_test_evol2 = X_test_evol[:, 2]

y_test_evol0 = Y_test_evol[:, 0]
y_test_evol1 = Y_test_evol[:, 1]
y_test_evol2 = Y_test_evol[:, 2]

Y_pred_evol = evol.predict(X_test)
_, Y_pred_evol_t = evol.transform(X_test, Y_pred_evol)

# (200 x 5) @ (5 x 1)
y_pred_evol0 = Y_pred_evol_t[:, 0]
y_pred_evol1 = Y_pred_evol_t[:, 1]
y_pred_evol2 = Y_pred_evol_t[:, 2]

paths = fig_paths("evol-predictions")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    axes[0].scatter(x_test_evol0, y_test_evol0, alpha=0.3,
                    label="ground truth")
    axes[0].scatter(x_test_evol0, y_pred_evol0, alpha=0.3,
                    label="predictions")
    axes[0].set(xlabel="Projected X onto 1st Evolved component",
                ylabel="Projected Y onto 1st Evolved component",
                title="Evol 1")
    axes[0].legend()

    axes[1].scatter(x_test_evol1, y_test_evol1, alpha=0.3,
                    label="ground truth")
    axes[1].scatter(x_test_evol1, y_pred_evol1, alpha=0.3,
                    label="predictions")
    axes[1].set(xlabel="Projected X onto 2nd evolved component",
                ylabel="Projected Y onto 2nd evolved component",
                title="Evol 2")
    axes[1].legend()

    axes[2].scatter(x_test_evol2, y_test_evol2, alpha=0.3,
                    label="ground truth")
    axes[2].scatter(x_test_evol2, y_pred_evol2, alpha=0.3,
                    label="predictions")
    axes[2].set(xlabel="Projected X onto 3rd evolved component",
                ylabel="Projected Y onto 3rd evolved component",
                title="Evol 3")
    axes[2].legend()

    for path in paths:
        fig.savefig(path)

model_labels = ["PCR", "PLSR", "SVR"]
models = [ScalerPCR, PLSRegression, ScalerSVR]

r2s = np.empty(150, dtype=object)
i = 0
for seed in range(1241, 1246):
    X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y, seed)

    x_pca = ScalerPCA(n_components=n_max).fit(X_train)
    y_pca = ScalerPCA(n_components=n_max).fit(Y_train)

    # TODO: train only multi-output regressors.
    x_train = x_pca.transform(X_train)[:, 0]
    x_test = x_pca.transform(X_test)[:, 0]

    y_train = y_pca.transform(Y_train)[:, 0]
    y_test = y_pca.transform(Y_test)[:, 0]

    for n in range(1, n_max + 1):
        for label, model in zip(model_labels, models):
            m = model(n_components=n).fit(X_train, y_train)

            # reverse model (predict sizing based on target metrics)
            rm = model(n_components=n).fit(Y_train, x_train)

            # impose [-1, 1] limit to avoid skewing the mean,
            # as the r2 is bound by positive 1, but not by
            # negative 1.
            # r2 = max(m.score(X_test, y_test), -1)
            r2_m = m.score(X_test, y_test)
            r2_rm = rm.score(Y_test, x_test)

            # TODO: have a numpy.ndarray for each column.
            r2s[i] = {"seed": seed, "n": n, "algo": label, "r2": r2_m}
            r2s[i + 1] = {"seed": seed, "n": n,
                          "algo": "r" + label, "r2": r2_rm}
            i += 2

# NOTE: avoid aggregation outside pandas!
# e.g., mean values over all seeds.
# TODO: avoid conversion back to python list,
# but pd.DataFrame seems not to work with np.array[dict]
r2s_df = pd.DataFrame(list(r2s))

print("\nPCA vs. PLS vs. SVR", end="")
print("\n===================")
print("(mean, std) over all five seeds")
print("-------------------------------")
for n in range(1, n_max + 1):
    print("\nn =", n)
    r2s_df_n = r2s_df[r2s_df["n"] == n]

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

r_labels = ["r" + label for label in model_labels]

print("\nPCA vs. PLS vs. SVR (reverse, X = model.predict(Y))", end="")
print("\n===================================================")
print("(mean, std) over all five seeds")
print("-------------------------------")
for n in range(1, n_max + 1):
    print("\nn =", n)
    r2s_df_n = r2s_df[r2s_df["n"] == n]

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

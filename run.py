import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize

from decomposition import PCAScaled, PCR
from evol import Evol, UV_EVOL
from model import load_leme, train_test_seed_split
from preprocessing import scale_transform
from util import fig_paths


X, Y = load_leme()

X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y)

n_samples = X_train.shape[0]
n_features = X_train.shape[1]
n_targets = Y_train.shape[1]

# n > n_targets will only throw an error if using
# PLSCanonical, not PLSRegression.
# for PLSCanonical: min(n_samples, n_features, n_targets)
# for PLSRegression: min(n_samples, n_features)
n_max = min(n_samples, n_features, n_targets)

pcr = PCR(n_components=n_max).fit(X_train, Y_train)
x_pca_step: PCA = pcr.named_steps["pca"]

y_pca = PCAScaled().fit(Y_train)
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
print("\nOnly X\n------")
while x_explained < 0.9236 and n < n_max:
    n += 1
    x_explained = x_pca_step.explained_variance_ratio_[:n].sum()
    print("\nn =", n)
    print(f"{100*x_explained:.2f}% of X's variance explained")

X_test_pca = x_pca_step.transform(X_test)
Y_test_pca = y_pca.transform(Y_test)

x0 = X_test_pca[:, 0]
x1 = X_test_pca[:, 1]
y = Y_test_pca[:, 0]

paths = fig_paths("pca-projections")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(x0, y, alpha=0.3)
    axes[0].set(xlabel="Projected X onto 1st PCA component",
                ylabel="Projected Y onto 1st PCA component")

    axes[1].scatter(x1, y, alpha=0.3)
    axes[1].set(xlabel="Projected X onto 2nd PCA component",
                ylabel="Projected Y onto 1st PCA component")

    fig.tight_layout()

    for path in paths:
        fig.savefig(path)

plsr = PLSRegression(n_components=n_max).fit(X_train, Y_train)

X_test_pls, Y_test_pls = plsr.transform(X_test, Y_test)

_, Y_pred_plsr_t = plsr.transform(X_test, plsr.predict(X_test))

paths = fig_paths("pls-predictions")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
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

    fig.tight_layout()

    for path in paths:
        fig.savefig(path)

descriptors = X.columns.drop(["N.", "Semente"])
targets = Y.columns.drop(["N.", "Semente"])

x_plsr_components = normalize(plsr.x_rotations_, axis=0)
y_plsr_components = normalize(plsr.y_rotations_, axis=0)

paths = fig_paths("pls-components")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(2, 3, figsize=(30, 12))

    axes[0, 0].bar(descriptors, x_plsr_components[:, 0])
    axes[0, 0].set_ylim((-1, 1))
    axes[0, 0].set(title="X PLS 1")

    axes[0, 1].bar(descriptors, x_plsr_components[:, 1])
    axes[0, 1].set_ylim((-1, 1))
    axes[0, 1].set(title="X PLS 2")

    axes[0, 2].bar(descriptors, x_plsr_components[:, 2])
    axes[0, 2].set_ylim((-1, 1))
    axes[0, 2].set(title="X PLS 3")

    axes[1, 0].bar(targets, y_plsr_components[:, 0])
    axes[1, 0].set_ylim((-1, 1))
    axes[1, 0].set(title="Y PLS 1")

    axes[1, 1].bar(targets, y_plsr_components[:, 1])
    axes[1, 1].set_ylim((-1, 1))
    axes[1, 1].set(title="Y PLS 2")

    axes[1, 2].bar(targets, y_plsr_components[:, 2])
    axes[1, 2].set_ylim((-1, 1))
    axes[1, 2].set(title="Y PLS 3")

    fig.tight_layout()

    for path in paths:
        fig.savefig(path)


x_train_mean = X_train.mean(axis=0)
x_train_std = X_train.std(axis=0)

y_train_mean = Y_train.mean(axis=0)
y_train_std = Y_train.std(axis=0)

X_test_pca = scale_transform(x_pca_step, X_test, x_train_mean, x_train_std)

Y_test_pca = scale_transform(y_pca, Y_test, y_train_mean, y_train_std)

Y_pred_pcr = pcr.predict(X_test)
Y_pred_pcr_t = scale_transform(y_pca, Y_pred_pcr, y_train_mean, y_train_std)

paths = fig_paths("pca_vs_pls-predictions")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # preview accuracy on first components.
    axes[0].scatter(X_test_pca[:, 0], Y_test_pca[:, 0], alpha=0.3,
                    label="ground truth")
    axes[0].scatter(X_test_pca[:, 0], Y_pred_pcr_t[:, 0], alpha=0.3,
                    label="predictions")
    axes[0].set(
        xlabel="Projected X onto 1st PCA component",
        ylabel="Projected Y onto 1st PCA component",
        title="PCR / PCA"
    )
    axes[0].legend()

    axes[1].scatter(X_test_pls[:, 0], Y_test_pls[:, 0], alpha=0.3,
                    label="ground truth")
    axes[1].scatter(X_test_pls[:, 0], Y_pred_plsr_t[:, 0], alpha=0.3,
                    label="predictions")
    axes[1].set(xlabel="Projected X onto 1st PLS component",
                ylabel="Projected Y onto 1st PLS component",
                title="PLS")
    axes[1].legend()

    fig.tight_layout()

    for path in paths:
        fig.savefig(path)

paths = fig_paths("pca_vs_pls-components")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    axes[0, 0].bar(targets, y_pca_step.components_[0])
    axes[0, 0].set_ylim((-1, 1))
    axes[0, 0].set(title="Y PCA 1")

    axes[0, 1].bar(targets, y_pca_step.components_[1])
    axes[0, 1].set_ylim((-1, 1))
    axes[0, 1].set(title="Y PCA 2")

    axes[0, 2].bar(targets, y_pca_step.components_[2])
    axes[0, 2].set_ylim((-1, 1))
    axes[0, 2].set(title="Y PCA 3")

    axes[1, 0].bar(targets, y_plsr_components[:, 0])
    axes[1, 0].set_ylim((-1, 1))
    axes[1, 0].set(title="Y PLS 1")

    axes[1, 1].bar(targets, y_plsr_components[:, 1])
    axes[1, 1].set_ylim((-1, 1))
    axes[1, 1].set(title="Y PLS 2")

    axes[1, 2].bar(targets, y_plsr_components[:, 2])
    axes[1, 2].set_ylim((-1, 1))
    axes[1, 2].set(title="Y PLS 3")

    fig.tight_layout()

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
paths = fig_paths("pca_vs_pls-regression")

# Only generate it once.
if not all(os.path.exists(path) for path in paths):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

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

    fig.tight_layout()

    for path in paths:
        fig.savefig(path)

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
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
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

    fig.tight_layout()

    for path in paths:
        fig.savefig(path)

r2s = []
for n in range(1, n_max + 1):
    # Semente
    # 1241    200
    # 1242    200
    # 1243    200
    # 1244    200
    # 1245    200
    for seed in range(1241, 1246):
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rng)
        X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y, seed)

        pcr = PCR(n_components=n).fit(X_train, Y_train)

        plsr = PLSRegression(n_components=n).fit(X_train, Y_train)

        r2_pcr = pcr.score(X_test, Y_test)
        r2_plsr = plsr.score(X_test, Y_test)

        r2s.append({"n": n, "seed": seed, "algo": "PCA", "r2": r2_pcr})
        r2s.append({"n": n, "seed": seed, "algo": "PLS", "r2": r2_plsr})

        # TODO: avoid aggregation outside pandas!
        # e.g., mean values over all seeds.

r2s_df = pd.DataFrame(r2s)

print("\nPCA vs. PLS\n===========")
print("mean over all five seeds.")
for n in range(1, n_max + 1):
    print("\nn =", n)
    r2s_df_n = r2s_df[r2s_df["n"] == n]

    print(
        f"PCR R-squared {r2s_df_n[r2s_df_n["algo"] == "PCA"]["r2"].mean():.3f}")
    print(
        f"PLS R-squared {r2s_df_n[r2s_df_n["algo"] == "PLS"]["r2"].mean():.3f}")

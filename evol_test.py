import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

from evol import Evol, UV_EVOL
from model import load_leme, train_test_seed_split
from util import fig_paths

X, Y = load_leme()

X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y, seed=1241)

# === Evol vs. PLSR ===

n_evol = 3

plsr = PLSRegression(n_components=n_evol).fit(X_train, Y_train)

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
print(f"\nregular PLS (n == {n_plsr}):",
      round(plsr.score(X_test, Y_test), 3))

evol = Evol(n_components=n_evol).fit(X_train, Y_train, UV_EVOL)

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

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

from evol import Evol, UV_EVOL
from model import load_leme, train_test_seed_split
from plots import plot_predictions
from util import get_globs, get_paths, show_or_save


X, Y = load_leme(idwl=False)

_PAUSE = True
_SHOW = True

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
Y_pred_plsr = plsr.predict(X_test)
r2_plsr = r2_score(Y_test, Y_pred_plsr)
print(f"\nregular PLS (n == {n_plsr}):", round(r2_plsr, 3))

evol = Evol(n_components=n_evol).fit(X_train, Y_train, UV_EVOL)

# np.dot(X.T, Y) ** 2: squared cross-covariance matrix.
print("\ncross covariance\n----------------")
for name, model in [("plsr", plsr), ("evol", evol)]:
    C2 = np.dot(model.x_scores_.T, model.y_scores_) ** 2

    print(f"diag(C_{name}^2) = {np.round(np.diag(C2), 3)}")
    print(f"|C_{name}^2| = {np.linalg.norm(C2):.3f}")

X_test_evol, Y_test_evol = (pd.DataFrame(t)
                            for t in evol.transform(X_test, Y_test))

Y_pred_evol = pd.DataFrame(evol.predict(X_test), columns=Y_train.columns)
_, Y_pred_evol_t = evol.transform(X_test, Y_pred_evol)
Y_pred_evol_t = pd.DataFrame(Y_pred_evol_t)

ords = ("1st", "2nd", "3rd")

R2_Y_evol_t = r2_score(Y_test_evol, Y_pred_evol_t, multioutput="raw_values")

# (200 x 5) @ (5 x 1)
evol_predictions = {
    "X": X_test_evol,
    "Y_true": Y_test_evol,
    "Y_pred": Y_pred_evol_t,
    "xlabels": [f"X's Evol {i}" for i in range(1, n_evol + 1)],
    "ylabels": [f"Y's Evol {i}" for i in range(1, n_evol + 1)],
    "R2": R2_Y_evol_t,
    "ncols": 3,
}

path = "evol-predictions"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
show_or_save(paths, globs, plot_predictions, _SHOW, _PAUSE,
             **evol_predictions)

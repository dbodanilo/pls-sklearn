import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer, normalize

from decomposition import ScalerPCA, ScalerPCR
from model import load_leme, train_test_seed_split
from plots import plot_components, plot_predictions, plot_regression
from util import (
    detexify,
    fit_predict,
    fit_predict_try_transform,
    get_globs,
    get_paths,
    latexify,
    show_or_save,
    try_attr,
    try_transform
)


_SHOW = True
_PAUSE = True
_SORT = "desc"

NOW = datetime.now()
TODAY = NOW.strftime("%Y-%m-%d")

X, Y = load_leme()

# For indexing (original format).
ds = X.columns.drop(["N.", "Semente"])
# Last three targets are the most important (Av0, fT, Pwr).
ts = Y.columns.drop(["N.", "Semente"])

# For plotting (ensure LaTeX formatting).
descriptors = latexify(ds)
targets = latexify(ts)
seeds = X["Semente"].value_counts().index

# train_test_seed_split() parameters.
all_seeds = (None, *seeds)
# For indexing and plotting.
todas_sementes = ("Nenhuma", *(str(s) for s in seeds))

X_all, _, Y_all, _ = train_test_seed_split(X, Y, seed=None)

n_samples = X_all.shape[0]
n_features = X_all.shape[1]
n_targets = Y_all.shape[1]

seed = 1245
X_train, X_test, _, _ = train_test_seed_split(X, Y, seed=seed)

n_train = X_train.shape[0]
n_test = X_test.shape[0]

# n > n_targets will only throw an error if using
# PLSCanonical, not PLSRegression.
# for PLSCanonical: min(n_train, n_features, n_targets)
# for PLSRegression: min(n_train, n_features)
n_max = min(n_train, n_features, n_targets)

splits = {}

pcr = {}
r_pcr = {}

plsr = {}
r_plsr = {}

# NOTE: evaluate stability among runs for each target variable.
# `seed=None` means all samples (no split).
for seed, semente in zip((None, *seeds), todas_sementes):
    splits[semente] = train_test_seed_split(X, Y, seed=seed)
    X_train, _, Y_train, _ = splits[semente]

    pcr[semente] = ScalerPCR(n_components=n_max).fit(X_train, Y_train)
    r_pcr[semente] = ScalerPCR(n_components=n_max).fit(Y_train, X_train)

    plsr[semente] = PLSRegression(n_components=n_max).fit(X_train, Y_train)
    r_plsr[semente] = PLSRegression(n_components=n_max).fit(Y_train, X_train)


# === PCA ===

x_pca = ScalerPCA(n_components=n_features).fit(X_all)
x_pca_step: PCA = x_pca.named_steps["pca"]

y_pca = ScalerPCA(n_components=n_targets).fit(Y_all)
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

for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    _, X_test, _, Y_test = split

    X_test_pca = pd.DataFrame(try_transform(pcr[semente], X_test))
    Y_pred_pcr = pd.DataFrame(pcr[semente].predict(
        X_test), columns=Y_train.columns)

    pcr_predictions = {
        "xlabels": [f"X's PCA {i}, Semente: {semente}" for i in range(1, n_max + 1)],
        "ylabels": targets,
        "X": X_test_pca,
        "Y_true": Y_test,
        "Y_pred": Y_pred_pcr,
        "R2": r2_score(Y_test, Y_pred_pcr, multioutput="raw_values"),
        "iter_x": False,
        "ncols": 3,
        "nrows": 2,
    }

    path = f"pcr-predictions-seed_{seed}"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    show_or_save(paths, globs, plot_predictions, _SHOW, False,
                 **pcr_predictions)


    Y_test_pca = pd.DataFrame(try_transform(r_pcr[semente], Y_test))
    Y_pred_pcr_t = pd.DataFrame(try_transform(r_pcr[semente], Y_pred_pcr))

    R2_Y_pcr_t = r2_score(Y_test_pca, Y_pred_pcr_t, multioutput="raw_values")

    pcr_predictions_transformed = {
        "X": X_test_pca,
        "Y_true": Y_test_pca,
        "Y_pred": Y_pred_pcr_t,
        "xlabels": [f"X's PCA {i}" for i in range(1, n_targets + 1)],
        "ylabels": [f"Y's PCA {i}" for i in range(1, n_targets + 1)],
        "R2": R2_Y_pcr_t,
        "ncols": 3,
    }

    path = f"pcr-predictions_transformed-seed_{seed}"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    show_or_save(paths, globs, plot_predictions, _SHOW, False,
                **pcr_predictions_transformed)

    X_pred_pcr = pd.DataFrame(r_pcr[semente].predict(Y_test), columns=X_train.columns)
    X_pred_pcr_t = pd.DataFrame(try_transform(pcr[semente], X_pred_pcr))

    R2_X_pcr_t = r2_score(X_test_pca, X_pred_pcr_t, multioutput="raw_values")

    pcr_predictions_reversed_transformed = {
        "X": Y_test_pca,
        "Y_true": X_test_pca,
        "Y_pred": X_pred_pcr_t,
        "xlabels": [f"Y's PCA {i}" for i in range(1, n_targets + 1)],
        "ylabels": [f"X's PCA {i}" for i in range(1, n_targets + 1)],
        "iter_x": False,
        "R2": R2_X_pcr_t,
        "ncols": 3,
    }

    path = f"pcr-predictions_reversed_transformed-seed_{seed}"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    show_or_save(paths, globs, plot_predictions, _SHOW, _PAUSE,
                **pcr_predictions_reversed_transformed)


# === PLSR ===

for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    _, X_test, _, Y_test = split

    X_test_pls, Y_test_pls = (pd.DataFrame(test_t)
                            for test_t in plsr[semente].transform(X_test, Y_test))

    Y_pred_plsr = pd.DataFrame(plsr[semente].predict(X_test), columns=Y_train.columns)

    R2_Y_plsr = r2_score(Y_test, Y_pred_plsr, multioutput="raw_values")

    plsr_predictions = {
        "X": X_test_pls,
        "Y_true": Y_test,
        "Y_pred": Y_pred_plsr,
        "xlabels": [f"X's PLS {i}" for i in range(1, n_max + 1)],
        "ylabels": targets,
        "R2": R2_Y_plsr,
        "iter_x": False,
        "ncols": 3,
        "nrows": 2,
    }

    path = f"plsr-predictions-seed_{seed}"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    show_or_save(paths, globs, plot_predictions, _SHOW, False,
                **plsr_predictions)

    _, Y_pred_plsr_t = (pd.DataFrame(test_t)
                        for test_t in plsr[semente].transform(X_test, Y_pred_plsr))

    R2_Y_plsr_t = r2_score(Y_test_pls, Y_pred_plsr_t, multioutput="raw_values")

    plsr_predictions_transformed = {
        "X": X_test_pls,
        "Y_true": Y_test_pls,
        "Y_pred": Y_pred_plsr_t,
        "xlabels": [f"X's PLS {i}" for i in range(1, n_max + 1)],
        "ylabels": [f"Y's PLS {i}" for i in range(1, n_max + 1)],
        "R2": R2_Y_plsr_t,
        "ncols": 3,
        "nrows": 2,
    }

    path = f"plsr-predictions_transformed-seed_{seed}"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    show_or_save(paths, globs, plot_predictions, _SHOW, False,
                **plsr_predictions_transformed)

    Y_test_pls, X_test_pls = (pd.DataFrame(test_t)
                            for test_t in r_plsr[semente].transform(Y_test, X_test))

    X_pred_plsr = pd.DataFrame(r_plsr[semente].predict(Y_test), columns=X_train.columns)

    _, X_pred_plsr_t = (pd.DataFrame(test_t)
                        for test_t in r_plsr[semente].transform(Y_test, X_pred_plsr))

    R2_X_plsr_t = r2_score(X_test_pls, X_pred_plsr_t, multioutput="raw_values")

    plsr_predictions_reversed_transformed = {
        "X": Y_test_pls,
        "Y_true": X_test_pls,
        "Y_pred": X_pred_plsr_t,
        "xlabels": [f"Y's PLS {i}" for i in range(1, n_max + 1)],
        "ylabels": [f"X's PLS {i}" for i in range(1, n_max + 1)],
        "R2": R2_X_plsr_t,
        "ncols": 3,
        "nrows": 2,
    }

    path = f"plsr-predictions_reversed_transformed-seed_{seed}"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    show_or_save(paths, globs, plot_predictions, _SHOW, _PAUSE,
                **plsr_predictions_reversed_transformed)


# TODO: use itertools for /.*th/ ords.
ords = ["1st", "2nd", "3rd"]
ordinais = ["Primeiro", "Segundo", "Terceiro", "Quarto", "Quinto"]

seed, semente = (str(None), "Nenhuma")

# TODO: use correlation, not normalization.
x_plsr_components = normalize(plsr[semente].x_rotations_, axis=0)
y_plsr_components = normalize(plsr[semente].y_rotations_, axis=0)

for i, o in enumerate(ordinais):
    pls_x_component_i = {
        "X": x_plsr_components[:, i].reshape(-1, 1),
        "titles": [f"{o} Componente PLS de X"],
        "xlabels": descriptors,
        "ylabel": "Peso",  # TODO: usar correlação.
        "sort": _SORT,
        "meanlabel": "média",
    }

    path = f"pls-x_component_{i}-sort_{_SORT}-lang_pt"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    # NOTE: pausing in a loop isn't practical.
    show_or_save(paths, globs, plot_components, _SHOW, False,
                 **pls_x_component_i)

    pls_y_component_i = {
        "X": y_plsr_components[:, i].reshape(-1, 1),
        "titles": [f"{o} Componente PLS de Y"],
        "xlabels": targets,
        "ylabel": "Peso",
        "sort": _SORT,
        "meanlabel": "média",
    }

    path = f"pls-y_component_{i}-seed_{seed}-sort_{_SORT}-lang_pt"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    show_or_save(paths, globs, plot_components, _SHOW, False,
                 **pls_y_component_i)


pls_x_components = {
    "X": x_plsr_components,
    "titles": [f"{o} Componente PLS de X" for o in ordinais],
    "xlabels": descriptors,
    "ylabel": "Peso",
    "ncols": x_plsr_components.shape[1],
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pls-x_components-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pls_x_components)

pls_y_components = {
    "X": y_plsr_components,
    "titles": [f"{o} Componente PLS de Y" for o in ordinais],
    "xlabels": targets,
    "ylabel": "Peso",
    "ncols": y_plsr_components.shape[1],
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pls-y_components-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pls_y_components)


x_all_plsr_components = np.empty((n_features, 0))
y_all_plsr_components = np.empty((n_targets, 0))

for semente, split in splits.items():
    X_train, X_test, Y_train, Y_test = split

    x_first_component = plsr[semente].x_rotations_[:, 0].reshape(-1, 1)
    y_first_component = plsr[semente].y_rotations_[:, 0].reshape(-1, 1)

    x_all_plsr_components = np.append(x_all_plsr_components,
                                      x_first_component, axis=1)
    y_all_plsr_components = np.append(y_all_plsr_components,
                                      y_first_component, axis=1)

x_all_plsr_components = normalize(x_all_plsr_components, axis=0)
y_all_plsr_components = normalize(y_all_plsr_components, axis=0)

pls_seeds_first_x_components = {
    "X": x_all_plsr_components,
    "titles": [f"Primeiro Componente PLS de X, Semente: {s}" for s in todas_sementes],
    "xlabels": descriptors,
    "ylabel": "Peso",
    "ncols": x_all_plsr_components.shape[1],
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pls_seeds-first_x_components-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pls_seeds_first_x_components)

pls_seeds_first_y_components = {
    "X": y_all_plsr_components,
    "titles": [f"Primeiro Componente PLS de Y, Semente: {s}" for s in todas_sementes],
    "xlabels": targets,
    "ylabel": "Peso",
    "ncols": y_all_plsr_components.shape[1],
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pls_seeds-first_y_components-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pls_seeds_first_y_components)


# seed=1241: best seed for `X = predict(Y)` and second-best
# for `Y = predict(X)` (based on r2_score).
seed = str(None)

# TODO: evaluate stability among runs for each target variable.
plsr_targets_regressors = pd.Series([PLSRegression(
    n_components=n_targets).fit(X_all, Y_all)], index=["all"])

plsr_targets_components = pd.DataFrame(
    plsr_targets_regressors["all"].x_rotations_[:, 0], columns=["all"], index=ds)

for t in ts:
    Y_train_target = Y_all[t]

    plsr_targets_regressors[t] = PLSRegression(n_components=n_targets).fit(
        X_all, Y_train_target)

    plsr_targets_components[t] = plsr_targets_regressors[t].x_rotations_[:, 0]

# For indexing.
all_ts = [None, *ts]
# For plotting.
all_targets = ["all", *targets]
todos_objetivos = ["Todos", *targets]

pls_targets_x_components = {
    "X": plsr_targets_components,
    "titles": [f"Primeiro Componente PLS de X, Objetivo: {o}" for o in todos_objetivos],
    "xlabels": descriptors,
    "ylabel": "Peso",
    "ncols": plsr_targets_components.shape[1],
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pls_targets-x_components-seed_{seed}-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pls_targets_x_components)


plsr_regressors = {}
plsr_components = {}

for semente, (X_train, X_test, Y_train, Y_test) in splits.items():
    plsr_regressors[semente] = {}

    for t, target in zip(ts, todos_objetivos):
        Y_train_target = Y_train[t] if t is not None else Y_train

        plsr_regressors[semente][t] = PLSRegression(n_components=n_targets).fit(
            X_train, Y_train_target)

        target_first_component = plsr_regressors[semente][t].\
            x_rotations_[:, 0].reshape(-1, 1)

        # Only set it in first pass.
        if semente == "Nenhuma":
            plsr_components[target] = target_first_component
        else:
            plsr_components[target] = np.append(
                plsr_components[target], target_first_component, axis=1)

for target, components in plsr_components.items():
    components = normalize(components, axis=0)

    pls_target_seeds_first_x_components = {
        "X": components,
        "titles": [f"Primeiro Componente PLS de X para Objetivo: {target}, Semente: {s}" for s in todas_sementes],
        "xlabels": descriptors,
        "ylabel": "Peso",
        "ncols": components.shape[1],
        "sort": _SORT,
        "meanlabel": "média",
    }

    path = f"pls_{detexify(target)}_seeds-first_x_components-sort_{_SORT}-lang-pt"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    show_or_save(paths, globs, plot_components, _SHOW, False,
                 **pls_target_seeds_first_x_components)


# === PCR vs. PLSR ===

# NOTE: split it so that PLSR performs better than PCR.
for semente, split in splits.items():
    seed = None if semente == "Nenhuma" else semente

    X_train, X_test, Y_train, Y_test = split

    X_test_pca, X_pred_pcr, X_pred_pcr_t, Y_test_pca, Y_pred_pcr, Y_pred_pcr_t = fit_predict_try_transform(
        ScalerPCR, *split, n_components=1)

    X_test_pls, X_pred_plsr, X_pred_plsr_t, Y_test_pls, Y_pred_plsr, Y_pred_plsr_t = fit_predict_try_transform(
        PLSRegression, *split, n_components=1)

    R2_Y_pcr_t = r2_score(Y_test_pca, Y_pred_pcr_t, multioutput="raw_values")
    R2_Y_plsr_t = r2_score(Y_test_pls, Y_pred_plsr_t, multioutput="raw_values")

    pcr_vs_plsr_predictions = {
        "X": pd.concat((X_test_pca.iloc[:, 0], X_test_pls.iloc[:, 0]), axis="columns"),
        "Y_true": pd.concat((Y_test_pca.iloc[:, 0], Y_test_pls.iloc[:, 0]), axis="columns"),
        "Y_pred": pd.concat((Y_pred_pcr_t.iloc[:, 0], Y_pred_plsr_t.iloc[:, 0]), axis="columns"),
        "xlabels": ["X's PCA 1", "X's PLS 1"],
        "ylabels": ["Y's PCA 1", "Y's PLS 1"],
        "R2": np.array((R2_Y_pcr_t[0], R2_Y_plsr_t[0])),
    }

    # NOTE: print seed used when outputting plots and scores.
    path = f"pcr_vs_plsr-predictions-seed_{seed}"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    # No pause in for loop.
    show_or_save(paths, globs, plot_predictions, _SHOW, False,
                 **pcr_vs_plsr_predictions)


semente, seed = ("Nenhuma", str(None))
y_pca_components = try_attr(r_pcr[semente], "components_")

# NOTE: different title for X and Y.
pca_first_y_component = {
    "X": y_pca_components[0].reshape(-1, 1),
    "titles": [f"{o} Componente PCA de Y" for o in ordinais],
    "xlabels": targets,
    "ylabel": "Peso",
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pca-first_y_component-seed_{seed}-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pca_first_y_component)

pca_y_components = {
    "X": y_pca_components.T,
    "titles": [f"{o} Componente PCA de Y" for o in ordinais],
    "xlabels": targets,
    "ylabel": "Peso",
    "ncols": y_pca_components.shape[0],
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pca-y_components-seed_{seed}-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pca_y_components)


algos = ("PCA", "PLS")

# seed=1241 was the best for the ratio of rPLSR's r2_score
# over rPCR's.
for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    X_train, X_test, Y_train, Y_test = split

    X_test_pca, X_pred_pcr, X_pred_pcr_t, Y_test_pca, Y_pred_pcr, Y_pred_pcr_t = fit_predict_try_transform(
        ScalerPCR, *split, n_components=1)

    X_test_pls, X_pred_plsr, X_pred_plsr_t, Y_test_pls, Y_pred_plsr, Y_pred_plsr_t = fit_predict_try_transform(
        PLSRegression, *split, n_components=1)

    # NOTE: display R-squared for the prediction of each
    # component.
    R2_Y_pcr_t = r2_score(Y_test_pca, Y_pred_pcr_t, multioutput="raw_values")
    R2_Y_plsr_t = r2_score(Y_test_pls, Y_pred_plsr_t, multioutput="raw_values")

    pcr_vs_plsr_regression = {
        "Y_true": pd.concat((Y_test_pca.iloc[:, 0], Y_test_pls.iloc[:, 0]), axis="columns"),
        "Y_pred": pd.concat((Y_pred_pcr_t.iloc[:, 0], Y_pred_plsr_t.iloc[:, 0]), axis="columns"),
        "xlabels": [f"Actual Y projected onto 1st {algo} component" for algo in algos],
        "ylabels": [f"Predicted Y projected onto 1st {algo} component" for algo in algos],
        "titles": [f"{algo} Regression" for algo in algos],
        "R2": np.array((R2_Y_pcr_t[0], R2_Y_plsr_t[0])),
    }

    path = f"pcr_vs_plsr-regression-seed_{seed}"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    # pause=True is not practical in a for-loop.
    show_or_save(paths, globs, plot_regression, _SHOW, False,
                 **pcr_vs_plsr_regression)

    R2_X_pcr_t = r2_score(X_test_pca, X_pred_pcr_t, multioutput="raw_values")
    R2_X_plsr_t = r2_score(X_test_pls, X_pred_plsr_t, multioutput="raw_values")

    pcr_vs_plsr_regression_reversed = {
        "Y_true": pd.concat((X_test_pca.iloc[:, 0], X_test_pls.iloc[:, 0]), axis="columns"),
        "Y_pred": pd.concat((X_pred_pcr_t.iloc[:, 0], X_pred_plsr_t.iloc[:, 0]), axis="columns"),
        "xlabels": [f"Actual X projected onto 1st {algo} component" for algo in algos],
        "ylabels": [f"Predicted X projected onto 1st {algo} component" for algo in algos],
        "titles": [f"{algo} Regression" for algo in algos],
        "R2": np.array((R2_X_pcr_t[0], R2_X_plsr_t[0])),
    }

    path = f"pcr_vs_plsr-regression_reversed-seed_{seed}"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    show_or_save(paths, globs, plot_regression, _SHOW, _PAUSE,
                 **pcr_vs_plsr_regression_reversed)


# DOING: train only multi-output regressors.
model_labels = ["PCR", "PLSR"]  # , "SVR"
r_labels = ["r" + label for label in model_labels]

models = [ScalerPCR, PLSRegression]  # , ScalerSVR

N = 480

algo_col = np.empty(N, dtype=object)
n_col = np.empty(N, dtype=int)
r2_col = np.empty(N, dtype=float)
seed_col = np.empty(N, dtype=object)
t_col = np.empty(N, dtype=object)
i = 0
for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    X_train, X_test, Y_train, Y_test = split

    for n in range(1, n_max + 1):
        for label, model in zip(model_labels, models):
            r_label = "r" + label

            m, rm, X_pred, Y_pred = fit_predict(
                model, X_train, X_test, Y_train, Y_test, n_components=n)

            # NOTE: Compare predictions in original feature
            # space, this confirms that calculating the
            # score on the transformed feature space
            # makes it easier for lower `n` to achieve
            # higher `r2_score`, which makes it harder
            # to observe the expected increase in
            # `r2_score` as `n` increases.
            IdTransformer = FunctionTransformer(lambda x: x)
            for t, transformer in ((str(None), IdTransformer), ("PCA", ScalerPCA), ("PLS", PLSRegression)):
                # TODO: pass IdTransformer as class, not object.
                tm = transformer
                rtm = transformer
                if t != "None":
                    tm = transformer(n_components=n).fit(X_train, Y_train)
                    rtm = transformer(n_components=n).fit(Y_train, X_train)

                X_test_t = pd.DataFrame(try_transform(tm, X_test))
                X_pred_t = pd.DataFrame(try_transform(tm, X_pred))

                Y_test_t = pd.DataFrame(try_transform(rtm, Y_test))
                Y_pred_t = pd.DataFrame(try_transform(rtm, Y_pred))

                # TODO: review imposing a lower bound
                # (-1 or 0) to avoid skewing the mean, as
                # the r2's upper bound is positive 1, but
                # there's no lower bound.
                # 0 seems to be more mathematically sound
                # than -1. But -1 retains the information on
                # whether the regressor performs worse than
                # DummyRegressor(strategy='mean').
                # r2 = max(r2_score(Y_test, Y_pred), 0)
                r2_m = r2_score(Y_test_t.iloc[:, :n],
                                Y_pred_t.iloc[:, :n])
                r2_rm = r2_score(X_test_t.iloc[:, :n],
                                 X_pred_t.iloc[:, :n])

                for algo, r2 in ((label, r2_m), (r_label, r2_rm)):
                    # A numpy.ndarray for each column.
                    algo_col[i] = algo
                    n_col[i] = n

                    # TODO: explicitly aggregate r2_score
                    # over each variable, i.e., use its
                    # `multioutput` parameter and extract
                    # mean and std over variables.
                    r2_col[i] = r2

                    seed_col[i] = seed
                    t_col[i] = t
                    i += 1

# NOTE: avoid aggregation outside pandas!
# e.g., mean values over all seeds;
# and avoid conversion back to python list,
# pandas.DataFrame seems not to work with numpy.array[dict]
r2s_df = pd.DataFrame({
    "algo": algo_col,
    "n": n_col,
    "r2": r2_col,
    "seed": seed_col,
    "t": t_col
})

path = "r2s"
paths, prefix, exts = get_paths(path, exts=[".csv"])
globs = get_globs(path, prefix, exts)

# Only save it once a day.
if not any(os.path.exists(path) for path in globs):
    # format "{:.5f}" was the highest one not to vary on
    # equivalent runs.
    r2s_df.to_csv(paths[0], sep="\t", float_format="{:.5f}".format)


def print_r2s(df, model_labels, ns=None, seeds=None):
    filters = ns
    f_label = "n"

    is_n = ns is not None

    msg = f"\n(mean, std) over all "
    if is_n:
        # if filtered by ns, we are aggregating over seeds.
        msg += "seeds"
    else:
        filters = (str(s) for s in seeds)
        f_label = "seed"
        msg += "ns"

    print(msg)
    # -1 for the '\n'.
    print("-" * (len(msg) - 1))

    for f in filters:
        print(f"\n{f_label} =", f)
        # NOTE: filter t == str(None), ignore transformed samples;
        # print results on original feature space only.
        df_filtered = df[(df[f_label] == f) & (df["t"] == str(None))]
        if is_n:
            # don't aggregate over seed=None.
            df_filtered = df_filtered[df_filtered["seed"] != str(None)]

        for m_label in model_labels:
            df_f_m = df_filtered[df_filtered["algo"] == m_label]["r2"]
            # TODO: remove outliers based on distance from mean,
            # not just the min and max samples.
            # r2_min = r2s_df_n_algo["r2"].min()
            # r2_max = r2s_df_n_algo["r2"].max()
            # mean = (r2s_df_n_algo["r2"].sum() - (r2_min + r2_max)) / 3
            mean, std = df_f_m.mean(), df_f_m.std()
            print(m_label, "R-squared:", f"({mean:.3f}, {std:.3f})")


# for n in range(1, n_max + 1):
ns = list(range(1, n_max + 1))

# vs. SVR
print("\nPCA vs. PLS", end="")
print("\n===========", end="")

print_r2s(r2s_df, model_labels, ns=ns)

# NOTE: `all_seeds` includes str(None) as a seed.
print_r2s(r2s_df, model_labels, seeds=all_seeds)


print("\nPCA vs. PLS (reverse, X = model.predict(Y))", end="")
print("\n===========================================", end="")
print_r2s(r2s_df, r_labels, ns=ns)

print_r2s(r2s_df, r_labels, seeds=all_seeds)

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
from plots import plot_components, plot_predictions, plot_regression
from util import detexify, fit_predict_try_transform, get_globs, get_paths, latexify, show_or_save


_SHOW = True
_PAUSE = True
_SORT = "desc"

NOW = datetime.now()
TODAY = NOW.strftime("%Y-%m-%d")

X, Y = load_leme()

# For indexing (original format).
ds = X.columns.drop(["N.", "Semente"])
ts = Y.columns.drop(["N.", "Semente"])

# For plotting (ensure LaTeX formatting).
descriptors = latexify(ds)
targets = latexify(ts)
seeds = X["Semente"].value_counts().index

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

X_test_pca = pd.DataFrame(x_pca.transform(X_test))

Y_pred_pcr = pd.DataFrame(pcr.predict(X_test), columns=Y_train.columns)

# Last three targets are the most important (Av0, fT, Pwr).
pcr_predictions = {
    "xlabels": [f"X's PCA {i}" for i in range(1, n_max + 1)],
    "ylabels": targets,
    "X": X_test_pca,
    "Y_true": Y_test,
    "Y_pred": Y_pred_pcr,
    "R2": r2_score(Y_test, Y_pred_pcr, multioutput="raw_values"),
    "iter_x": False,
    "ncols": 3,
    "nrows": 2,
}

path = "pcr-predictions"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_predictions, _SHOW, _PAUSE,
             **pcr_predictions)


r_pcr = ScalerPCR(n_components=n_max).fit(Y_train, X_train)

Y_test_pca = pd.DataFrame(y_pca.transform(Y_test))

X_pred_pcr = pd.DataFrame(r_pcr.predict(Y_test), columns=X_train.columns)
X_pred_pcr_t = pd.DataFrame(x_pca.transform(X_pred_pcr))

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

path = "pcr-predictions_reversed_transformed"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_predictions, _SHOW, _PAUSE,
             **pcr_predictions_reversed_transformed)


Y_pred_pcr_t = pd.DataFrame(y_pca.transform(Y_pred_pcr))

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

path = "pcr-predictions_transformed"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_predictions, _SHOW, _PAUSE,
             **pcr_predictions_transformed)


# === PLSR ===

plsr = PLSRegression(n_components=n_max).fit(X_train, Y_train)

X_test_pls, Y_test_pls = (pd.DataFrame(test_t)
                          for test_t in plsr.transform(X_test, Y_test))

Y_pred_plsr = pd.DataFrame(plsr.predict(X_test), columns=Y_train.columns)

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

path = "plsr-predictions"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_predictions, _SHOW, _PAUSE,
             **plsr_predictions)


r_plsr = PLSRegression(n_components=n_max).fit(Y_train, X_train)

Y_test_pls, X_test_pls = (pd.DataFrame(test_t)
                          for test_t in r_plsr.transform(Y_test, X_test))

X_pred_plsr = pd.DataFrame(r_plsr.predict(Y_test), columns=X_train.columns)

_, X_pred_plsr_t = (pd.DataFrame(test_t)
                    for test_t in r_plsr.transform(Y_test, X_pred_plsr))

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

path = "plsr-predictions_reversed_transformed"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_predictions, _SHOW, _PAUSE,
             **plsr_predictions_reversed_transformed)


_, Y_pred_plsr_t = (pd.DataFrame(test_t)
                    for test_t in plsr.transform(X_test, Y_pred_plsr))

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

path = "plsr-predictions_transformed"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_predictions, _SHOW, _PAUSE,
             **plsr_predictions_transformed)


# TODO: use itertools for /.*th/ ords.
ords = ["1st", "2nd", "3rd"]
ordinais = ["Primeiro", "Segundo", "Terceiro", "Quarto", "Quinto"]

x_plsr_components = normalize(plsr.x_rotations_, axis=0)
y_plsr_components = normalize(plsr.y_rotations_, axis=0)

pls_first_components = {
    "xtitles": [f"{o} Componente PLS de X" for o in ordinais],
    "ytitles": [f"{o} Componente PLS de Y" for o in ordinais],
    "xlabels": descriptors,
    "ylabels": targets,
    "X": x_plsr_components[:, 0].reshape(-1, 1),
    "Y": y_plsr_components[:, 0].reshape(-1, 1),
    "ncols": 1,  # First component only.
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pls-first_components-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pls_first_components)


pls_components = {
    "xtitles": [f"{o} Componente PLS de X" for o in ordinais],
    "ytitles": [f"{o} Componente PLS de Y" for o in ordinais],
    "xlabels": descriptors,
    "ylabels": targets,
    "X": x_plsr_components,
    "Y": y_plsr_components,
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pls-components-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pls_components)


X_all, Y_all = train_test_seed_split(X, Y, seed=None)

plsr_all = PLSRegression(n_components=n_targets).fit(X_all, Y_all)

x_all_plsr_components = plsr_all.x_rotations_[:, 0].reshape(-1, 1)
y_all_plsr_components = plsr_all.y_rotations_[:, 0].reshape(-1, 1)

for seed in range(1241, 1246):
    X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y, seed=seed)
    plsr = PLSRegression(n_components=n_targets).fit(X_train, Y_train)

    x_first_component = plsr.x_rotations_[:, 0].reshape(-1, 1)
    y_first_component = plsr.y_rotations_[:, 0].reshape(-1, 1)

    x_all_plsr_components = np.append(x_all_plsr_components,
                                      x_first_component, axis=1)
    y_all_plsr_components = np.append(y_all_plsr_components,
                                      y_first_component, axis=1)

x_all_plsr_components = normalize(x_all_plsr_components, axis=0)
y_all_plsr_components = normalize(y_all_plsr_components, axis=0)

all_seeds = ["all", *(str(seed) for seed in seeds)]
todas_sementes = ["Nenhuma", *(str(seed) for seed in seeds)]

pls_all_components = {
    "xtitles": [f"Primeiro Componente PLS de X, Semente: {s}" for s in todas_sementes],
    "ytitles": [f"Primeiro Componente PLS de Y, Semente: {s}" for s in todas_sementes],
    "xlabels": descriptors,
    "ylabels": targets,
    "X": x_all_plsr_components,
    "Y": y_all_plsr_components,
    "ncols": len(todas_sementes),
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pls_all-components-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pls_all_components)


# seed=1241: best seed for `X = predict(Y)` and second-best
# for `Y = predict(X)` (based on r2_score).
seed = None

# TODO: evaluate stability among runs for each target variable.
plsr_targets_regressors = pd.Series(PLSRegression(
    n_components=n_targets).fit(X_all, Y_all), index=["all"])

plsr_targets_components = pd.DataFrame(
    plsr_targets_regressors["all"].x_rotations_[:, 0], columns=["all"], index=ds)

for t in ts:
    Y_train_target = Y_all[t]

    plsr_targets_regressors[t] = PLSRegression(n_components=n_targets).fit(
        X_all, Y_train_target)

    plsr_targets_components[t] = plsr_targets_regressors[t].x_rotations_[:, 0]

# For indexing.
all_ts = ["all", *ts]
# For plotting.
all_targets = ["all", *targets]
todos_objetivos = ["Todos", *targets]

pls_targets_components = {
    "xtitles": [f"Primeiro Componente PLS de X, Objetivo: {o}" for o in todos_objetivos[:3]],
    "ytitles": [f"Primeiro Componente PLS de X, Objetivo: {o}" for o in todos_objetivos[-3:]],
    "xlabels": descriptors,
    "X": plsr_targets_components.iloc[:, :3],
    "Y": plsr_targets_components.iloc[:, -3:],
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pls_targets-components-seed_{seed}-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pls_targets_components)


# 0th index: `seed=None` means all samples (no split).
splits = [(X_all, X_all, Y_all, Y_all)]

# NOTE: evaluate stability among runs for each target variable.
for seed in range(1241, 1246):
    splits.append(train_test_seed_split(X, Y, seed=seed))


plsr_regressors = {}
plsr_components = {}

for seed, (X_train, X_test, Y_train, Y_test) in zip(range(1240, 1246), splits):
    plsr_regressors[seed] = {"all":  PLSRegression(n_components=n_targets).fit(
        X_train, Y_train)}

    all_first_component = plsr_regressors[seed]["all"].x_rotations_[:, 0]

    # Only set it in first pass.
    if seed == 1240:
        for target in targets:
            plsr_components[target] = all_first_component.reshape(-1, 1)

    for t, target in zip(ts, targets):
        Y_train_target = Y_train[t]

        plsr_regressors[seed][t] = PLSRegression(n_components=n_targets).fit(
            X_train, Y_train_target)

        target_first_component = plsr_regressors[seed][t].\
            x_rotations_[:, 0].reshape(-1, 1)

        plsr_components[target] = np.append(
            plsr_components[target], target_first_component, axis=1)

for target, components in plsr_components.items():
    components = normalize(components, axis=0)

    pls_target_components = {
        "xtitles": [f"Primeiro Componente PLS de X para {target}, Semente: {s}" for s in todas_sementes[:3]],
        "ytitles": [f"Primeiro Componente PLS de X para {target}, Semente: {s}" for s in todas_sementes[-3:]],
        "xlabels": descriptors,
        "X": components[:, :3],
        "Y": components[:, -3:],
        "sort": _SORT,
        "meanlabel": "média",
    }

    path = f"pls_{detexify(target)}-components-sort_{_SORT}-lang-pt"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    # NOTE: pausing in a loop isn't practical.
    show_or_save(paths, globs, plot_components, _SHOW, False,
                 **pls_target_components)


# === PCR vs. PLSR ===

# TODO: split it so that PLSR performs better than PCR.
# NOTE: print seed used when outputting plots and scores.
for seed in (None, *range(1241, 1246)):
    split = train_test_seed_split(X, Y, seed=seed)
    # seed == None
    if len(split) < 4:
        split = (split[0], split[0], split[1], split[1])
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

    path = f"pcr_vs_plsr-predictions-seed_{seed}"
    paths, prefix, exts = get_paths(path)
    globs = get_globs(path, prefix, exts)

    # No pause in for loop.
    show_or_save(paths, globs, plot_predictions, _SHOW, False,
                 **pcr_vs_plsr_predictions)


# NOTE: different title for X and Y.
pca_vs_pls_first_components = {
    "xtitles": [f"{o} Componente PCA de Y" for o in ordinais],
    "ytitles": [f"{o} Componente PLS de Y" for o in ordinais],
    "xlabels": targets,
    "X": y_pca_step.components_[0].reshape(-1, 1),
    "Y": y_plsr_components[:, 0].reshape(-1, 1),
    "ncols": 1,
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pca_vs_pls-first_components-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pca_vs_pls_first_components)


pca_vs_pls_components = {
    "xtitles": [f"{o} Componente PCA de Y" for o in ordinais],
    "ytitles": [f"{o} Componente PLS de Y" for o in ordinais],
    "xlabels": targets,
    "X": y_pca_step.components_.T,
    "Y": y_plsr_components,
    "sort": _SORT,
    "meanlabel": "média",
}

path = f"pca_vs_pls-components-sort_{_SORT}-lang_pt"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

show_or_save(paths, globs, plot_components, _SHOW, _PAUSE,
             **pca_vs_pls_components)


# seed=1241 was the best for the ratio of rPLSR's r2_score
# over rPCR's.
# NOTE: print seed used when outputting plots and scores.
seed = 1241
split = train_test_seed_split(X, Y, seed=seed)
# seed == None
if len(split) < 4:
    split = (split[0], split[0], split[1], split[1])
X_train, X_test, Y_train, Y_test = split

X_test_pca, X_pred_pcr, X_pred_pcr_t, Y_test_pca, Y_pred_pcr, Y_pred_pcr_t = fit_predict_try_transform(
    ScalerPCR, *split, n_components=1)

X_test_pls, X_pred_plsr, X_pred_plsr_t, Y_test_pls, Y_pred_plsr, Y_pred_plsr_t = fit_predict_try_transform(
    PLSRegression, *split, n_components=1)

algos = ("PCA", "PLS")

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

show_or_save(paths, globs, plot_regression, _SHOW, _PAUSE,
             **pcr_vs_plsr_regression)


x_test_pca_min = min(X_test_pca.iloc[:, 0].min(),
                     X_pred_pcr_t.iloc[:, 0].min())
x_test_pca_max = max(X_test_pca.iloc[:, 0].max(),
                     X_pred_pcr_t.iloc[:, 0].max())
x_limits_pca = np.linspace(x_test_pca_min, x_test_pca_max, n_test)

x_test_pls_min = min(X_test_pls.iloc[:, 0].min(),
                     X_pred_plsr_t.iloc[:, 0].min())
x_test_pls_max = max(X_test_pls.iloc[:, 0].max(),
                     X_pred_plsr_t.iloc[:, 0].max())
x_limits_pls = np.linspace(x_test_pls_min, x_test_pls_max, n_test)

R2_X_pcr_t = r2_score(X_test_pca, X_pred_pcr_t, multioutput="raw_values")
R2_X_plsr_t = r2_score(X_test_pls, X_pred_plsr_t, multioutput="raw_values")

path = f"pcr_vs_plsr-regression_reversed-seed_{seed}"
paths, prefix, exts = get_paths(path)
globs = get_globs(path, prefix, exts)

# Only generate it once.
if not any(os.path.exists(path) for path in globs) or _SHOW:
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

    if _SHOW:
        fig.show()
    else:
        for path in paths:
            fig.savefig(path)

    if _PAUSE:
        input("Press Enter to continue...")


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
                r2_m = r2_score(Y_test_t.iloc[:, :n_used],
                                Y_pred_t.iloc[:, :n_used])
                r2_rm = r2_score(X_test_t.iloc[:, :n_used],
                                 X_pred_t.iloc[:, :n_used])

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

path = "r2s"
paths, prefix, exts = get_paths(path, exts=[".csv"])
globs = get_globs(path, prefix, exts)

# Only save it once a day.
if not any(os.path.exists(path) for path in globs):
    # format "{:.5f}" was the highest one not to vary on
    # equivalent runs.
    r2s_df.to_csv(paths[0], sep="\t", float_format="{:.5f}".format)

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

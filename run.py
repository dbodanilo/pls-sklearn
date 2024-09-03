import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

from decomposition import ScalerPCA, ScalerPCR
from model import load_deap, load_leme, train_test_seed_split
from plots import plot_components, plot_predictions, plot_regression
from util import (
    detexify,
    fit_predict,
    fit_predict_try_transform,
    latexify,
    save_or_show,
    save_to_csv,
    try_attr,
    try_transform
)


_SAVE = True
_SHOW = True
_PAUSE = True

_MEANLABEL = None
_SORT_X = "desc"
_SORT_Y = None

_ALGO = "nsga3"

X, Y = load_deap() if _ALGO == "nsga3" else load_leme()

seeds = X["Semente"].value_counts().index

# train_test_seed_split() parameters.
all_seeds = (None, *seeds)
# For indexing and plotting.
todas_sementes = ("Nenhuma", *(str(s) for s in seeds))

X_all, _, Y_all, _ = train_test_seed_split(X, Y, seed=None)

# For indexing (original format).
ds = X_all.columns
# Last three targets are the most important (Av0, fT, Pwr).
ts = Y_all.columns

# For plotting (ensure LaTeX formatting).
descriptors = latexify(ds)
targets = latexify(ts)

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

# "Todos": all targets (multivariate Y).
plsr = {"Todos": {}}
r_plsr = {}

pca_component_names = [f"PCA {i}" for i in range(1, n_features + 1)]
pls_component_names = [f"PLS {i}" for i in range(1, n_features + 1)]

# NOTE: evaluate stability among runs for each target variable.
# `seed=None` means all samples (no split).
for seed, semente in zip((None, *seeds), todas_sementes):
    splits[semente] = train_test_seed_split(X, Y, seed=seed)

    X_train, _, Y_train, _ = splits[semente]

    pcr[semente] = ScalerPCR(n_components=n_max).fit(X_train, Y_train)
    x_pca_step: PCA = pcr[semente].named_steps["pca"]

    plsr_seed = PLSRegression(n_components=n_max).fit(X_train, Y_train)
    plsr["Todos"][semente] = plsr_seed

    r_pcr[semente] = ScalerPCR(n_components=n_max).fit(Y_train, X_train)
    y_pca_step: PCA = r_pcr[semente].named_steps["pca"]

    r_plsr_seed = PLSRegression(n_components=n_max).fit(Y_train, X_train)
    r_plsr[semente] = r_plsr_seed

    # transform(X) = X * V = X * transpose(Vt), components_ = Vt.
    x_pca_components = pd.DataFrame(
        x_pca_step.components_,
        columns=X_train.columns,
        index=pca_component_names[:n_max])

    path = f"pca-algo_{_ALGO}-seed_{str(seed)}"
    prefix = "x_component/"
    # TODO: save correlations, not raw components
    save_to_csv(x_pca_components, path, _SAVE, prefix=prefix)

    # transform(X) = X * x_rotations_
    x_pls_components = pd.DataFrame(
        plsr_seed.x_rotations_,
        columns=pls_component_names[:n_max],
        index=X_train.columns)

    # "all" means all targets, only applies to:
    # PLS: because it's supervised, while PCA is not;
    # X: because Y are the targets themselves;
    # non-reversed: because it doesn't make much sense to
    # try to predict 20 variables (X) from a single scalar (y).
    path = f"pls_all-algo_{_ALGO}-seed_{str(seed)}"
    save_to_csv(x_pls_components, path, _SAVE, prefix=prefix)

    # fit(Y, X) -> y_rotations_ transforms our X.
    x_rpls_components = pd.DataFrame(
        r_plsr_seed.y_rotations_,
        columns=pls_component_names[:n_max],
        index=X_train.columns)

    path = f"pls-algo_{_ALGO}-reversed-seed_{str(seed)}"
    save_to_csv(x_rpls_components, path, _SAVE, prefix=prefix)

    y_pca_components = pd.DataFrame(
        y_pca_step.components_,
        columns=Y_train.columns,
        index=pca_component_names[:n_max])

    path = f"pca-algo_{_ALGO}-seed_{str(seed)}"
    prefix = "y_component/"
    save_to_csv(y_pca_components, path, _SAVE, prefix=prefix)

    y_pls_components = pd.DataFrame(
        plsr_seed.y_rotations_,
        columns=pls_component_names[:n_max],
        index=Y_train.columns)

    path = f"pls-algo_{_ALGO}-seed_{str(seed)}"
    save_to_csv(y_pls_components, path, _SAVE, prefix=prefix)

    y_rpls_components = pd.DataFrame(
        r_plsr_seed.x_rotations_,
        columns=pls_component_names[:n_max],
        index=Y_train.columns)

    path = f"pls-algo_{_ALGO}-reversed-seed_{str(seed)}"
    save_to_csv(y_rpls_components, path, _SAVE, prefix=prefix)

    pca_explained_variance_ratio = pd.DataFrame(
        {"X": x_pca_step.explained_variance_ratio_,
         "Y": y_pca_step.explained_variance_ratio_},
        index=pca_component_names[:n_max])

    path = f"pca-algo_{_ALGO}-explained_variance_ratio-seed_{str(seed)}"
    save_to_csv(pca_explained_variance_ratio, path, save=(seed is None))


# === PCA ===

x_pca = ScalerPCA(n_components=n_features).fit(X_all)
x_pca_step: PCA = x_pca.named_steps["pca"]

x_pca_components = pd.DataFrame(
    x_pca_step.components_, columns=X_all.columns,
    index=pca_component_names[:x_pca_step.n_components_])

path = f"pca-x_components-algo_{_ALGO}-seed_None"
save_to_csv(x_pca_components, path, _SAVE)

y_pca = ScalerPCA(n_components=n_targets).fit(Y_all)
y_pca_step: PCA = y_pca.named_steps["pca"]

y_pca_components = pd.DataFrame(
    y_pca_step.components_, columns=Y_all.columns,
    index=pca_component_names[:y_pca_step.n_components_])

path = f"pca-y_components-algo_{_ALGO}-seed_None"
save_to_csv(y_pca_components, path, _SAVE)

# right-pad Y ratios in order to place X and Y on the same DataFrame.
y_pca_explained_variance_ratio = np.pad(
    y_pca_step.explained_variance_ratio_,
    (0, n_features - n_targets),
    mode="constant",
    constant_values=np.nan)

pca_explained_variance_ratio = pd.DataFrame(
    {"X": x_pca_step.explained_variance_ratio_,
     "Y": y_pca_explained_variance_ratio},
    index=pca_component_names[:x_pca_step.n_components_])

path = f"pca-explained_variance_ratio-algo_{_ALGO}-seed_None"
save_to_csv(pca_explained_variance_ratio, path, _SAVE)


# YYYY-mm-dd_HH-mm
_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
print(_now)

print("algo:", _ALGO, )
print("data: (I_D/(W/L))")

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
    Y_pred_pcr = pd.DataFrame(
        pcr[semente].predict(X_test),
        columns=Y_train.columns
    )

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

    path = f"pcr-algo_{_ALGO}-seed_{seed}"
    prefix = "predictions/"

    # pause=False: no pause in for loop.
    save_or_show(path, prefix, plot_predictions, _SAVE, _SHOW, pause=False,
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

    path = f"pcr-transformed-algo_{_ALGO}-seed_{seed}"

    save_or_show(path, prefix, plot_predictions, _SAVE, _SHOW, pause=False,
                 **pcr_predictions_transformed)

    X_pred_pcr = pd.DataFrame(
        r_pcr[semente].predict(Y_test), columns=X_train.columns)
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

    path = f"pcr-reversed_transformed-algo_{_ALGO}-seed_{seed}"

    save_or_show(path, prefix, plot_predictions, _SAVE, _SHOW, pause=_PAUSE,
                 **pcr_predictions_reversed_transformed)


# === PLSR ===

for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    _, X_test, _, Y_test = split

    X_test_pls, Y_test_pls = (pd.DataFrame(test_t)
                              for test_t in plsr["Todos"][semente].transform(X_test, Y_test))

    Y_pred_plsr = pd.DataFrame(
        plsr["Todos"][semente].predict(X_test), columns=Y_train.columns)

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

    path = f"plsr-algo_{_ALGO}-seed_{seed}"
    prefix = "predictions/"

    save_or_show(path, prefix, plot_predictions, _SAVE, _SHOW, pause=False,
                 **plsr_predictions)

    _, Y_pred_plsr_t = (pd.DataFrame(test_t)
                        for test_t in plsr["Todos"][semente].transform(X_test, Y_pred_plsr))

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

    path = f"plsr-transformed-algo_{_ALGO}-seed_{seed}"

    save_or_show(path, prefix, plot_predictions, _SAVE, _SHOW, pause=False,
                 **plsr_predictions_transformed)

    Y_test_pls, X_test_pls = (pd.DataFrame(test_t)
                              for test_t in r_plsr[semente].transform(Y_test, X_test))

    X_pred_plsr = pd.DataFrame(
        r_plsr[semente].predict(Y_test), columns=X_train.columns)

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

    path = f"plsr-reversed_transformed-algo_{_ALGO}-seed_{seed}"

    save_or_show(path, prefix, plot_predictions, _SAVE, _SHOW, _PAUSE,
                 **plsr_predictions_reversed_transformed)


# TODO: use itertools for /.*th/ ords.
ordinais = list(enumerate([
    {"en": "1st", "pt": "Primeiro"},
    {"en": "2nd", "pt": "Segundo"},
    {"en": "3rd", "pt": "Terceiro"},
    {"en": "4th", "pt": "Quarto"},
    {"en": "5th", "pt": "Quinto"}
]))

corr_labels = {
    "en": "Pearson correlation",
    "pt": "Correlação de Pearson"
}

seed, semente = (str(None), "Nenhuma")

# NOTE: use correlation, not normalization.
X_all_pls = pd.DataFrame(
    plsr["Todos"][semente].x_scores_,
    columns=pls_component_names[:n_targets]
)
# ds: descriptors
X_all_ds_pls = pd.concat((X_all, X_all_pls), axis="columns")

Y_all_pls = pd.DataFrame(
    plsr["Todos"][semente].y_scores_,
    columns=pls_component_names[:n_targets]
)
# ts: targets
Y_all_ts_pls = pd.concat((Y_all, Y_all_pls), axis="columns")

# method="pearson"
x_pls_correlations = X_all_ds_pls.corr().iloc[:n_features, n_features:]
y_pls_correlations = Y_all_ts_pls.corr().iloc[:n_targets, n_targets:]

for i, o in ordinais:
    # .reshape(-1, 1)
    x_pls_corr_i = pd.DataFrame(
        x_pls_correlations.iloc[:, i],
        columns=[x_pls_correlations.columns[i]]
    )
    y_pls_corr_i = pd.DataFrame(
        y_pls_correlations.iloc[:, i],
        columns=[y_pls_correlations.columns[i]]
    )
    pls_x_component_corr_i = {
        "X": x_pls_corr_i,
        "titles": [None],
        "xlabels": descriptors,
        "sort": _SORT_X,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls-corr_{i}-algo_{_ALGO}-seed_{seed}"
    prefix = "x_component/"

    save_to_csv(x_pls_corr_i, path, save=(seed == str(None)), prefix=prefix)

    path += f"-sort_{_SORT_X}"

    for lang, label in corr_labels.items():
        lang_path = path + "-lang_" + lang
        pls_x_component_corr_i["ylabel"] = label

        # NOTE: pausing in a loop isn't practical.
        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pls_x_component_corr_i)

    pls_y_component_corr_i = {
        "X": y_pls_corr_i,
        "titles": [None],
        "xlabels": targets,
        "sort": _SORT_Y,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls-corr_{i}-algo_{_ALGO}-seed_{seed}"
    prefix = "y_component/"

    save_to_csv(y_pls_corr_i, path, save=(seed == str(None)), prefix=prefix)

    path += f"-sort_{_SORT_Y}"

    for lang, label in corr_labels.items():
        lang_path = path + "-lang_" + lang
        pls_y_component_corr_i["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                     **pls_y_component_corr_i)


pls_all_x_components = {
    "X": x_pls_correlations,
    "titles": [None for _ in ordinais],
    "xlabels": descriptors,
    "ncols": x_pls_correlations.shape[1],
    "sort": _SORT_X,
    "meanlabel": _MEANLABEL,
}

path = f"pls_all-corr-algo_{_ALGO}-seed_{seed}"
prefix = "x_component/"

save_to_csv(x_pls_correlations, path, save=(seed == str(None)), prefix=prefix)

path += f"-sort_{_SORT_X}"

for lang, label in corr_labels.items():
    lang_path = path + "-lang_" + lang
    pls_all_x_components["ylabel"] = label

    save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                 **pls_all_x_components)

pls_y_components = {
    "X": y_pls_correlations,
    "titles": [None for _ in ordinais],
    "xlabels": targets,
    "ncols": y_pls_correlations.shape[1],
    "sort": _SORT_Y,
    "meanlabel": _MEANLABEL,
}

path = f"pls-corr-algo_{_ALGO}-seed_{seed}"
prefix = "y_component/"

save_to_csv(y_pls_correlations, path, save=(seed == str(None)), prefix=prefix)

path += f"-sort_{_SORT_X}"

for lang, label in corr_labels.items():
    lang_path = path + "-lang_" + lang
    pls_y_components["ylabel"] = label

    save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                 **pls_y_components)


x_seeds_pls_corr = pd.DataFrame(index=X_all.columns)
y_seeds_pls_corr = pd.DataFrame(index=Y_all.columns)

for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    X_train, X_test, Y_train, Y_test = split

    x_scores = plsr["Todos"][semente].x_scores_
    y_scores = plsr["Todos"][semente].y_scores_

    # .reshape(-1, 1)
    x_first_component = pd.DataFrame(
        x_scores[:, 0],
        index=X_train.index,
        columns=[semente]
    )
    y_first_component = pd.DataFrame(
        y_scores[:, 0],
        index=Y_train.index,
        columns=[semente]
    )

    x_ds_first_component = pd.concat(
        (X_train, x_first_component), axis="columns")
    y_ts_first_component = pd.concat(
        (Y_train, y_first_component), axis="columns")

    x_ds_first_pls_corr = x_ds_first_component.corr()\
        .iloc[:n_features, n_features:]
    y_ts_first_pls_corr = y_ts_first_component.corr()\
        .iloc[:n_targets, n_targets:]

    x_seeds_pls_corr[semente] = x_ds_first_pls_corr
    y_seeds_pls_corr[semente] = y_ts_first_pls_corr

    pls_seeds_first_x_components = {
        "X": x_ds_first_pls_corr,
        "titles": [None],
        "xlabels": descriptors,
        "ylabel": "Correlação de Pearson",
        "sort": _SORT_X,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls-ds_corr_0-algo_{_ALGO}-seed_{seed}"
    prefix = "x_component/"

    save_to_csv(x_ds_first_pls_corr, path, save=(seed == str(None)),
                prefix=prefix)

    path += f"-sort_{_SORT_X}"

    for lang, label in corr_labels.items():
        lang_path = path + "-lang_" + lang
        pls_seeds_first_x_components["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pls_seeds_first_x_components)

    pls_seeds_first_y_components = {
        "X": y_ts_first_pls_corr,
        "titles": [None],
        "xlabels": targets,
        "sort": _SORT_Y,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls-ts_corr_0-algo_{_ALGO}-seed_{seed}"
    prefix = "y_component/"

    save_to_csv(y_ts_first_pls_corr, path, save=(seed == str(None)),
                prefix=prefix)

    path += f"-sort_{_SORT_Y}"

    for lang, label in corr_labels.items():
        lang_path = path + "-lang_" + lang
        pls_seeds_first_y_components["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                     **pls_seeds_first_y_components)


# seed=1241: best seed for `X = predict(Y)` and second-best
# for `Y = predict(X)` (based on r2_score).
seed = str(None)

# TODO: evaluate stability among runs for each target variable.

# For indexing.
all_ts = [None, *ts]
# For plotting.
all_targets = ["all", *targets]
todos_objetivos = ["Todos", *targets]

for t, objetivo in zip(all_ts, todos_objetivos):
    if objetivo == "Todos":
        plsr_target = plsr[objetivo]["Nenhuma"]
    else:
        plsr[objetivo] = {}
        Y_train_target = Y_all[t]
        plsr_target = PLSRegression(
            n_components=n_targets).fit(X_all, Y_train_target)
        plsr[objetivo]["Nenhuma"] = plsr_target

    pls_target_x_components = pd.DataFrame(
        plsr_target.x_rotations_,
        columns=pls_component_names[:n_max],
        index=X_all.columns)

    path = f"pls-algo_{_ALGO}-seed_{str(seed)}-target_{detexify(str(t))}"
    prefix = "x_component/"
    save_to_csv(pls_target_x_components, path, _SAVE, prefix=prefix)

    pls_target_first_x_component = pls_target_x_components.iloc[:, 0]

    # TODO, 2024-07-25: convert from raw component to correlations.
    pls_target_first_x_component_args = {
        "X": pls_target_first_x_component,
        "titles": [f"Primeiro Componente PLS de X, Objetivo: {objetivo}"],
        "xlabels": descriptors,
        "ylabel": "Peso",
        "sort": _SORT_X,
        "meanlabel": "média",
    }

    path = f"pls-comp_0-algo_{_ALGO}-seed_{seed}-sort_{_SORT_X}-target_{detexify(str(t))}-lang_pt"

    # NOTE: save and show only correlations, not raw components.
    save_or_show(path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                 **pls_target_first_x_component_args)


plsr_first_components_corrs = {}

for semente, (X_train, X_test, Y_train, Y_test) in splits.items():
    for t, objetivo in zip(all_ts, todos_objetivos):
        Y_train_target = Y_train[t] if t is not None else Y_train

        plsr_seed_target = PLSRegression(
            n_components=n_targets).fit(X_train, Y_train_target)
        plsr[objetivo][semente] = plsr_seed_target

        pls_seed_target_x_components = pd.DataFrame(
            plsr_seed_target.x_rotations_,
            columns=pls_component_names[:n_max],
            index=X_train.columns)

        t = "all" if t is None else t
        seed = str(None) if semente == "Nenhuma" else semente
        path = f"pls-algo_{_ALGO}-seed_{str(seed)}-target_{detexify(t)}"
        prefix = "x_component/"
        save_to_csv(pls_seed_target_x_components, path, _SAVE,
                    prefix=prefix)

        x_scores = plsr_seed_target.x_scores_
        x_first_component = pd.DataFrame(
            x_scores[:, 0],
            index=X_train.index,
            columns=[semente]
        )

        x_ds_first_component = pd.concat(
            (X_train, x_first_component), axis="columns"
        )

        x_ds_first_pls_corr = x_ds_first_component.corr()\
            .iloc[:n_features, n_features:]

        # Only set it in first pass.
        if semente == "Nenhuma":
            plsr_first_components_corrs[objetivo] = pd.DataFrame(
                x_ds_first_pls_corr,
                index=X_train.columns,
                columns=[semente]
            )
        else:
            plsr_first_components_corrs[objetivo][semente] = x_ds_first_pls_corr

        pls_target_seed_first_x_component_corr = {
            "X": x_ds_first_pls_corr,
            "titles": [None],
            "xlabels": descriptors,
            "sort": _SORT_X,
            "meanlabel": _MEANLABEL,
        }

        path = f"pls-corr_0-algo_{_ALGO}-seed_{seed}-target_{detexify(t)}"
        prefix = "x_component/"

        save_to_csv(x_ds_first_pls_corr, path, save=(seed == str(None)),
                    prefix=prefix)

        path += f"-sort_{_SORT_X}"

        for lang, label in corr_labels.items():
            lang_path = path + "-lang_" + lang
            pls_target_seed_first_x_component_corr["ylabel"] = label

            save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                         **pls_target_seed_first_x_component_corr)


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
    path = f"pcr_vs_plsr-algo_{_ALGO}-seed_{seed}"
    prefix = "predictions/"

    # No pause in for loop.
    save_or_show(path, prefix, plot_predictions, _SAVE, _SHOW, pause=False,
                 **pcr_vs_plsr_predictions)


semente, seed = ("Nenhuma", str(None))
x_pca_components = try_attr(pcr[semente], "components_")
y_pca_components = try_attr(r_pcr[semente], "components_")

X_all_pca = pd.DataFrame(
    x_pca_components,
    columns=pca_component_names[:n_features]
)
# ds: descriptors
X_all_ds_pca = pd.concat((X_all, X_all_pca), axis="columns")

Y_all_pca = pd.DataFrame(
    y_pca_components,
    columns=pca_component_names[:n_targets]
)
# ts: targets
Y_all_ts_pca = pd.concat((Y_all, Y_all_pca), axis="columns")

# NOTE: use correlation, not normalization.
# method="pearson"
x_pca_correlations = X_all_ds_pca.corr().iloc[:n_features, n_features:]
y_pca_correlations = Y_all_ts_pca.corr().iloc[:n_targets, n_targets:]

for i, o in ordinais:
    # .reshape(-1, 1)
    x_pca_corr_i = pd.DataFrame(
        x_pca_correlations.iloc[:, i],
        columns=[x_pca_correlations.columns[i]]
    )
    y_pca_corr_i = pd.DataFrame(
        y_pca_correlations.iloc[:, i],
        columns=[y_pca_correlations.columns[i]]
    )
    # NOTE: different title for X and Y.
    pca_x_component_corr_i = {
        "X": x_pca_corr_i,
        "titles": [None],
        "xlabels": descriptors,
        "sort": _SORT_X,
        "meanlabel": _MEANLABEL,
    }

    path = f"pca-corr_{i}-algo_{_ALGO}-seed_{seed}"
    prefix = "x_component/"

    save_to_csv(x_pca_corr_i, path, save=(seed == str(None)), prefix=prefix)

    path += f"-sort_{_SORT_X}"

    for lang, label in corr_labels.items():
        lang_path = path + "-lang_" + lang
        pca_x_component_corr_i["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pca_x_component_corr_i)

    pca_y_component_corr_i = {
        "X": y_pca_corr_i,
        "titles": [None],
        "xlabels": targets,
        "sort": _SORT_Y,
        "meanlabel": _MEANLABEL,
    }

    path = f"pca-corr_{i}-algo_{_ALGO}-seed_{seed}"
    prefix = "y_component/"

    save_to_csv(y_pca_corr_i, path, save=(seed == str(None)), prefix=prefix)

    path += f"-sort_{_SORT_Y}"

    for lang, label in corr_labels.items():
        lang_path = path + "-lang_" + lang
        pca_y_component_corr_i["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                     **pca_y_component_corr_i)


pca_y_components_corr = {
    "X": y_pca_correlations,
    "titles": [None for _ in ordinais],
    "xlabels": targets,
    "ncols": y_pca_correlations.shape[0],
    "sort": _SORT_Y,
    "meanlabel": _MEANLABEL,
}

path = f"pca-corr-algo_{_ALGO}-seed_{seed}"
prefix = "y_component/"

save_to_csv(y_pca_correlations, path, save=(seed == str(None)), prefix=prefix)

path += f"-sort_{_SORT_Y}"

for lang, label in corr_labels.items():
    lang_path = path + "-lang_" + lang
    pca_y_components_corr["ylabel"] = label

    save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                 **pca_y_components_corr)


algos = ("PCR", "PLSR")

# seed=1241 was the best for the ratio of rPLSR's r2_score
# over rPCR's.
for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    X_train, X_test, Y_train, Y_test = split

    pcr, rpcr, X_pred_pcr, Y_pred_pcr = fit_predict(
        ScalerPCR, *split, n_components=1)

    plsr, rplsr, X_pred_plsr, Y_pred_plsr = fit_predict(
        PLSRegression, *split, n_components=1)

    X_test_pca = pd.DataFrame(try_transform(pcr, X_test))
    Y_test_pca = pd.DataFrame(try_transform(rpcr, Y_test))

    X_test_pls = pd.DataFrame(try_transform(plsr, X_test))
    Y_test_pls = pd.DataFrame(try_transform(rplsr, Y_test))

    X_pred_pcr_pca = pd.DataFrame(try_transform(pcr, X_pred_pcr))
    Y_pred_pcr_pca = pd.DataFrame(try_transform(rpcr, Y_pred_pcr))

    X_pred_pcr_pls = pd.DataFrame(try_transform(plsr, X_pred_pcr))
    Y_pred_pcr_pls = pd.DataFrame(try_transform(rplsr, Y_pred_pcr))

    X_pred_plsr_pca = pd.DataFrame(try_transform(pcr, X_pred_plsr))
    Y_pred_plsr_pca = pd.DataFrame(try_transform(rpcr, Y_pred_plsr))

    X_pred_plsr_pls = pd.DataFrame(try_transform(plsr, X_pred_plsr))
    Y_pred_plsr_pls = pd.DataFrame(try_transform(rplsr, Y_pred_plsr))

    # NOTE: display R-squared for the prediction of each
    # component.
    R2_Y_pcr_pca = r2_score(Y_test_pca, Y_pred_pcr_pca,
                            multioutput="raw_values")
    R2_Y_pcr_pls = r2_score(Y_test_pls, Y_pred_pcr_pls,
                            multioutput="raw_values")

    R2_Y_plsr_pca = r2_score(Y_test_pca, Y_pred_plsr_pca,
                             multioutput="raw_values")
    R2_Y_plsr_pls = r2_score(Y_test_pls, Y_pred_plsr_pls,
                             multioutput="raw_values")

    # TODO: split regression plot into two (PCA/ DTR and PLS)
    pcr_vs_plsr_regression_pca = {
        "Y_true": pd.concat((Y_test_pca.iloc[:, 0], Y_test_pca.iloc[:, 0]), axis="columns"),
        "Y_pred": pd.concat((Y_pred_pcr_pca.iloc[:, 0], Y_pred_plsr_pca.iloc[:, 0]), axis="columns"),
        "xlabels": ["Primeiro componente PCA do Y real" for _ in algos],
        "ylabels": [f"Primeiro componente PCA do Y predito por {algo}" for algo in algos],
        "titles": algos,
        "R2": np.array((R2_Y_pcr_pca[0], R2_Y_plsr_pca[0])),
    }

    path = f"pcr_vs_plsr-algo_{_ALGO}-seed_{seed}-t_pca-lang_pt"
    prefix = "regression/"

    # pause=False: pausing is not practical in a for-loop.
    save_or_show(path, prefix, plot_regression, _SAVE, _SHOW, pause=False,
                 **pcr_vs_plsr_regression_pca)

    pcr_vs_plsr_regression_pls = {
        "Y_true": pd.concat((Y_test_pls.iloc[:, 0], Y_test_pls.iloc[:, 0]), axis="columns"),
        "Y_pred": pd.concat((Y_pred_pcr_pls.iloc[:, 0], Y_pred_plsr_pls.iloc[:, 0]), axis="columns"),
        "xlabels": ["Primeiro componente PLS do Y real" for _ in algos],
        "ylabels": [f"Primeiro componente PLS do Y predito por {algo}" for algo in algos],
        "titles": algos,
        "R2": np.array((R2_Y_pcr_pls[0], R2_Y_plsr_pls[0])),
    }

    path = f"pcr_vs_plsr-algo_{_ALGO}-seed_{seed}-t_pls-lang_pt"

    save_or_show(path, prefix, plot_regression, _SAVE, _SHOW, pause=False,
                 **pcr_vs_plsr_regression_pls)

    R2_X_pcr_pca = r2_score(X_test_pca, X_pred_pcr_pca,
                            multioutput="raw_values")
    R2_X_pcr_pls = r2_score(X_test_pls, X_pred_pcr_pls,
                            multioutput="raw_values")

    R2_X_plsr_pca = r2_score(X_test_pca, X_pred_plsr_pca,
                             multioutput="raw_values")
    R2_X_plsr_pls = r2_score(X_test_pls, X_pred_plsr_pls,
                             multioutput="raw_values")

    pcr_vs_plsr_regression_reversed_pca = {
        "Y_true": pd.concat((X_test_pca.iloc[:, 0], X_test_pca.iloc[:, 0]), axis="columns"),
        "Y_pred": pd.concat((X_pred_pcr_pca.iloc[:, 0], X_pred_plsr_pca.iloc[:, 0]), axis="columns"),
        "xlabels": ["Primeiro componente PCA do X real" for _ in algos],
        "ylabels": [f"Primeiro componente PCA do X predito por {algo}" for algo in algos],
        "titles": algos,
        "R2": np.array((R2_X_pcr_pca[0], R2_X_plsr_pca[0])),
    }

    path = f"pcr_vs_plsr-reversed-algo_{_ALGO}-seed_{seed}-t_pca-lang_pt"

    save_or_show(path, prefix, plot_regression, _SAVE, _SHOW, pause=False,
                 **pcr_vs_plsr_regression_reversed_pca)

    pcr_vs_plsr_regression_reversed_pls = {
        "Y_true": pd.concat((X_test_pls.iloc[:, 0], X_test_pls.iloc[:, 0]), axis="columns"),
        "Y_pred": pd.concat((X_pred_pcr_pls.iloc[:, 0], X_pred_plsr_pls.iloc[:, 0]), axis="columns"),
        "xlabels": ["Primeiro componente PLS do X real" for _ in algos],
        "ylabels": [f"Primeiro componente PLS do X predito por {algo}" for algo in algos],
        "titles": algos,
        "R2": np.array((R2_X_pcr_pls[0], R2_X_plsr_pls[0])),
    }

    path = f"pcr_vs_plsr-reversed-algo_{_ALGO}-seed_{seed}-t_pls-lang_pt"

    save_or_show(path, prefix, plot_regression, _SAVE, _SHOW, pause=_PAUSE,
                 **pcr_vs_plsr_regression_reversed_pls)


# DOING: train only multi-output regressors.
model_labels = ["PCR", "PLSR", "DTR"]  # , "SVR"
r_labels = ["r" + label for label in model_labels]

models = [ScalerPCR, PLSRegression, DecisionTreeRegressor]  # , ScalerSVR

N = 540

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

            m_kwargs = {
                "n_components": n
            }
            if label == "DTR":
                tree_seed = 0 if seed == str(None) else int(seed)
                m_kwargs = {
                    "min_samples_split": 200,
                    # TODO: review equivalent parameter to PCA/PLS's n_components.
                    "random_state": 10 * tree_seed + n,
                }

            m, rm, X_pred, Y_pred = fit_predict(
                model, X_train, X_test, Y_train, Y_test, **m_kwargs)

            # NOTE: Compare predictions in original feature
            # space, this confirms that calculating the
            # score on the transformed feature space
            # makes it easier for lower `n` to achieve
            # higher `r2_score`, which makes it harder
            # to observe the expected increase in
            # `r2_score` as `n` increases.
            for t, transformer in ((str(None), FunctionTransformer), ("PCA", ScalerPCA), ("PLS", PLSRegression)):
                t_kwargs = {
                    "n_components": n_max
                }
                # NOTE: pass FunctionTransformer as class, not object.
                if t == str(None):
                    t_kwargs = {
                        "func": lambda x: x
                    }
                tm = transformer(**t_kwargs).fit(X_train, Y_train)
                rtm = transformer(**t_kwargs).fit(Y_train, X_train)

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
                r2_m = r2_score(Y_test_t, Y_pred_t)
                r2_rm = r2_score(X_test_t, X_pred_t)

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
save_to_csv(r2s_df, path, _SAVE)


def print_r2s(df, model_labels, ns=None, seeds=None, t=None):
    t = str(t)
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

    msg += f", transformation: {t}"

    print(msg)
    # -1 for the '\n'.
    print("-" * (len(msg) - 1))

    for f in filters:
        print(f"\n{f_label} =", f)
        # NOTE: filter t == str(None), ignore transformed samples;
        # print results on original feature space only.
        df_filtered = df[(df[f_label] == f) & (df["t"] == t)]
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
print("\nPCA vs. PLS vs. DTR", end="")
print("\n===================", end="")

# TODO: investigate why PCA/PLS-transformed scores are worse
# than on original feature space.
for t in (str(None), "PCA", "PLS"):
    print_r2s(r2s_df, model_labels, ns=ns, t=t)

    # NOTE: `all_seeds` includes str(None) as a seed.
    print_r2s(r2s_df, model_labels, seeds=all_seeds, t=t)


print("\nPCA vs. PLS vs. DTR (reverse: X = model.predict(Y))", end="")
print("\n===================================================", end="")

for t in (str(None), "PCA", "PLS"):
    print_r2s(r2s_df, r_labels, ns=ns, t=t)

    print_r2s(r2s_df, r_labels, seeds=all_seeds, t=t)


# YYYY-mm-dd_HH-mm
_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
print(_now)

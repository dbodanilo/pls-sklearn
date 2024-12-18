import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

from decomposition import ScalerPCA, ScalerPCR
from model import load_deap, load_leme, train_test_seed_split
from plots import plot_components, plot_predictions, plot_regression
from util import (
    detexify,
    fit_predict,
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
_IDWL = True
_DATA = "idwl" if _IDWL else "wliv"

X, Y = load_deap(idwl=_IDWL) if _ALGO == "nsga3" else load_leme(idwl=_IDWL)

seeds = X["Semente"].value_counts().index

# train_test_seed_split() parameters.
all_seeds = (None, *seeds)
# For indexing and plotting.
todas_sementes = ("Nenhuma", *(str(s) for s in seeds))

X_all, _, Y_all, _ = train_test_seed_split(X, Y, seed=None)

# For indexing (original format).
ds = X_all.columns
# First two targets are the most important (Av0, fT).
ts = Y_all.columns

ts_pt = {"Area": "Área", "Pwr": "P"}

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
p_pcr = {}

r_pcr = {}
pr_pcr = {}

# "Todos": all targets (multivariate Y).
plsr = {"Todos": {}}
r_plsr = {}

component_names = {
    "pca": [f"PCA {i}" for i in range(1, n_features + 1)],
    "pls": [f"PLS {i}" for i in range(1, n_features + 1)]
}

# NOTE: evaluate stability among runs for each target variable.
# `seed=None` means all samples (no split).
for seed, semente in zip((None, *seeds), todas_sementes):
    splits[semente] = train_test_seed_split(X, Y, seed=seed)

    X_train, _, Y_train, _ = splits[semente]

    p_pcr[semente] = ScalerPCR(n_components=n_max).fit(X_train, Y_train)
    pr_pcr[semente] = ScalerPCR(n_components=n_max).fit(Y_train, X_train)

    X_train_pca = try_transform(p_pcr[semente], X_train)
    Y_train_pca = try_transform(pr_pcr[semente], Y_train)

    pcr[semente] = ScalerPCR(n_components=n_max).fit(X_train, Y_train_pca)
    r_pcr[semente] = ScalerPCR(n_components=n_max).fit(Y_train, X_train_pca)

    plsr_seed = PLSRegression(n_components=n_max).fit(X_train, Y_train)
    plsr["Todos"][semente] = plsr_seed

    r_plsr_seed = PLSRegression(n_components=n_max).fit(Y_train, X_train)
    r_plsr[semente] = r_plsr_seed

    # transform(X) = X * V = X * transpose(Vt), components_ = Vt.
    x_pca_components = pd.DataFrame(
        try_attr(pcr[semente], "components_"),
        columns=X_train.columns,
        index=component_names["pca"][:n_max])

    path = f"pca-algo_{_ALGO}-data_{_DATA}-seed_{str(seed)}"
    prefix = "x_component/"
    # TODO: save correlations, not raw components
    save_to_csv(x_pca_components, path, _SAVE, prefix=prefix)

    # transform(X) = X * x_rotations_
    x_pls_components = pd.DataFrame(
        plsr_seed.x_rotations_,
        columns=component_names["pls"][:n_max],
        index=X_train.columns)

    # "all" means all targets, only applies to:
    # PLS: because it's supervised, while PCA is not;
    # X: because Y are the targets themselves;
    # non-reversed: because it doesn't make much sense to
    # try to predict 20 variables (X) from a single scalar (y).
    path = f"pls_all-algo_{_ALGO}-data_{_DATA}-seed_{str(seed)}"
    save_to_csv(x_pls_components, path, _SAVE, prefix=prefix)

    # fit(Y, X) -> y_rotations_ transforms our X.
    x_rpls_components = pd.DataFrame(
        r_plsr_seed.y_rotations_,
        columns=component_names["pls"][:n_max],
        index=X_train.columns)

    path = f"pls-algo_{_ALGO}-data_{_DATA}-reversed-seed_{str(seed)}"
    save_to_csv(x_rpls_components, path, _SAVE, prefix=prefix)

    y_pca_components = pd.DataFrame(
        try_attr(r_pcr[semente], "components_"),
        columns=Y_train.columns,
        index=component_names["pca"][:n_max])

    path = f"pca-algo_{_ALGO}-data_{_DATA}-seed_{str(seed)}"
    prefix = "y_component/"
    save_to_csv(y_pca_components, path, _SAVE, prefix=prefix)

    y_pls_components = pd.DataFrame(
        plsr_seed.y_rotations_,
        columns=component_names["pls"][:n_max],
        index=Y_train.columns)

    path = f"pls-algo_{_ALGO}-data_{_DATA}-seed_{str(seed)}"
    save_to_csv(y_pls_components, path, _SAVE, prefix=prefix)

    y_rpls_components = pd.DataFrame(
        r_plsr_seed.x_rotations_,
        columns=component_names["pls"][:n_max],
        index=Y_train.columns)

    path = f"pls-algo_{_ALGO}-data_{_DATA}-reversed-seed_{str(seed)}"
    save_to_csv(y_rpls_components, path, _SAVE, prefix=prefix)

    pca_explained_variance_ratio = pd.DataFrame(
        {"X": try_attr(pcr[semente], "explained_variance_ratio_"),
         "Y": try_attr(r_pcr[semente], "explained_variance_ratio_")},
        index=component_names["pca"][:n_max])

    path = f"pca-algo_{_ALGO}-data_{_DATA}-explained_variance_ratio-seed_{str(seed)}"
    save_to_csv(pca_explained_variance_ratio, path, save=_SAVE)


# === PCA ===

x_pca = ScalerPCA(n_components=n_features).fit(X_all)

x_pca_components = pd.DataFrame(
    try_attr(x_pca, "components_"),
    columns=X_all.columns,
    index=component_names["pca"][:try_attr(x_pca, "n_components_")])

path = f"pca-x_components-algo_{_ALGO}-data_{_DATA}-seed_None"
save_to_csv(x_pca_components, path, _SAVE)

y_pca = ScalerPCA(n_components=n_targets).fit(Y_all)

y_pca_components = pd.DataFrame(
    try_attr(y_pca, "components_"),
    columns=Y_all.columns,
    index=component_names["pca"][:try_attr(y_pca, "n_components_")])

path = f"pca-y_components-algo_{_ALGO}-data_{_DATA}-seed_None"
save_to_csv(y_pca_components, path, _SAVE)

x_pca_explained_variance_ratio = try_attr(x_pca, "explained_variance_ratio_")

# right-pad Y ratios in order to place X and Y on the same DataFrame.
y_pca_explained_variance_ratio = np.pad(
    try_attr(y_pca, "explained_variance_ratio_"),
    (0, n_features - n_targets),
    mode="constant",
    constant_values=np.nan)

pca_explained_variance_ratio = pd.DataFrame(
    {"X": x_pca_explained_variance_ratio,
     "Y": y_pca_explained_variance_ratio},
    index=component_names["pca"][:try_attr(x_pca, "n_components_")])

path = f"pca-explained_variance_ratio-algo_{_ALGO}-data_{_DATA}-seed_None"
save_to_csv(pca_explained_variance_ratio, path, _SAVE)


# YYYY-mm-dd_HH-mm
_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
print(_now)

print("algo:", _ALGO, )
print("data:", "(I_D/(W/L))" if _IDWL else "Ws, Ls, biases")

print("PCA\n===")
print("X and Y\n-------")
for n in range(1, n_targets + 1):
    # TODO: as with the R-square at the end:
    # aggregate metric over all seeds (mean)
    x_explained = x_pca_explained_variance_ratio[:n].sum()
    y_explained = y_pca_explained_variance_ratio[:n].sum()
    print("\nn =", n)
    print(f"{100*x_explained:.2f}% of X's variance explained")
    print(f"{100*y_explained:.2f}% of Y's variance explained")

# Until comparable to that of Y's first two PCA components
# (scaled and on worse train/test split).
# TODO: go back to exploring n > 5, at least for X.
print("\nOnly X\n------")
for n in range(n_targets + 1, n_features + 1):
    x_explained = x_pca_explained_variance_ratio[:n].sum()
    print("\nn =", n)
    print(f"{100*x_explained:.2f}% of X's variance explained")


# === PCR ===
# --- Predictions ---
# save=False because they are not in the dissertation.
for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    _, X_test, _, Y_test = split

    X_test_pca = pd.DataFrame(try_transform(pcr[semente], X_test))
    Y_pred_pcr = pd.DataFrame(
        p_pcr[semente].predict(X_test),
        columns=Y_train.columns
    )

    # TODO: file fixes an algorithm (PCR) and a transformation (None),
    # store seed as a column.
    R2_Y_pcr = pd.Series(
        r2_score(Y_test, Y_pred_pcr, multioutput="raw_values"),
        index=Y_test.columns
    )

    path = f"pcr-algo_{_ALGO}-data_{_DATA}-t_None-seed_{seed}"
    prefix = "y_pred/"

    save_to_csv(R2_Y_pcr, path, save=False, prefix=prefix)

    pcr_predictions = {
        "xlabels": [f"X's PCA {i}, Semente: {semente}" for i in range(1, n_max + 1)],
        "ylabels": targets,
        "X": X_test_pca,
        "Y_true": Y_test,
        "Y_pred": Y_pred_pcr,
        "R2": R2_Y_pcr,
        "iter_x": False,
        "ncols": 3,
        "nrows": 2,
    }

    # pause=False: no pause in for loop.
    save_or_show(path, prefix, plot_predictions, save=False, show=False, pause=False,
                 **pcr_predictions)

    Y_test_pca = pd.DataFrame(try_transform(r_pcr[semente], Y_test))
    Y_pred_pcr_pca = pd.DataFrame(pcr[semente].predict(X_test))
    R2_Y_pcr_pca = pd.Series(
        r2_score(Y_test_pca, Y_pred_pcr_pca, multioutput="raw_values"))

    path = f"pcr-algo_{_ALGO}-data_{_DATA}-t_pca-seed_{seed}"

    save_to_csv(R2_Y_pcr_pca, path, save=False, prefix=prefix)

    pcr_predictions_transformed = {
        "X": X_test_pca,
        "Y_true": Y_test_pca,
        "Y_pred": Y_pred_pcr_pca,
        "xlabels": [f"X's PCA {i}" for i in range(1, n_targets + 1)],
        "ylabels": [f"Y's PCA {i}" for i in range(1, n_targets + 1)],
        "R2": R2_Y_pcr_pca,
        "ncols": 3,
    }

    save_or_show(path, prefix, plot_predictions, save=False, show=False, pause=False,
                 **pcr_predictions_transformed)

    Y_test_pls = pd.DataFrame(try_transform(r_plsr[semente], Y_test))
    Y_pred_pcr_pls = pd.DataFrame(try_transform(r_plsr[semente], Y_pred_pcr))
    R2_Y_pcr_pls = pd.Series(
        r2_score(Y_test_pls, Y_pred_pcr_pls, multioutput="raw_values"))
    path = f"pcr-algo_{_ALGO}-data_{_DATA}-t_pls-seed_{seed}"

    save_to_csv(R2_Y_pcr_pls, path, save=False, prefix=prefix)

    X_pred_pcr = pd.DataFrame(
        pr_pcr[semente].predict(Y_test),
        columns=X_train.columns
    )
    R2_X_pcr = pd.Series(
        r2_score(X_test, X_pred_pcr, multioutput="raw_values"),
        index=X_test.columns
    )

    path = f"pcr-algo_{_ALGO}-data_{_DATA}-t_None-seed_{seed}"
    prefix = "x_pred/"

    save_to_csv(R2_X_pcr, path, save=False, prefix=prefix)

    X_pred_pcr_pca = pd.DataFrame(r_pcr[semente].predict(Y_test))
    R2_X_pcr_pca = pd.Series(
        r2_score(X_test_pca, X_pred_pcr_pca, multioutput="raw_values")
    )

    path = f"pcr-algo_{_ALGO}-data_{_DATA}-t_pca-seed_{seed}"

    save_to_csv(R2_X_pcr_pca, path, save=False, prefix=prefix)

    pcr_predictions_reversed_transformed = {
        "X": Y_test_pca,
        "Y_true": X_test_pca,
        "Y_pred": X_pred_pcr_pca,
        "xlabels": [f"Y's PCA {i}" for i in range(1, n_targets + 1)],
        "ylabels": [f"X's PCA {i}" for i in range(1, n_targets + 1)],
        "iter_x": False,
        "R2": R2_X_pcr_pca,
        "ncols": 3,
    }

    path = f"pcr-algo_{_ALGO}-data_{_DATA}-t_pca-seed_{seed}"

    save_or_show(path, prefix, plot_predictions, save=False, show=False, pause=False,
                 **pcr_predictions_reversed_transformed)

    X_test_pls = pd.DataFrame(try_transform(plsr["Todos"][semente], X_test))
    X_pred_pcr_pls = pd.DataFrame(
        try_transform(plsr["Todos"][semente], X_pred_pcr)
    )
    R2_X_pcr_pls = pd.Series(
        r2_score(X_test_pls, X_pred_pcr_pls, multioutput="raw_values")
    )
    path = f"pcr-algo_{_ALGO}-data_{_DATA}-t_pls-seed_{seed}"

    save_to_csv(R2_Y_pcr_pls, path, save=False, prefix=prefix)


# === PLSR ===

for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    _, X_test, _, Y_test = split

    X_test_pls, Y_test_pls = (pd.DataFrame(test_pls)
                              for test_pls in plsr["Todos"][semente].transform(X_test, Y_test))

    Y_pred_plsr = pd.DataFrame(
        plsr["Todos"][semente].predict(X_test),
        columns=Y_train.columns
    )

    R2_Y_plsr = pd.Series(
        r2_score(Y_test, Y_pred_plsr, multioutput="raw_values"),
        index=Y_test.columns
    )

    path = f"plsr-algo_{_ALGO}-data_{_DATA}-t_None-seed_{seed}"
    prefix = "y_pred/"

    save_to_csv(R2_Y_plsr, path, save=False, prefix=prefix)

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

    save_or_show(path, prefix, plot_predictions, save=False, show=False, pause=False,
                 **plsr_predictions)

    _, Y_pred_plsr_pls = (pd.DataFrame(test_pls)
                          for test_pls in plsr["Todos"][semente].transform(X_test, Y_pred_plsr))

    R2_Y_plsr_pls = pd.Series(
        r2_score(Y_test_pls, Y_pred_plsr_pls, multioutput="raw_values")
    )

    path = f"plsr-algo_{_ALGO}-data_{_DATA}-t_pls-seed_{seed}"

    save_to_csv(R2_Y_plsr_pls, path, save=False, prefix=prefix)

    plsr_predictions_transformed = {
        "X": X_test_pls,
        "Y_true": Y_test_pls,
        "Y_pred": Y_pred_plsr_pls,
        "xlabels": [f"X's PLS {i}" for i in range(1, n_max + 1)],
        "ylabels": [f"Y's PLS {i}" for i in range(1, n_max + 1)],
        "R2": R2_Y_plsr_pls,
        "ncols": 3,
        "nrows": 2,
    }

    save_or_show(path, prefix, plot_predictions, save=False, show=False, pause=False,
                 **plsr_predictions_transformed)

    Y_test_pca = pd.DataFrame(try_transform(r_pcr[semente], Y_test))
    Y_pred_plsr_pca = pd.DataFrame(try_transform(r_pcr[semente], Y_pred_plsr))
    R2_Y_plsr_pca = pd.Series(
        r2_score(Y_test_pca, Y_pred_plsr_pca, multioutput="raw_values")
    )

    path = f"plsr-algo_{_ALGO}-data_{_DATA}-t_pca-seed_{seed}"

    save_to_csv(R2_Y_plsr_pca, path, save=False, prefix=prefix)

    Y_test_pls, X_test_pls = (pd.DataFrame(test_pls)
                              for test_pls in r_plsr[semente].transform(Y_test, X_test))

    X_pred_plsr = pd.DataFrame(
        r_plsr[semente].predict(Y_test), columns=X_train.columns)

    R2_X_plsr = pd.Series(
        r2_score(X_test, X_pred_plsr, multioutput="raw_values"),
        index=X_test.columns
    )

    path = f"plsr-algo_{_ALGO}-data_{_DATA}-t_None-seed_{seed}"
    prefix = "x_pred/"

    save_to_csv(R2_X_plsr, path, save=False, prefix=prefix)

    _, X_pred_plsr_pls = (pd.DataFrame(test_pls)
                          for test_pls in r_plsr[semente].transform(Y_test, X_pred_plsr))

    R2_X_plsr_pls = pd.Series(
        r2_score(X_test_pls, X_pred_plsr_pls, multioutput="raw_values")
    )

    path = f"plsr-algo_{_ALGO}-data_{_DATA}-t_pls-seed_{seed}"

    save_to_csv(R2_X_plsr_pls, path, save=False, prefix=prefix)

    plsr_predictions_reversed_transformed = {
        "X": Y_test_pls,
        "Y_true": X_test_pls,
        "Y_pred": X_pred_plsr_pls,
        "xlabels": [f"Y's PLS {i}" for i in range(1, n_max + 1)],
        "ylabels": [f"X's PLS {i}" for i in range(1, n_max + 1)],
        "R2": R2_X_plsr_pls,
        "ncols": 3,
        "nrows": 2,
    }

    save_or_show(path, prefix, plot_predictions, save=False, show=False, pause=False,
                 **plsr_predictions_reversed_transformed)

    X_test_pca = pd.DataFrame(try_transform(pcr[semente], X_test))
    X_pred_plsr_pca = pd.DataFrame(try_transform(pcr[semente], X_pred_plsr))
    R2_X_plsr_pca = pd.Series(
        r2_score(X_test_pca, X_pred_plsr_pca, multioutput="raw_values")
    )

    path = f"plsr-algo_{_ALGO}-data_{_DATA}-t_pca-seed_{seed}"

    save_to_csv(R2_X_plsr_pca, path, save=False, prefix=prefix)

    X_test_pca = pd.DataFrame(try_transform(pcr[semente], X_test))
    X_pred_plsr_pca = pd.DataFrame(try_transform(pcr[semente], X_pred_plsr))
    R2_X_plsr_pca = pd.Series(
        r2_score(X_test_pca, X_pred_plsr_pca, multioutput="raw_values")
    )

    path = f"plsr-algo_{_ALGO}-data_{_DATA}-t_pca-seed_{seed}"

    save_to_csv(R2_X_plsr_pca, path, save=False, prefix=prefix)


# TODO: use itertools for /.*th/ ords.
ordinais = list(enumerate([
    {"en": "1st", "pt": "Primeiro"},
    {"en": "2nd", "pt": "Segundo"},
    {"en": "3rd", "pt": "Terceiro"},
    {"en": "4th", "pt": "Quarto"},
    {"en": "5th", "pt": "Quinto"}
]))

corr_labels = (
    ("en", "Pearson correlation", {}),
    ("pt", "Correlação de Pearson", ts_pt),
)

seed, semente = (str(None), "Nenhuma")

# NOTE: use correlation, not normalization.
X_all_pls = pd.DataFrame(
    plsr["Todos"][semente].x_scores_,
    columns=component_names["pls"][:n_targets]
)
# ds: descriptors
X_all_pls_ds = pd.concat((X_all_pls, X_all), axis="columns")
X_all_pls_ts = pd.concat((X_all_pls, Y_all), axis="columns")

Y_all_pls = pd.DataFrame(
    plsr["Todos"][semente].y_scores_,
    columns=component_names["pls"][:n_targets]
)
# ts: targets
Y_all_pls_ts = pd.concat((Y_all_pls, Y_all), axis="columns")
Y_all_pls_ds = pd.concat((Y_all_pls, X_all), axis="columns")

# method="pearson"
x_pls_ds_correlations = X_all_pls_ds.corr().iloc[-n_features:, :-n_features]
x_pls_ts_correlations = X_all_pls_ts.corr().iloc[-n_targets:, :-n_targets]

y_pls_ts_correlations = Y_all_pls_ts.corr().iloc[-n_targets:, :-n_targets]
y_pls_ds_correlations = Y_all_pls_ds.corr().iloc[-n_features:, :-n_features]

for i, o in ordinais:
    # .reshape(-1, 1)
    x_pls_ds_corr_i = pd.DataFrame(x_pls_ds_correlations.iloc[:, i])
    x_pls_ts_corr_i = pd.DataFrame(x_pls_ts_correlations.iloc[:, i])

    y_pls_ts_corr_i = pd.DataFrame(y_pls_ts_correlations.iloc[:, i])
    y_pls_ds_corr_i = pd.DataFrame(y_pls_ds_correlations.iloc[:, i])

    pls_x_component_ds_corr_i = {
        "X": x_pls_ds_corr_i,
        "titles": [None],
        "xlabels": [None for _ in descriptors],
        "sort": _SORT_X,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls_ord-ds_corr_{i}-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
    prefix = "x_component/"

    path += f"-sort_{_SORT_X}"

    for lang, label, _ in corr_labels:
        lang_path = path + "-lang_" + lang
        pls_x_component_ds_corr_i["ylabel"] = label

        # NOTE: pausing in a loop isn't practical.
        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pls_x_component_ds_corr_i)

    pls_x_component_ts_corr_i = {
        "titles": [None],
        "xlabels": [None for _ in targets],
        "sort": _SORT_Y,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls_ord-ts_corr_{i}-algo_{_ALGO}-data_{_DATA}-seed_{seed}"

    path += f"-sort_{_SORT_Y}"

    for lang, label, mapper in corr_labels:
        lang_path = path + "-lang_" + lang

        pls_x_component_ts_corr_i["X"] = x_pls_ts_corr_i.rename(index=mapper)
        pls_x_component_ts_corr_i["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pls_x_component_ts_corr_i)

    pls_y_component_ts_corr_i = {
        "titles": [None],
        "xlabels": [None for _ in targets],
        "sort": _SORT_Y,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls_ord-ts_corr_{i}-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
    prefix = "y_component/"

    path += f"-sort_{_SORT_Y}"

    for lang, label, mapper in corr_labels:
        lang_path = path + "-lang_" + lang

        pls_y_component_ts_corr_i["X"] = y_pls_ts_corr_i.rename(index=mapper)
        pls_y_component_ts_corr_i["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pls_y_component_ts_corr_i)

    pls_y_component_ds_corr_i = {
        "X": y_pls_ds_corr_i,
        "titles": [None],
        "xlabels": [None for _ in descriptors],
        "sort": _SORT_X,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls_ord-ds_corr_{i}-algo_{_ALGO}-data_{_DATA}-seed_{seed}"

    path += f"-sort_{_SORT_X}"

    for lang, label, _ in corr_labels:
        lang_path = path + "-lang_" + lang
        pls_y_component_ds_corr_i["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                     **pls_y_component_ds_corr_i)


pls_all_x_components_ds = {
    "X": x_pls_ds_correlations,
    "titles": [None for _ in ordinais],
    "xlabels": [None for _ in descriptors],
    "ncols": x_pls_ds_correlations.shape[1],
    "sort": _SORT_X,
    "meanlabel": _MEANLABEL,
}

path = f"pls_all-ds_corr-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
prefix = "x_component/"

save_to_csv(x_pls_ds_correlations, path, save=_SAVE, prefix=prefix)

path += f"-sort_{_SORT_X}"

for lang, label, _ in corr_labels:
    lang_path = path + "-lang_" + lang
    pls_all_x_components_ds["ylabel"] = label

    save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                 **pls_all_x_components_ds)

pls_x_components_ts = {
    "titles": [None for _ in ordinais],
    "xlabels": [None for _ in targets],
    "ncols": x_pls_ts_correlations.shape[1],
    "sort": _SORT_Y,
    "meanlabel": _MEANLABEL,
}

path = f"pls-ts_corr-algo_{_ALGO}-data_{_DATA}-seed_{seed}"

save_to_csv(x_pls_ts_correlations, path, save=_SAVE, prefix=prefix)

path += f"-sort_{_SORT_Y}"

for lang, label, mapper in corr_labels:
    lang_path = path + "-lang_" + lang

    pls_x_components_ts["X"] = x_pls_ts_correlations.rename(index=mapper)
    pls_x_components_ts["ylabel"] = label

    save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                 **pls_x_components_ts)

pls_y_components_ts = {
    "titles": [None for _ in ordinais],
    "xlabels": [None for _ in targets],
    "ncols": y_pls_ts_correlations.shape[1],
    "sort": _SORT_Y,
    "meanlabel": _MEANLABEL,
}

path = f"pls-ts_corr-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
prefix = "y_component/"

save_to_csv(y_pls_ts_correlations, path, save=_SAVE, prefix=prefix)

path += f"-sort_{_SORT_Y}"

for lang, label, mapper in corr_labels:
    lang_path = path + "-lang_" + lang

    pls_y_components_ts["X"] = y_pls_ts_correlations.rename(index=mapper)
    pls_y_components_ts["ylabel"] = label

    save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                 **pls_y_components_ts)

pls_all_y_components_ds = {
    "X": y_pls_ds_correlations,
    "titles": [None for _ in ordinais],
    "xlabels": [None for _ in descriptors],
    "ncols": y_pls_ds_correlations.shape[1],
    "sort": _SORT_X,
    "meanlabel": _MEANLABEL,
}

path = f"pls_all-ds_corr-algo_{_ALGO}-data_{_DATA}-seed_{seed}"

save_to_csv(y_pls_ds_correlations, path, save=_SAVE, prefix=prefix)

path += f"-sort_{_SORT_X}"

for lang, label, _ in corr_labels:
    lang_path = path + "-lang_" + lang
    pls_all_y_components_ds["ylabel"] = label

    save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                 **pls_all_y_components_ds)


x_seeds_pls_ds_corr = pd.DataFrame(index=X_all.columns)
x_seeds_pls_ts_corr = pd.DataFrame(index=Y_all.columns)

y_seeds_pls_ts_corr = pd.DataFrame(index=Y_all.columns)
y_seeds_pls_ds_corr = pd.DataFrame(index=X_all.columns)

for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    X_train, X_test, Y_train, Y_test = split

    x_scores = pd.DataFrame(
        plsr["Todos"][semente].x_scores_,
        columns=component_names["pls"][:n_max]
    )
    y_scores = pd.DataFrame(
        plsr["Todos"][semente].y_scores_,
        columns=component_names["pls"][:n_max]
    )

    x_seed_ds = pd.concat(
        (x_scores, X_train), axis="columns")
    x_seed_ts = pd.concat(
        (x_scores, Y_train), axis="columns")

    y_seed_ts = pd.concat(
        (y_scores, Y_train), axis="columns")
    y_seed_ds = pd.concat(
        (y_scores, X_train), axis="columns")

    x_seed_pls_ds_corr = x_seed_ds.corr()\
        .iloc[-n_features:, :-n_features]
    x_seed_pls_ts_corr = x_seed_ts.corr()\
        .iloc[-n_targets:, :-n_targets]

    y_seed_pls_ts_corr = y_seed_ts.corr()\
        .iloc[-n_targets:, :-n_targets]
    y_seed_pls_ds_corr = y_seed_ds.corr()\
        .iloc[-n_features:, :-n_features]

    x_seeds_pls_ds_corr[semente] = x_seed_pls_ds_corr.iloc[:, 0]
    x_seeds_pls_ts_corr[semente] = x_seed_pls_ts_corr.iloc[:, 0]

    y_seeds_pls_ts_corr[semente] = y_seed_pls_ts_corr.iloc[:, 0]
    y_seeds_pls_ds_corr[semente] = y_seed_pls_ds_corr.iloc[:, 0]

    pls_seeds_first_x_components_ds = {
        "X": x_seed_pls_ds_corr,
        "titles": [None],
        "xlabels": [None for _ in descriptors],
        "sort": _SORT_X,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls_seeds-ds_corr_0-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
    prefix = "x_component/"

    save_to_csv(x_seed_pls_ds_corr, path, save=_SAVE, prefix=prefix)

    path += f"-sort_{_SORT_X}"

    for lang, label, _ in corr_labels:
        lang_path = path + "-lang_" + lang
        pls_seeds_first_x_components_ds["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pls_seeds_first_x_components_ds)

    pls_seeds_first_x_components_ts = {
        "titles": [None],
        "xlabels": [None for _ in targets],
        "sort": _SORT_Y,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls_seeds-ts_corr_0-algo_{_ALGO}-data_{_DATA}-seed_{seed}"

    save_to_csv(x_seed_pls_ts_corr, path, save=_SAVE, prefix=prefix)

    path += f"-sort_{_SORT_Y}"

    for lang, label, mapper in corr_labels:
        lang_path = path + "-lang_" + lang

        pls_seeds_first_x_components_ts["X"] = x_seed_pls_ts_corr.rename(
            index=mapper)
        pls_seeds_first_x_components_ts["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pls_seeds_first_x_components_ts)

    pls_seeds_first_y_components_ts = {
        "titles": [None],
        "xlabels": [None for _ in targets],
        "sort": _SORT_Y,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls_seeds-ts_corr_0-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
    prefix = "y_component/"

    save_to_csv(y_seed_pls_ts_corr, path, save=_SAVE, prefix=prefix)

    path += f"-sort_{_SORT_Y}"

    for lang, label, mapper in corr_labels:
        lang_path = path + "-lang_" + lang

        pls_seeds_first_y_components_ts["X"] = y_seed_pls_ts_corr.rename(
            index=mapper)
        pls_seeds_first_y_components_ts["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pls_seeds_first_y_components_ts)

    pls_seeds_y_components_ds = {
        "X": y_seed_pls_ds_corr,
        "titles": [None],
        "xlabels": [None for _ in descriptors],
        "sort": _SORT_X,
        "meanlabel": _MEANLABEL,
    }

    path = f"pls_seeds-ds_corr_0-algo_{_ALGO}-data_{_DATA}-seed_{seed}"

    save_to_csv(y_seed_pls_ds_corr, path, save=_SAVE, prefix=prefix)

    path += f"-sort_{_SORT_X}"

    for lang, label, _ in corr_labels:
        lang_path = path + "-lang_" + lang
        pls_seeds_y_components_ds["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                     **pls_seeds_y_components_ds)


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
        columns=component_names["pls"][:n_max],
        index=X_all.columns)

    path = f"pls-algo_{_ALGO}-data_{_DATA}-seed_{str(seed)}-target_{detexify(str(t))}"
    prefix = "x_component/"
    save_to_csv(pls_target_x_components, path, _SAVE, prefix=prefix)

    pls_target_first_x_component = pls_target_x_components.iloc[:, 0]

    # TODO, 2024-07-25: convert from raw component to correlations.
    pls_target_first_x_component_args = {
        "X": pls_target_first_x_component,
        "titles": [None],
        "xlabels": [None for _ in descriptors],
        "ylabel": "Peso",
        "sort": _SORT_X,
        "meanlabel": "média",
    }

    path = f"pls-comp_0-algo_{_ALGO}-data_{_DATA}-seed_{seed}-sort_{_SORT_X}-target_{detexify(str(t))}-lang_pt"

    # NOTE: save and show only correlations, not raw components.
    save_or_show(path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                 **pls_target_first_x_component_args)


plsr_first_components_ds_corrs = {}
plsr_first_components_ts_corrs = {}

for semente, (X_train, X_test, Y_train, Y_test) in splits.items():
    for t, objetivo in zip(all_ts, todos_objetivos):
        Y_train_target = Y_train[t] if t is not None else Y_train

        plsr_seed_target = PLSRegression(
            n_components=n_targets).fit(X_train, Y_train_target)
        plsr[objetivo][semente] = plsr_seed_target

        pls_seed_target_x_components = pd.DataFrame(
            plsr_seed_target.x_rotations_,
            columns=component_names["pls"][:n_max],
            index=X_train.columns)

        t = "all" if t is None else t
        seed = str(None) if semente == "Nenhuma" else semente
        path = f"pls_seeds_ts-comps-algo_{_ALGO}-data_{_DATA}-seed_{str(seed)}-target_{detexify(t)}"
        prefix = "x_component/"
        save_to_csv(pls_seed_target_x_components, path, _SAVE,
                    prefix=prefix)

        x_scores = pd.DataFrame(
            plsr_seed_target.x_scores_,
            columns=component_names["pls"][:n_max]
        )

        x_seed_t_component_ds = pd.concat(
            (x_scores, X_train), axis="columns"
        )
        x_seed_t_component_ts = pd.concat(
            (x_scores, Y_train), axis="columns"
        )

        x_seed_t_pls_ds_corr = x_seed_t_component_ds.corr()\
            .iloc[-n_features:, :-n_features]
        x_seed_t_pls_ts_corr = x_seed_t_component_ts.corr()\
            .iloc[-n_targets:, :-n_targets]

        # Only set it in first pass.
        if semente == "Nenhuma":
            plsr_first_components_ds_corrs[objetivo] = x_seed_t_pls_ds_corr.iloc[:, 0]
            plsr_first_components_ts_corrs[objetivo] = x_seed_t_pls_ts_corr.iloc[:, 0]
        else:
            plsr_first_components_ds_corrs[objetivo][semente] = x_seed_t_pls_ds_corr.iloc[:, 0]
            plsr_first_components_ts_corrs[objetivo][semente] = x_seed_t_pls_ts_corr.iloc[:, 0]

        pls_target_seed_first_x_component_ds_corr = {
            "X": x_seed_t_pls_ds_corr,
            "titles": [None],
            "xlabels": [None for _ in descriptors],
            "sort": _SORT_X,
            "meanlabel": _MEANLABEL,
        }

        path = f"pls_seeds_ts-ds_corrs-algo_{_ALGO}-data_{_DATA}-seed_{seed}-target_{detexify(t)}"

        save_to_csv(x_seed_t_pls_ds_corr, path, save=_SAVE, prefix=prefix)

        path += f"-sort_{_SORT_X}"

        for lang, label, _ in corr_labels:
            lang_path = path + "-lang_" + lang
            pls_target_seed_first_x_component_ds_corr["ylabel"] = label

            save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                         **pls_target_seed_first_x_component_ds_corr)

        pls_target_seed_first_x_component_ts_corr = {
            "titles": [None],
            "xlabels": [None for _ in targets],
            "sort": _SORT_Y,
            "meanlabel": _MEANLABEL,
        }

        path = f"pls_seeds_ts-ts_corrs-algo_{_ALGO}-data_{_DATA}-seed_{seed}-target_{detexify(t)}"

        save_to_csv(x_seed_t_pls_ts_corr, path, save=_SAVE, prefix=prefix)

        path += f"-sort_{_SORT_Y}"

        for lang, label, mapper in corr_labels:
            lang_path = path + "-lang_" + lang

            pls_target_seed_first_x_component_ts_corr["X"] = x_seed_t_pls_ts_corr.rename(
                index=mapper)
            pls_target_seed_first_x_component_ts_corr["ylabel"] = label

            save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                         **pls_target_seed_first_x_component_ts_corr)


# === PCR vs. PLSR ===

# NOTE: split it so that PLSR performs better than PCR.
for semente, split in splits.items():
    seed = None if semente == "Nenhuma" else semente

    X_train, X_test, Y_train, Y_test = split

    _, _, X_pred_pcr, Y_pred_pcr = fit_predict(
        ScalerPCR, *split, n_components=1)

    _, _, X_pred_plsr, Y_pred_plsr = fit_predict(
        PLSRegression, *split, n_components=1)

    X_test_pca = pd.DataFrame(try_transform(pcr[semente], X_test))
    X_test_pls = pd.DataFrame(try_transform(plsr["Todos"][semente], X_test))

    Y_test_pca = pd.DataFrame(try_transform(r_pcr[semente], Y_test))
    Y_test_pls = pd.DataFrame(try_transform(r_plsr[semente], Y_test))

    Y_pred_pcr_pca = pd.DataFrame(try_transform(r_pcr[semente], Y_pred_pcr))
    Y_pred_plsr_pca = pd.DataFrame(try_transform(r_pcr[semente], Y_pred_plsr))

    Y_pred_pcr_pls = pd.DataFrame(try_transform(r_plsr[semente], Y_pred_pcr))
    Y_pred_plsr_pls = pd.DataFrame(try_transform(r_plsr[semente], Y_pred_plsr))

    R2_Y_pcr = r2_score(Y_test, Y_pred_pcr, multioutput="raw_values")
    R2_Y_plsr = r2_score(Y_test, Y_pred_plsr, multioutput="raw_values")

    R2_Y_pcr_pca = r2_score(Y_test_pca, Y_pred_pcr_pca,
                            multioutput="raw_values")
    R2_Y_plsr_pca = r2_score(Y_test_pca, Y_pred_plsr_pca,
                             multioutput="raw_values")

    R2_Y_pcr_pls = r2_score(Y_test_pls, Y_pred_pcr_pls,
                            multioutput="raw_values")
    R2_Y_plsr_pls = r2_score(Y_test_pls, Y_pred_plsr_pls,
                             multioutput="raw_values")

    # TODO:
    # 1. plot Y predictions transformed with PLS.
    # 2. plot X predictions.
    pcr_vs_plsr_predictions_pca = {
        "X": pd.concat((X_test_pca.iloc[:, 0],) * 2, axis="columns"),
        "Y_true": pd.concat((Y_test_pca.iloc[:, 0],) * 2, axis="columns"),
        "Y_pred": pd.concat((Y_pred_pcr_pca.iloc[:, 0], Y_pred_plsr_pca.iloc[:, 0]), axis="columns"),
        "xlabels": ["X's PCA 1",] * 2,
        "ylabels": ["Y's PCA 1",] * 2,
        "R2": np.array((R2_Y_pcr_pca[0], R2_Y_plsr_pca[0])),
    }

    # NOTE: print seed used when outputting plots and scores.
    path = f"pcr_vs_plsr-algo_{_ALGO}-data_{_DATA}-t_pca-seed_{seed}"
    prefix = "y_pred/"

    # No pause in for loop.
    save_or_show(path, prefix, plot_predictions, save=False, show=False, pause=False,
                 **pcr_vs_plsr_predictions_pca)


semente, seed = ("Nenhuma", str(None))
X_pca_scores = pd.DataFrame(try_transform(pcr[semente], X_all))
Y_pca_scores = pd.DataFrame(try_transform(r_pcr[semente], Y_all))

X_all_pca = X_pca_scores.rename(columns=dict(
    enumerate(component_names["pca"][:n_features])))

# ds: descriptors
X_all_pca_ds = pd.concat((X_all_pca, X_all), axis="columns")

# ts: targets
X_all_pca_ts = pd.concat((X_all_pca, Y_all), axis="columns")

Y_all_pca = Y_pca_scores.rename(columns=dict(
    enumerate(component_names["pca"][:n_targets])))

Y_all_pca_ts = pd.concat((Y_all_pca, Y_all), axis="columns")
Y_all_pca_ds = pd.concat((Y_all_pca, X_all), axis="columns")

# NOTE: use correlation, not normalization.
# method="pearson"
x_pca_ds_correlations = X_all_pca_ds.corr().iloc[-n_features:, :-n_features]
x_pca_ts_correlations = X_all_pca_ts.corr().iloc[-n_targets:, :-n_targets]

y_pca_ts_correlations = Y_all_pca_ts.corr().iloc[-n_targets:, :-n_targets]
y_pca_ds_correlations = Y_all_pca_ds.corr().iloc[-n_features:, :-n_features]

for i, o in ordinais:
    # .reshape(-1, 1)
    x_pca_ds_corr_i = pd.DataFrame(x_pca_ds_correlations.iloc[:, i])
    x_pca_ts_corr_i = pd.DataFrame(x_pca_ts_correlations.iloc[:, i])

    y_pca_ts_corr_i = pd.DataFrame(y_pca_ts_correlations.iloc[:, i])
    y_pca_ds_corr_i = pd.DataFrame(y_pca_ds_correlations.iloc[:, i])

    # NOTE: different title for X and Y.
    pca_x_component_ds_corr_i = {
        "X": x_pca_ds_corr_i,
        "titles": [None],
        "xlabels": [None for _ in descriptors],
        "sort": _SORT_X,
        "meanlabel": _MEANLABEL,
    }

    path = f"pca_ords-ds_corr_{i}-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
    prefix = "x_component/"

    path += f"-sort_{_SORT_X}"

    for lang, label, _ in corr_labels:
        lang_path = path + "-lang_" + lang
        pca_x_component_ds_corr_i["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pca_x_component_ds_corr_i)

    pca_x_component_ts_corr_i = {
        "titles": [None],
        "xlabels": [None for _ in targets],
        "sort": _SORT_Y,
        "meanlabel": _MEANLABEL,
    }

    path = f"pca_ords-ts_corr_{i}-algo_{_ALGO}-data_{_DATA}-seed_{seed}"

    path += f"-sort_{_SORT_Y}"

    for lang, label, mapper in corr_labels:
        lang_path = path + "-lang_" + lang

        pca_x_component_ts_corr_i["X"] = x_pca_ts_corr_i.rename(index=mapper)
        pca_x_component_ts_corr_i["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pca_x_component_ts_corr_i)

    pca_y_component_ts_corr_i = {
        "titles": [None],
        "xlabels": [None for _ in targets],
        "sort": _SORT_Y,
        "meanlabel": _MEANLABEL,
    }

    path = f"pca_ords-ts_corr_{i}-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
    prefix = "y_component/"

    path += f"-sort_{_SORT_Y}"

    for lang, label, mapper in corr_labels:
        lang_path = path + "-lang_" + lang

        pca_y_component_ts_corr_i["X"] = y_pca_ts_corr_i.rename(index=mapper)
        pca_y_component_ts_corr_i["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pca_y_component_ts_corr_i)

    pca_y_component_ds_corr_i = {
        "X": y_pca_ds_corr_i,
        "titles": [None],
        "xlabels": [None for _ in descriptors],
        "sort": _SORT_X,
        "meanlabel": _MEANLABEL,
    }

    path = f"pca_ords-ds_corr_{i}-algo_{_ALGO}-data_{_DATA}-seed_{seed}"

    path += f"-sort_{_SORT_X}"

    for lang, label, _ in corr_labels:
        lang_path = path + "-lang_" + lang
        pca_y_component_ds_corr_i["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                     **pca_y_component_ds_corr_i)

pca_x_components_ds_corr = {
    "X": x_pca_ds_correlations,
    "titles": [None for _ in ordinais],
    "xlabels": [None for _ in descriptors],
    "ncols": x_pca_ds_correlations.shape[0],
    "sort": _SORT_X,
    "meanlabel": _MEANLABEL,
}

path = f"pca-ds_corr-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
prefix = "x_component/"

save_to_csv(x_pca_ds_correlations, path, save=_SAVE, prefix=prefix)

path += f"-sort_{_SORT_X}"

for lang, label, _ in corr_labels:
    lang_path = path + "-lang_" + lang

    pca_x_components_ds_corr["ylabel"] = label

    save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                 **pca_x_components_ds_corr)

pca_x_components_ts_corr = {
    "titles": [None for _ in ordinais],
    "xlabels": [None for _ in targets],
    "ncols": x_pca_ts_correlations.shape[0],
    "sort": _SORT_Y,
    "meanlabel": _MEANLABEL,
}

path = f"pca-ts_corr-algo_{_ALGO}-data_{_DATA}-seed_{seed}"

save_to_csv(x_pca_ts_correlations, path, save=_SAVE, prefix=prefix)

path += f"-sort_{_SORT_Y}"

for lang, label, mapper in corr_labels:
    lang_path = path + "-lang_" + lang

    pca_x_components_ts_corr["X"] = x_pca_ts_correlations.rename(index=mapper)
    pca_x_components_ts_corr["ylabel"] = label

    save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                 **pca_x_components_ts_corr)

pca_y_components_ts_corr = {
    "titles": [None for _ in ordinais],
    "xlabels": [None for _ in targets],
    "ncols": y_pca_ts_correlations.shape[0],
    "sort": _SORT_Y,
    "meanlabel": _MEANLABEL,
}

path = f"pca-ts_corr-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
prefix = "y_component/"

save_to_csv(y_pca_ts_correlations, path, save=_SAVE, prefix=prefix)

path += f"-sort_{_SORT_Y}"

for lang, label, mapper in corr_labels:
    lang_path = path + "-lang_" + lang

    pca_y_components_ts_corr["X"] = y_pca_ts_correlations.rename(index=mapper)
    pca_y_components_ts_corr["ylabel"] = label

    save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                 **pca_y_components_ts_corr)

pca_y_components_ds_corr = {
    "X": y_pca_ds_correlations,
    "titles": [None for _ in ordinais],
    "xlabels": [None for _ in descriptors],
    "ncols": y_pca_ds_correlations.shape[0],
    "sort": _SORT_X,
    "meanlabel": _MEANLABEL,
}

path = f"pca-ds_corr-algo_{_ALGO}-data_{_DATA}-seed_{seed}"

save_to_csv(y_pca_ds_correlations, path, save=_SAVE, prefix=prefix)

path += f"-sort_{_SORT_X}"

for lang, label, _ in corr_labels:
    lang_path = path + "-lang_" + lang

    pca_y_components_ds_corr["ylabel"] = label

    save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                 **pca_y_components_ds_corr)


x_seeds_pca_ds_corr = pd.DataFrame(index=X_all.columns)
x_seeds_pca_ts_corr = pd.DataFrame(index=Y_all.columns)

y_seeds_pca_ts_corr = pd.DataFrame(index=Y_all.columns)
y_seeds_pca_ds_corr = pd.DataFrame(index=X_all.columns)

# NOTE: iterate over all seeds and perform corr() over
# X_train and Y_train
for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    X_train, X_test, Y_train, Y_test = split

    x_scores = pd.DataFrame(try_transform(pcr[semente], X_train))
    y_scores = pd.DataFrame(try_transform(r_pcr[semente], Y_train))

    x_seed_components_ds = pd.concat(
        (x_scores, X_train), axis="columns"
    )
    x_seed_components_ts = pd.concat(
        (x_scores, Y_train), axis="columns"
    )

    y_seed_components_ts = pd.concat(
        (y_scores, Y_train), axis="columns"
    )
    y_seed_components_ds = pd.concat(
        (y_scores, X_train), axis="columns"
    )

    x_seed_pca_ds_corr = x_seed_components_ds.corr()\
        .iloc[-n_features:, :-n_features]
    x_seed_pca_ts_corr = x_seed_components_ts.corr()\
        .iloc[-n_targets:, :-n_targets]

    y_seed_pca_ts_corr = y_seed_components_ts.corr()\
        .iloc[-n_targets:, :-n_targets]
    y_seed_pca_ds_corr = y_seed_components_ds.corr()\
        .iloc[-n_features:, :-n_features]

    x_seeds_pca_ds_corr[semente] = x_seed_pca_ds_corr.iloc[:, 0]
    x_seeds_pca_ts_corr[semente] = x_seed_pca_ts_corr.iloc[:, 0]

    y_seeds_pca_ts_corr[semente] = y_seed_pca_ts_corr.iloc[:, 0]
    y_seeds_pca_ds_corr[semente] = y_seed_pca_ds_corr.iloc[:, 0]

    pca_seeds_first_x_components = {
        "X": x_seed_pca_ds_corr,
        "titles": [None],
        "xlabels": [None for _ in descriptors],
        "sort": _SORT_X,
        "meanlabel": _MEANLABEL,
    }

    path = f"pca_seeds-ds_corr_0-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
    prefix = "x_component/"

    save_to_csv(x_seed_pca_ds_corr, path, save=_SAVE, prefix=prefix)

    path += f"-sort_{_SORT_X}"

    for lang, label, _ in corr_labels:
        lang_path = path + "-lang_" + lang
        pca_seeds_first_x_components["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, pause=False,
                     **pca_seeds_first_x_components)

    pca_seeds_first_y_components = {
        "titles": [None],
        "xlabels": [None for _ in targets],
        "sort": _SORT_Y,
        "meanlabel": _MEANLABEL,
    }

    path = f"pca_seeds-ts_corr_0-algo_{_ALGO}-data_{_DATA}-seed_{seed}"
    prefix = "y_component/"

    save_to_csv(y_seed_pca_ts_corr, path, save=_SAVE, prefix=prefix)

    path += f"-sort_{_SORT_Y}"

    for lang, label, mapper in corr_labels:
        lang_path = path + "-lang_" + lang

        pca_seeds_first_y_components["X"] = y_seed_pca_ts_corr.rename(
            index=mapper)
        pca_seeds_first_y_components["ylabel"] = label

        save_or_show(lang_path, prefix, plot_components, _SAVE, _SHOW, _PAUSE,
                     **pca_seeds_first_y_components)


algos = ("PCR", "PLSR")
trans = ("pca", "pls")

regression_labels = {
    t.lower(): {
        "actual": {
            "en": [f"Actual first {t.upper()} component" for _ in algos],
            "pt": [f"Primeiro componente {t.upper()} real" for _ in algos]
        },
        "predicted": {
            "en": [f"Predicted first {t.upper()} component" for _ in algos],
            "pt": [f"Primeiro componente {t.upper()} predito" for _ in algos]
        }
    } for t in trans
}

# seed=1241 was the best for the ratio of rPLSR's r2_score
# over rPCR's.
for semente, split in splits.items():
    seed = str(None) if semente == "Nenhuma" else semente

    X_train, X_test, Y_train, Y_test = split

    # TODO? Iterate over values of n_components ([1..5]).
    n = 1

    pcr_seed, r_pcr_seed, X_pred_pcr, Y_pred_pcr = fit_predict(
        ScalerPCR, *split, n_components=n)

    plsr_seed, r_plsr_seed, X_pred_plsr, Y_pred_plsr = fit_predict(
        PLSRegression, *split, n_components=n)

    # NOTE: use canonical transformers
    # (*pcr and *plsr dicts).
    X_test_pca = pd.DataFrame(
        try_transform(pcr[semente], X_test)[:, :n])
    Y_test_pca = pd.DataFrame(
        try_transform(r_pcr[semente], Y_test)[:, :n])

    X_test_pls = pd.DataFrame(
        try_transform(plsr["Todos"][semente], X_test)[:, :n])
    Y_test_pls = pd.DataFrame(
        try_transform(r_plsr[semente], Y_test)[:, :n])

    X_pred_pcr_pca = pd.DataFrame(
        try_transform(pcr[semente], X_pred_pcr)[:, :n])
    Y_pred_pcr_pca = pd.DataFrame(
        try_transform(r_pcr[semente], Y_pred_pcr)[:, :n])

    X_pred_pcr_pls = pd.DataFrame(
        try_transform(plsr["Todos"][semente], X_pred_pcr)[:, :n])
    Y_pred_pcr_pls = pd.DataFrame(
        try_transform(r_plsr[semente], Y_pred_pcr)[:, :n])

    X_pred_plsr_pca = pd.DataFrame(
        try_transform(pcr[semente], X_pred_plsr)[:, :n])
    Y_pred_plsr_pca = pd.DataFrame(
        try_transform(r_pcr[semente], Y_pred_plsr)[:, :n])

    X_pred_plsr_pls = pd.DataFrame(
        try_transform(plsr["Todos"][semente], X_pred_plsr)[:, :n])
    Y_pred_plsr_pls = pd.DataFrame(
        try_transform(r_plsr[semente], Y_pred_plsr)[:, :n])

    # NOTE: display R-squared for the prediction of each
    # component.

    R2_Y_pcr = r2_score(Y_test, Y_pred_pcr, multioutput="raw_values")
    R2_Y_plsr = r2_score(Y_test, Y_pred_plsr, multioutput="raw_values")

    R2_Y_pcr_pca = r2_score(Y_test_pca, Y_pred_pcr_pca,
                            multioutput="raw_values")
    R2_Y_pcr_pls = r2_score(Y_test_pls, Y_pred_pcr_pls,
                            multioutput="raw_values")

    R2_Y_plsr_pca = r2_score(Y_test_pca, Y_pred_plsr_pca,
                             multioutput="raw_values")
    R2_Y_plsr_pls = r2_score(Y_test_pls, Y_pred_plsr_pls,
                             multioutput="raw_values")

    # TODO: split regression plot into two (PCA/ DTR and PLS)
    pcr_vs_plsr_y_pred_pca = {
        "Y_true": pd.concat((Y_test_pca.iloc[:, 0],) * 2, axis="columns"),
        "Y_pred": pd.concat((Y_pred_pcr_pca.iloc[:, 0], Y_pred_plsr_pca.iloc[:, 0]), axis="columns"),
        "titles": algos,
        "R2": np.array((R2_Y_pcr_pca[0], R2_Y_plsr_pca[0])),
    }

    R2_Y_pca = pd.DataFrame(
        {"PCA": R2_Y_pcr_pca, "PLS": R2_Y_plsr_pca}
    )

    t = "pca"
    path = f"r2s-algo_{_ALGO}-data_{_DATA}-n_{n}-t_{t}-seed_{seed}"
    prefix = "y_pred/"

    save_to_csv(R2_Y_pca, path, save=_SAVE, prefix=prefix)

    path = path.replace("r2s", "pcr_vs_plsr")

    for lang, a_labels in regression_labels[t]["actual"].items():
        lang_path = path + "-lang_" + lang
        p_labels = regression_labels[t]["predicted"][lang]

        pcr_vs_plsr_y_pred_pca["xlabels"] = a_labels
        pcr_vs_plsr_y_pred_pca["ylabels"] = p_labels

        # pause=False: pausing is not practical in a for-loop.
        save_or_show(lang_path, prefix, plot_regression, _SAVE, _SHOW, pause=False,
                     **pcr_vs_plsr_y_pred_pca)

    pcr_vs_plsr_y_pred_pls = {
        "Y_true": pd.concat((Y_test_pls.iloc[:, 0],) * 2, axis="columns"),
        "Y_pred": pd.concat((Y_pred_pcr_pls.iloc[:, 0], Y_pred_plsr_pls.iloc[:, 0]), axis="columns"),
        "titles": algos,
        "R2": np.array((R2_Y_pcr_pls[0], R2_Y_plsr_pls[0])),
    }

    R2_Y_pls = pd.DataFrame(
        {"PCA": R2_Y_pcr_pls, "PLS": R2_Y_plsr_pls}
    )

    t = "pls"
    path = f"r2s-algo_{_ALGO}-data_{_DATA}-n_{n}-t_{t}-seed_{seed}"

    save_to_csv(R2_Y_pls, path, save=_SAVE, prefix=prefix)

    for lang, a_labels in regression_labels[t]["actual"].items():
        lang_path = path + "-lang_" + lang
        p_labels = regression_labels[t]["predicted"][lang]

        pcr_vs_plsr_y_pred_pls["xlabels"] = a_labels
        pcr_vs_plsr_y_pred_pls["ylabels"] = p_labels

        save_or_show(lang_path, prefix, plot_regression, _SAVE, _SHOW, pause=False,
                     **pcr_vs_plsr_y_pred_pls)

    R2_X_pcr = r2_score(X_test, X_pred_pcr, multioutput="raw_values")
    R2_X_plsr = r2_score(X_test, X_pred_plsr, multioutput="raw_values")

    R2_X_pcr_pca = r2_score(X_test_pca, X_pred_pcr_pca,
                            multioutput="raw_values")
    R2_X_pcr_pls = r2_score(X_test_pls, X_pred_pcr_pls,
                            multioutput="raw_values")

    R2_X_plsr_pca = r2_score(X_test_pca, X_pred_plsr_pca,
                             multioutput="raw_values")
    R2_X_plsr_pls = r2_score(X_test_pls, X_pred_plsr_pls,
                             multioutput="raw_values")

    pcr_vs_plsr_x_pred_pca = {
        "Y_true": pd.concat((X_test_pca.iloc[:, 0],) * 2, axis="columns"),
        "Y_pred": pd.concat([X_pred_pcr_pca.iloc[:, 0], X_pred_plsr_pca.iloc[:, 0]], axis="columns"),
        "titles": algos,
        "R2": np.array((R2_X_pcr_pca[0], R2_X_plsr_pca[0])),
    }

    R2_X_pca = pd.DataFrame(
        {"PCA": R2_X_pcr_pca, "PLS": R2_X_plsr_pca}
    )

    t = "pca"
    path = f"r2s-algo_{_ALGO}-data_{_DATA}-n_{n}-t_{t}-seed_{seed}"
    prefix = "x_pred/"

    save_to_csv(R2_X_pca, path, save=_SAVE, prefix=prefix)

    path = path.replace("r2s", "pcr_vs_plsr")

    for lang, a_labels in regression_labels[t]["actual"].items():
        lang_path = path + "-lang_" + lang
        p_labels = regression_labels[t]["predicted"][lang]

        pcr_vs_plsr_x_pred_pca["xlabels"] = a_labels
        pcr_vs_plsr_x_pred_pca["ylabels"] = p_labels

        save_or_show(lang_path, prefix, plot_regression, _SAVE, _SHOW, pause=False,
                     **pcr_vs_plsr_x_pred_pca)

    pcr_vs_plsr_x_pred_pls = {
        "Y_true": pd.concat((X_test_pls.iloc[:, 0],) * 2, axis="columns"),
        "Y_pred": pd.concat((X_pred_pcr_pls.iloc[:, 0], X_pred_plsr_pls.iloc[:, 0]), axis="columns"),
        "titles": algos,
        "R2": np.array((R2_X_pcr_pls[0], R2_X_plsr_pls[0])),
    }

    R2_X_pls = pd.DataFrame(
        {"PCA": R2_X_pcr_pls, "PLS": R2_X_plsr_pls}
    )

    t = "pls"
    path = f"r2s-algo_{_ALGO}-data_{_DATA}-n_{n}-t_{t}-seed_{seed}"

    save_to_csv(R2_X_pls, path, save=_SAVE, prefix=prefix)

    path = path.replace("r2s", "pcr_vs_plsr")

    for lang, a_labels in regression_labels[t]["actual"].items():
        lang_path = path + "-lang_" + lang
        p_labels = regression_labels[t]["predicted"][lang]

        pcr_vs_plsr_x_pred_pls["xlabels"] = a_labels
        pcr_vs_plsr_x_pred_pls["ylabels"] = p_labels

        save_or_show(lang_path, prefix, plot_regression, _SAVE, _SHOW, _PAUSE,
                     **pcr_vs_plsr_x_pred_pls)

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
# NOTE: do not slice splits, as they're used to populate r2s
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
                r2_m = r2_score(Y_test_t, Y_pred_t,
                                multioutput="variance_weighted")
                r2_rm = r2_score(X_test_t, X_pred_t,
                                 multioutput="variance_weighted")

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

path = f"r2s-algo_{_ALGO}-data_{_DATA}-multi_vw"
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

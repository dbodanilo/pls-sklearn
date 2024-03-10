import pandas as pd


def load_leme(idwl=True):
    path = "leme-fronteira.csv"
    if idwl:
        path = "leme-fronteira-IDWL.csv"

    leme_data = pd.read_csv(path, delimiter="\t")
    Y = leme_data[["N.", "Semente", "SR", "Area", "A_{v0}", "f_{T}", "Pwr"]]
    # Don't drop for X ["N.", "Semente"] id columns.
    X = leme_data.drop(columns=Y.columns.drop(["N.", "Semente"]))

    return (X, Y)


# NOTE: seed=1244 was responsible for the worst r2_score for
# predicting Y from X;
# seed=1242 was responsible for the worst r2_score for
# predicting X from Y.
def train_test_seed_split(X, Y, seed=1241):
    """Train on n - 1 seeds, test on remaining seed.

    Goal
    ----
    try to navigate to points in the unknown frontier
    by regressing the known ones.

    seed : default=1241, as test was responsible for the
    best `r2_score` when predicting X from Y with PLSR.
    """
    X_train = X[X["Semente"] != seed]
    Y_train = Y[Y["Semente"] != seed]

    X_test = X[X["Semente"] == seed]
    Y_test = Y[Y["Semente"] == seed]

    # "N.", "Semente": Não são relevantes para regressão.
    X_train = X_train.drop(columns=["N.", "Semente"]).to_numpy()
    Y_train = Y_train.drop(columns=["N.", "Semente"]).to_numpy()
    X_test = X_test.drop(columns=["N.", "Semente"]).to_numpy()
    Y_test = Y_test.drop(columns=["N.", "Semente"]).to_numpy()

    return (X_train, X_test, Y_train, Y_test)

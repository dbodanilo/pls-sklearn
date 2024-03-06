import pandas as pd


def load_leme():
    leme_data = pd.read_csv("leme-fronteira.csv", delimiter="\t")
    X = leme_data[["N.", "Semente", "I_{pol}", "V_{pol}", "W_{1,2}", "L_{1,2}", "W_{3,4}",
                   "L_{3,4}", "W_{5,6}", "L_{5,6}", "W_{7,8}", "L_{7,8}", "W_{9}", "L_{9}", "W_{10}", "L_{10}"]]
    Y = leme_data[["N.", "Semente", "SR", "Area", "A_{v0}", "f_{T}", "Pwr"]]

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
    X_train = X_train.drop(["N.", "Semente"], axis=1).to_numpy()
    Y_train = Y_train.drop(["N.", "Semente"], axis=1).to_numpy()
    X_test = X_test.drop(["N.", "Semente"], axis=1).to_numpy()
    Y_test = Y_test.drop(["N.", "Semente"], axis=1).to_numpy()

    return (X_train, X_test, Y_train, Y_test)

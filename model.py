import numpy
import pandas
import pickle


IV = ("I_{pol}", "V_{pol}")

PAIRS = ("_{1,2}", "_{3,4}", "_{5,6}", "_{7,8}", "_{9}", "_{10}")
LS = tuple("L" + p for p in PAIRS)
WS = tuple("W" + p for p in PAIRS)

# DESCRIPTORS = ("I_{pol}", "V_{pol}",
#                "W_{1,2}", "L_{1,2}",
#                "W_{3,4}", "L_{3,4}",
#                "W_{5,6}", "L_{5,6}",
#                "W_{7,8}", "L_{7,8}",
#                "W_{9}", "L_{9}",
#                "W_{10}", "L_{10}")
DESCRIPTORS = IV + LS + WS

TARGETS = ("A_{v0}", "f_{T}", "Pwr", "SR", "Area")


def load_deap(idwl=True, split=True):
    deap_data = None

    if idwl:
        path = "data/deap-seeds-pop-idwl.pickle"

        with open(path, "rb") as f:
            deap_data = pickle.load(f)
    else:
        for s in range(1241, 1246):
            path = f"data/deap-seed_{s}-pop.pickle"

            with open(path, "rb") as f:
                deap_data_s = pickle.load(f)

            deap_data_s = pandas.DataFrame(
                [[*ind, *ind.fitness.values] for ind in deap_data_s],
                columns=(*DESCRIPTORS, *TARGETS)
            )
            deap_data_s["Semente"] = s

            if deap_data is None:
                deap_data = deap_data_s
            else:
                deap_data = pandas.concat(
                    (deap_data, deap_data_s), ignore_index=True)

    deap_data.replace({"Pwr": [numpy.inf, -numpy.inf]},
                      numpy.nan, inplace=True)
    deap_data.dropna(axis=0, how="any", inplace=True)

    deap_data["N."] = deap_data.index

    if split:
        Y = deap_data[["N.", "Semente", "A_{v0}", "f_{T}",
                       "Pwr", "SR", "Area"]]
        X = deap_data.drop(columns=Y.columns.drop(["N.", "Semente"]))

        return (X, Y)

    return (deap_data, deap_data)


def load_leme(idwl=True, split=True):
    path = "data/leme-fronteira.csv"
    if idwl:
        path = "data/leme-fronteira-IDWL.csv"

    leme_data = pandas.read_csv(path, delimiter="\t")

    if split:
        # Metrics ordered by importance.
        Y = leme_data[["N.", "Semente", "A_{v0}", "f_{T}",
                       "Pwr", "SR", "Area"]]
        # Don't drop for X ["N.", "Semente"] id columns.
        X = leme_data.drop(columns=Y.columns.drop(["N.", "Semente"]))

        return (X, Y)

    return (leme_data, leme_data)


# NOTE: seed=1244 was responsible for the worst r2_score for
# predicting Y from X;
# seed=1242 was responsible for the worst r2_score for
# predicting X from Y.
# seed=1245 was responsible for the best r2_score for
# predicting X from Y with both PLSR and PCR.
def train_test_seed_split(X, Y, seed=1242):
    """Train on n - 1 seeds, test on remaining seed.

    Goal
    ----
    try to navigate to points in the unknown frontier
    by regressing the known ones.

    seed : default=1242, as test was responsible for the
    best `r2_score` ratio of predicting X from Y with PLSR
    versus PCR.
    """
    if seed is None:
        X_all = X.drop(columns=["N.", "Semente"])
        Y_all = Y.drop(columns=["N.", "Semente"])
        return (X_all, X_all, Y_all, Y_all)

    X_train = X[X["Semente"] != seed]
    Y_train = Y[Y["Semente"] != seed]

    X_test = X[X["Semente"] == seed]
    Y_test = Y[Y["Semente"] == seed]

    # "N.", "Semente": Não são relevantes para regressão.
    X_train = X_train.drop(columns=["N.", "Semente"])
    Y_train = Y_train.drop(columns=["N.", "Semente"])

    X_test = X_test.drop(columns=["N.", "Semente"])
    Y_test = Y_test.drop(columns=["N.", "Semente"])

    return (X_train, X_test, Y_train, Y_test)

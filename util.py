import os

import numpy as np
import pandas as pd

from datetime import datetime
from glob import glob


def detexify(path):
    return path.replace("$", "").replace("{", "").replace("}", "")


def get_globs(path, prefix, exts):
    # [0-9] ensures paths with the same suffix are not considered equal.
    # Example: `./*/*plsr-predictions.png` and `./*/*pcr_vs_plsr-predictions.png`.
    return set(g for ext in exts for g in glob(f"{prefix}*[0-9]_{path}{ext}"))


def get_paths(path, prefix="./out/", exts=[".pdf", ".png"], timestamp=True):
    if timestamp:
        _now = datetime.now()
        today = _now.strftime("%Y-%m-%d")

        # YYYY-mm-dd/
        prefix += f"{today}/"

        if not os.path.exists(prefix):
            os.makedirs(prefix, exist_ok=True)

        # YYYY-mm-dd_HH-mm_<path>
        path = today + _now.strftime("_%H-%M_") + path
    return ["".join([prefix + path, e]) for e in exts], prefix, exts


def fit_predict(model, X_train, X_test, Y_train, Y_test, **mkargs):
    m = model(**mkargs).fit(X_train, Y_train)
    Y_pred = pd.DataFrame(m.predict(X_test), columns=Y_train.columns)

    # reverse model (predict sizing based on target metrics)
    rm = model(**mkargs).fit(Y_train, X_train)
    X_pred = pd.DataFrame(rm.predict(Y_test), columns=X_train.columns)

    return m, rm, X_pred, Y_pred


def fit_predict_try_transform(model, X_train, X_test, Y_train, Y_test, **mkargs):
    m, rm, X_pred, Y_pred = fit_predict(
        model, X_train, X_test, Y_train, Y_test, **mkargs)

    X_test_t = pd.DataFrame(try_transform(m, X_test))
    Y_test_t = pd.DataFrame(try_transform(rm, Y_test))

    X_pred_t = pd.DataFrame(try_transform(m, X_pred))
    Y_pred_t = pd.DataFrame(try_transform(rm, Y_pred))

    return X_test_t, X_pred, X_pred_t, Y_test_t, Y_pred, Y_pred_t


def latexify(strs):
    return np.fromiter((f"${s}$" if s.find("_") > 0 else s for s in strs), dtype="<U21")


# format "{:.4f}" was the highest one not to vary on
# equivalent runs.
def save_to_csv(X, path, exts=[".csv"], sep="\t", float_format="{:.4f}".format, **kwargs):
    paths, prefix, exts = get_paths(path, exts=exts)
    globs = get_globs(path, prefix, exts)

    path = paths[0]

    # Only generate it once.
    if not any(os.path.exists(glob) for glob in globs):
        X.to_csv(path, sep=sep, float_format=float_format, **kwargs)


    # Only generate it once.
    if not any(os.path.exists(path) for path in globs) or show:
        fig = plot(**kwargs)

        if show:
            fig.show()
        else:
            for path in paths:
                fig.savefig(path)

        if pause:
            input("Press Enter to continue...")


def try_attr(model, a):
    if hasattr(model, a):
        return model.a

    if hasattr(model, "steps"):
        for _, step in model.steps:
            if hasattr(step, a):
                return getattr(step, a)

    name = type(model).__name__
    raise AttributeError(f"{name} object has no attribute '{a}'")


def try_transform(model, X):
    if hasattr(model, "transform"):
        return model.transform(X)

    if hasattr(model, "steps"):
        X_t = X
        for _, step in model.steps:
            # TODO: check if `break` or `continue` would be
            # most appropriate, as in `ScalerPCR` only the
            # `LinearRegression` step has no `transform`.
            if not hasattr(step, "transform"):
                continue

            X_t = step.transform(X_t)

        return X_t

    name = type(model).__name__
    raise AttributeError(f"{name} object has no method 'transform'")

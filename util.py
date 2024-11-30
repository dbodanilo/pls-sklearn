import os

import matplotlib.pyplot as plt
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


def get_paths(path, outdir="./out/", prefix="", exts=[".pdf", ".png"], timestamp=True):
    if timestamp:
        _now = datetime.now()
        today = _now.strftime("%Y-%m-%d")

        # TODO: use different folders for different data,
        # e.g., components, predictions, regression.
        # YYYY-mm-dd/
        outdir += f"{today}/"

        # YYYY-mm-dd_HH-mm_<path>
        path = today + _now.strftime("_%H-%M_") + path

    prefix = outdir + prefix

    if not os.path.exists(prefix):
        os.makedirs(prefix, exist_ok=True)

    return ["".join([prefix + path, e]) for e in exts], prefix, exts


def fit_predict(model, X_train, X_test, Y_train, Y_test, **mkargs):
    m = model(**mkargs).fit(X_train, Y_train)
    Y_pred = pd.DataFrame(m.predict(X_test), columns=Y_train.columns)

    # reverse model (predict sizing based on target metrics)
    rm = model(**mkargs).fit(Y_train, X_train)
    X_pred = pd.DataFrame(rm.predict(Y_test), columns=X_train.columns)

    return m, rm, X_pred, Y_pred


def latexify(strs):
    return np.fromiter((f"${s}$" if s.find("_") > 0 else s for s in strs), dtype="<U21")


# format "{:.4f}" was the highest one not to vary on
# equivalent runs.
def save_to_csv(X, path, save=True, prefix="", exts=[".csv"], sep="\t", float_format="{:.4f}".format, **kwargs):
    paths, prefix, exts = get_paths(path, prefix=prefix, exts=exts)
    globs = get_globs(path, prefix, exts)

    path = paths[0]

    # Only generate it once.
    if save and not any(os.path.exists(glob) for glob in globs):
        X.to_csv(path, sep=sep, float_format=float_format, **kwargs)


def save_or_show(path, prefix, plot, save=True, show=False, pause=False, **kwargs):
    paths, prefix, exts = get_paths(path, prefix=prefix)
    globs = get_globs(path, prefix, exts)

    # Only generate it once.
    if show or (save and not any(os.path.exists(path) for path in globs)):
        fig = plot(**kwargs)

        if save:
            for path in paths:
                fig.savefig(path)

        if show:
            fig.show()

        if pause:
            input("Press Enter to continue...")

        # close when not showing, or after pausing.
        if not show or pause:
            plt.close("all")


def try_attr(model, a):
    if hasattr(model, a):
        return getattr(model, a)

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

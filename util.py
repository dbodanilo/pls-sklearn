import os

from datetime import datetime
from glob import glob


def get_globs(path, prefix, exts):
    return set(g for ext in exts for g in glob(f"{prefix}*{path}{ext}"))


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


def fit_predict_try_transform(model, X_train, X_test, Y_train, Y_test, **mkargs):
    m = model(**mkargs).fit(X_train, Y_train)

    # reverse model (predict sizing based on target metrics)
    rm = model(**mkargs).fit(Y_train, X_train)

    X_test_t = try_transform(m, X_test)
    Y_test_t = try_transform(rm, Y_test)

    Y_pred = m.predict(X_test)
    X_pred = rm.predict(Y_test)

    X_pred_t = try_transform(m, X_pred)
    Y_pred_t = try_transform(rm, Y_pred)

    return X_test_t, X_pred, X_pred_t, Y_test_t, Y_pred, Y_pred_t


def latexify(strs):
    return [f"${s}$" if s.find("_") > 0 else s for s in strs]


def show_or_save(paths, globs, show, plot, *args, **kwargs):
    # Only generate it once.
    if not any(os.path.exists(path) for path in globs) or show:
        fig = plot(*args, **kwargs)

        if show:
            fig.show()
            input("Press Enter to continue...")
        else:
            for path in paths:
                fig.savefig(path)


def try_transform(model, X):
    if hasattr(model, "transform"):
        return model.transform(X)

    elif hasattr(model, "steps"):
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
    raise AttributeError(f"{name} has no 'transform' method")

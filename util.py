def fig_paths(path, prefix="./out/", exts=[".pdf", ".png"]):
    return ["".join([prefix + path, e]) for e in exts]


def fit_predict_try_transform(model, X_train, X_test, Y_train, Y_test):
    m = model().fit(X_train, Y_train)

    # reverse model (predict sizing based on target metrics)
    rm = model().fit(Y_train, X_train)

    X_test_t = try_transform(m, X_test)
    Y_test_t = try_transform(rm, Y_test)

    Y_pred = m.predict(X_test)
    X_pred = rm.predict(Y_test)

    X_pred_t = try_transform(m, X_pred)
    Y_pred_t = try_transform(rm, Y_pred)

    return X_test_t, X_pred_t, Y_test_t, Y_pred_t


def try_transform(model, X):
    if hasattr(model, "transform"):
        X_t = model.transform(X)

    elif hasattr(model, "steps"):
        X_t = X
        for _, step in model.steps:
            # TODO: check if `break` or `continue` would be
            # most appropriate, as in `ScalerPCR` only the
            # `LinearRegression` step has no `transform`.
            if not hasattr(step, "transform"):
                continue

            X_t = step.transform(X_t)

    else:
        label = type(model).__name__
        raise AttributeError(f"{label} has no 'transform' method")

    return X_t

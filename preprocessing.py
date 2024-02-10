def scale_test(X, mean, std):
    X_scaled = X - mean
    X_scaled /= std

    return X_scaled


def scale_transform(model, X, mean, std):
    return model.transform(scale_test(X, mean, std))

def scale_test(X, mean, std):
    X_scaled = X - mean
    X_scaled /= std

    return X_scaled


def scale_transform(scaler, model, X):
    return model.transform(scaler.transform(X))

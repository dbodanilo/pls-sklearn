from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# TODO: avoid dummy `n_components`.
def ScalerSVR(kernel="rbf", degree=3, gamma="scale",
              coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, n_components=1):
    if n_components is not None:
        if n_components == 1:
            kernel = "linear"
        elif n_components < 5:
            kernel = "poly"
            degree = n_components
            coef0 = 1.0
        else:
            kernel = "rbf"

    return make_pipeline(StandardScaler(),
                         SVR(kernel=kernel, degree=degree,
                             gamma=gamma, coef0=coef0,
                             tol=tol, C=C, epsilon=epsilon))

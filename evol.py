import numpy as np

from scipy.linalg import pinv

from sklearn.preprocessing import normalize, scale


UV_EVOL = np.array([0.10069952,  0.07932159, -0.12746936, -0.38998521,  0.02003923, -0.14191234,
                    -0.0480223, -0.11860172,  0.25530319, -0.15315524, -0.27860995,  0.03321066,
                    0.06471584,  0.12206272, -0.47155087, -0.03701225, -0.03100685, -0.08405825,
                    0.03494814,  0.0123027, -0.02752777,  0.38195902, -0.09674531, -0.00537525,
                    0.01402724,  0.04784506, -0.38913367, -0.99950046, -1., -1.,
                    -0.81233796, -0.75537525, -0.91616727, -0.30749526, -0.53608318, -0.56908043,
                    -0.06996026, -0.01343946,  0.2800989, -0.44373839, -0.2334909, -0.64359148,
                    -0.0505262,  0.05374062,  0.07531023,  0.98872992, -0.67944099,  1.,
                    -0.01902401, -0.46290354,  0.2357227, -0.07489049, -0.16771531,  0.06834561,
                    -0.07506276,  0.04118705, -0.21227254])


class Evol():
    """Evolutionary regression.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in
        `[1, min(n_samples, n_features, n_targets)]`.

    Attributes
    ----------
    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_targets, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_targets, n_features)
        The coefficients of the linear model such that `Y`
        is approximated as
        `Y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `Y`
        is approximated as
        `Y = X @ coef_.T + intercept_`.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, Y, uv):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the
            number of samples and `n_features` is the number
            of predictors.

        Y : array-like of shape (n_samples,) or
        (n_samples, n_targets)
            Target vectors, where `n_samples` is the number
            of samples and `n_targets` is the number of
            response variables.


        uv : ndarray of shape
        ((n_features + n_targets) * n_components,)
            The flattened and concatenated weights/rotations
            of `X` (U) and `Y` (V).
        """
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        n_components = self.n_components

        assert len(uv) == (p + q) * n_components, "Invalid UV dimensions."

        # TODO: Adaptar saída do PLS para meta-heurística,
        # ainda não apresenta resultados úteis.
        self.x_weights_ = np.array(
            uv[:(p * n_components)]).reshape(p, n_components)  # U
        self.y_weights_ = np.array(
            uv[-(q * n_components):]).reshape(q, n_components)  # V

        # |Uk| == |Vk| == 1
        self.x_weights_ = normalize(self.x_weights_, axis=1)
        self.y_weights_ = normalize(self.y_weights_, axis=1)

        self.x_scores_ = np.zeros((n, n_components))  # Xi
        self.y_scores_ = np.zeros((n, n_components))  # Omega

        self.x_loadings_ = np.zeros((p, n_components))  # Gamma
        self.y_loadings_ = np.zeros((q, n_components))  # Delta

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        Xk = scale(X)

        self._y_mean = Y.mean(axis=0)
        self._y_std = Y.std(axis=0)
        Yk = scale(Y)

        for k in range(n_components):
            self.x_scores_[:, k] = np.dot(Xk, self.x_weights_[:, k])
            self.y_scores_[:, k] = np.dot(Yk, self.y_weights_[:, k])
            self.x_loadings_[:, k] = np.dot(
                self.x_scores_[:, k], Xk) / np.dot(self.x_scores_[:, k], self.x_scores_[:, k])
            self.y_loadings_[:, k] = np.dot(
                self.x_scores_[:, k], Yk) / np.dot(self.x_scores_[:, k], self.x_scores_[:, k])

        # Covariance matrix: C = X' @ Y
        C = np.dot(self.x_scores_.T, self.y_scores_)
        # Ordered from higher to lower covariance between
        # X's and Y's rotations.
        ords = np.argsort(-(np.diag(C) ** 2))

        self.x_weights_ = self.x_weights_[:, ords]
        self.y_weights_ = self.y_weights_[:, ords]
        self.x_scores_ = self.x_scores_[:, ords]
        self.y_scores_ = self.y_scores_[:, ords]
        self.x_loadings_ = self.x_loadings_[:, ords]
        self.y_loadings_ = self.y_loadings_[:, ords]

        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv(np.dot(self.x_loadings_.T, self.x_weights_), check_finite=False)
        )
        self.y_rotations_ = np.dot(
            self.y_weights_,
            pinv(np.dot(self.y_loadings_.T, self.y_weights_), check_finite=False)
        )
        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.coef_ = (self.coef_ * self._y_std).T
        self.intercept_ = self._y_mean

        return self

    def transform(self, X, Y=None):
        """Apply the dimension reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Y : array-like of shape (n_samples, n_targets),
        default=None
            Target vectors.

        Returns
        -------
        x_scores, y_scores : array-like or tuple of
        array-like
            Return `x_scores` if `Y` is not given,
            `(x_scores, y_scores)` otherwise.
        """
        Xk = X - self._x_mean
        Xk /= self._x_std
        x_scores = np.dot(Xk, self.x_rotations_)

        if Y is not None:
            Yk = Y - self._y_mean
            Yk /= self._y_std
            y_scores = np.dot(Yk, self.y_rotations_)

            return x_scores, y_scores

        return x_scores

    def inverse_transform(self, X, Y=None):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where `n_samples` is the number of
            samples and `n_components` is the number of
            evolved components.

        Y : array-like of shape (n_samples, n_components)
            New target, where `n_samples` is the number of
            samples and `n_components` is the number of
            evolved components.

        Returns
        -------
        X_reconstructed : ndarray of shape
        (n_samples, n_features)
            Return the reconstructed `X` data.

        Y_reconstructed : ndarray of shape
        (n_samples, n_targets)
            Return the reconstructed `Y` target. Only
            returned when `Y` is given.
        """
        X_reconstructed = np.matmul(X, self.x_loadings_.T)
        # Denormalize.
        X_reconstructed *= self._x_std
        X_reconstructed += self._x_mean

        if Y is not None:
            Y_reconstructed = np.matmul(Y, self.y_loadings_.T)
            # Denormalize.
            Y_reconstructed *= self._y_std
            Y_reconstructed += self._y_mean

            return X_reconstructed, Y_reconstructed

        return X_reconstructed

    def predict(self, X):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        Y_pred : ndarray of shape (n_samples,) or
        (n_samples, n_targets)
            Returns predicted values.
        """
        # Normalize.
        X_norm = X - self._x_mean
        X_norm /= self._x_std

        Y_pred = X @ self.coef_.T + self.intercept_

        return Y_pred

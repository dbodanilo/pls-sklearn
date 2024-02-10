from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def PCAScaled(n_components=None):
    return make_pipeline(StandardScaler(), PCA(n_components=n_components))


def PCR(n_components=None):
    return make_pipeline(StandardScaler(), PCA(n_components=n_components), LinearRegression())

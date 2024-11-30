from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# n_components=2 to match PLSRegression's default.
def ScalerPCA(n_components=2):
    return make_pipeline(StandardScaler(), PCA(n_components=n_components))


def ScalerPCR(n_components=2):
    return make_pipeline(StandardScaler(), PCA(n_components=n_components), LinearRegression())

# NOTE: ScalerPLSR() would be redundant,
# as PLSRegression already scales its input by default.

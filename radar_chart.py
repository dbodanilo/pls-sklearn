"""
Radar chart (aka spider or star chart)
======================================

This example creates a radar chart, also known as a spider
or star chart [1]_.

Although this example allows a frame of either 'circle' or
'polygon', polygon frames don't have proper gridlines (the
lines are circles instead of polygons). It's possible to get
a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
`matplotlib.axis` to the desired number of vertices, but the
orientation of the polygon is not aligned with the radial
axes.

.. [1] https://en.wikipedia.org/wiki/Radar_chart
"""

import matplotlib.pyplot as plt
import numpy as np

from math import ceil, floor

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler

from decomposition import ScalerPCR
from model import load_leme, train_test_seed_split
from util import fit_predict_try_transform, try_getattr


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and
    registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps
            # correspond to gridlines, in which case we
            # force interpolation (to defeat
            # PolarTransform's autoconversion to circular
            # arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)

            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the
            # top.
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5)
            # and of radius 0.5 in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=0.5, edgecolor="k")
            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

        def _gen_axes_spines(self):
            if frame == 'circle':
                spines = super()._gen_axes_spines()
                return {'polar': spines['polar']}
            elif frame == 'polygon':
                # spine_type must be 'left'|'right'|'top'|'bottom'|'circle'.
                spine = Spine(axes=self, spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of
                # radius 1 centered at (0, 0) but we want a
                # polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5)
                                    + self.transAxes)

                return {'polar': spine}

            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

    register_projection(RadarAxes)

    return theta


def example_data():
    n = 5

    X, Y = load_leme()

    spoke_labels = Y.columns.drop(['N.', 'Semente'])

    X_train, X_test, Y_train, Y_test = train_test_seed_split(X, Y)
    y_test_mean = Y_test.mean(axis=0)

    pcr = ScalerPCR(n_components=n).fit(X_train, Y_train)
    # rpcr = ScalerPCR(n_components=n).fit(Y_train, X_train)

    Y_pred_pcr = pcr.predict(X_test)
    y_pred_pcr_mean = Y_pred_pcr.mean(axis=0)

    plsr = PLSRegression(n_components=n).fit(X_train, Y_train)
    # rplsr = PLSRegression(n_components=n).fit(Y_train, X_train)

    Y_pred_plsr = plsr.predict(X_test)
    y_pred_plsr_mean = Y_pred_plsr.mean(axis=0)

    y_scaler = MinMaxScaler((0, 100)).fit(Y_test)
    y_test_mean_scaled = y_scaler.transform(y_test_mean.reshape(1, -1))
    y_pred_pcr_mean_scaled = y_scaler.transform(y_pred_pcr_mean.reshape(1, -1))
    y_pred_plsr_mean_scaled = y_scaler.transform(
        y_pred_plsr_mean.reshape(1, -1))

    # pca_components = try_getattr(rpcr, "components_")

    # `x_` prefix merely denotes the first training matrix,
    # in this case: `Y_train`.
    # pls_components = try_getattr(rplsr, "x_rotations_").T

    data = [
        spoke_labels,
        ("PCA", np.concatenate([y_test_mean_scaled, y_pred_pcr_mean_scaled])),
        ("PLS", np.concatenate([y_test_mean_scaled, y_pred_plsr_mean_scaled]))
    ]

    return data


if __name__ == '__main__':
    N = 5
    theta = radar_factory(N, frame='polygon')

    data = example_data()
    spoke_labels = data.pop(0)

    fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=1,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['r', 'g', 'b', 'y', 'm']
    # Plot the two cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        grid_min = floor(10 * min(c.min() for (_, c) in data)) / 10
        grid_max = ceil(10 * max(c.max() for (_, c) in data)) / 10
        grid_abs = max(abs(grid_max), abs(grid_min))
        n = ceil(5 * (grid_max - grid_min))

        ax.set_rgrids(np.linspace(0, 100, 21))
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')

        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color,
                    alpha=0.25, label='_nolegend_')

        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    # labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    labels = ('Actual', 'Predicted', 'Actual - std',
              'Actual + std', 'Factor 5')
    legend = axs[0].legend(labels, loc=(0.9, 0.95),
                           labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, f'Mean of test samples for {len(data)} algorithms',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.show()

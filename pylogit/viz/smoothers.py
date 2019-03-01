# -*- coding: utf-8 -*-
"""
This file contains functions and classes for producing discrete and/or
continuous smooths of a binary or continuous variable against another
continuous variable.
"""
import numpy as np

# Use ExtRaTrees for continuously smoothed marginal model plots
from sklearn.ensemble import (ExtraTreesClassifier,
                              ExtraTreesRegressor)

from .plot_utils import _determine_bin_obs
from .plot_utils import _populate_bin_means_for_plots


def _get_extra_smooth_xy(x, y,
                         n_estimators=50,
                         min_samples_leaf=10,
                         random_state=None):
    """
    Creates an ensemble of extremely randomized trees that predict y given x,
    and returns the smoothed (i.e. predicted) y and original x.

    Parameters
    ----------
    x, y : 1D ndarray of real values in [0, 1].
        X should be an array of continuous values. y should be an array of
        either continuous or binary (0 or 1) data.
    n_estimators : positive int, optional.
        Determines the number of trees in the ensemble. This parameter controls
        how smooth one's resulting estimate is. The more estimators the
        smoother one's estimated relationship and the lower the variance in
        that estimated relationship. Default == 50.
    min_samples_leaf : positive int, optional.
        Determines the minimum number of observations allowed in a leaf node in
        any tree in the ensemble. This parameter is conceptually equivalent to
        the bandwidth parameter in a kernel density estimator.

    Returns
    -------
    x, smoothed_y : 1D ndarray of real values.
        x is the same as above. `smoothed_y` is the predicted y values based on
        the ensemble of extremely randomized trees.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 1:
        msg = 'x MUST be a 1D ndarray'
        raise ValueError(msg)
    if not isinstance(y, np.ndarray) or len(y.shape) != 1:
        msg = 'y MUST be a 1D ndarray'
        raise ValueError(msg)
    # The if condition checks if we are dealing with continuous y vs discrete y
    if ((y < 1.0) & (y > 0)).any():
        smoother = ExtraTreesRegressor(n_estimators=n_estimators,
                                       min_samples_leaf=min_samples_leaf,
                                       max_features=1,
                                       random_state=random_state)
        smoother.fit(x[:, None], y)
        smoothed_y = smoother.predict(x[:, None])
    else:
        smoother = ExtraTreesClassifier(n_estimators=n_estimators,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=1,
                                        random_state=random_state)
        smoother.fit(x[:, None], y)
        # Note we use [:, 1] to get the predicted probabilities of y = 1
        smoothed_y = smoother.predict_proba(x[:, None])[:, 1]
    return x, smoothed_y


class Smoother(object):
    """
    Base class for the discrete and continuous smoothers. Instances of
    subclasses of `Smoother` will take in raw X and Y values, and they will
    output new x and y values that can be plotted to show the smoothed
    conditional expectation function, E[y | x].
    """
    def __init__(self):
        return None

    def __call__(self, X, Y):
        """
        Takes in raw X and Y and produces smoothed_x and smoothed_y. The
        outputs can then be plotted to visualize the smoothed,
        conditional expectation function, E[y | x], according to the specified
        smoother.

        Parameters
        ----------
        X, Y : 1D ndarrays
            Should contain the raw data for which we want to visualize a smooth
            of the conditional expectation function, E[y | x].

        Returns
        -------
        smoothed_x, smoothed_y : 1D ndarrays
            Contains the smoothed values to be plotted, respectively, on the
            x-axis and y-axis to show the smoothed E[y | x].
        """
        return self.smooth(X, Y)

    def smooth(self, X, Y):
        """
        Takes in raw X and Y and produces smoothed_x and smoothed_y. The
        outputs can then be plotted to visualize the smoothed,
        conditional expectation function, E[y | x], according to the specified
        smoother.

        Parameters
        ----------
        X, Y : 1D ndarrays
            Should contain the raw data for which we want to visualize a smooth
            of the conditional expectation function, E[y | x].

        Returns
        -------
        smoothed_x, smoothed_y : 1D ndarrays
            Contains the smoothed values to be plotted, respectively, on the
            x-axis and y-axis to show the smoothed E[y | x].
        """
        raise NotImplementedError


class DiscreteSmoother(Smoother):
    def __init__(self,
                 num_obs,
                 partitions=10):
        super(DiscreteSmoother, self).__init__()
        self.num_obs = num_obs
        self.partitions = partitions

        # Initialize attributes for discrete smoothing (i.e. binning)
        self.mean_x_per_group = np.zeros(self.partitions)
        self.mean_y_per_group = np.zeros(self.partitions)

        # Determine the number of observations in each partition
        self.obs_per_partition = _determine_bin_obs(num_obs, self.partitions)
        return None

    def smooth(self, X, Y):
        """
        Takes in raw X and Y and produces smoothed_x and smoothed_y. The
        outputs can then be plotted to visualize the smoothed,
        conditional expectation function, E[y | x], according to the specified
        smoother.

        Parameters
        ----------
        X, Y : 1D ndarrays
            Should contain the raw data for which we want to visualize a smooth
            of the conditional expectation function, E[y | x].

        Returns
        -------
        smoothed_x, smoothed_y : 1D ndarrays
            Contains the smoothed values to be plotted, respectively, on the
            x-axis and y-axis to show the smoothed E[y | x].
        """
        return _populate_bin_means_for_plots(X,
                                             Y,
                                             self.obs_per_partition,
                                             self.mean_x_per_group,
                                             self.mean_y_per_group)[:2]


class ContinuousSmoother(Smoother):
    def __init__(self,
                 n_estimators=50,
                 min_samples_leaf=10,
                 random_state=None):
        super(ContinuousSmoother, self).__init__()
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        return None

    def smooth(self, X, Y):
        """
        Takes in raw X and Y and produces smoothed_x and smoothed_y. The
        outputs can then be plotted to visualize the smoothed,
        conditional expectation function, E[y | x], according to the specified
        smoother.

        Parameters
        ----------
        X, Y : 1D ndarrays
            Should contain the raw data for which we want to visualize a smooth
            of the conditional expectation function, E[y | x].

        Returns
        -------
        smoothed_x, smoothed_y : 1D ndarrays
            Contains the smoothed values to be plotted, respectively, on the
            x-axis and y-axis to show the smoothed E[y | x].
        """
        return _get_extra_smooth_xy(X, Y,
                                    n_estimators=self.n_estimators,
                                    min_samples_leaf=self.min_samples_leaf,
                                    random_state=self.random_state)


class SmoothPlotter(object):
    def __init__(self, smoother, ax):
        self.ax = ax
        self.smoother = smoother
        return None

    def plot(self, X, Y, label=None, color='#a6cee3', alpha=0.5):
        """
        Plots a smooth estimate of the conditional expectation function E[y|x].

        Parameters
        ----------
        X, Y : 1D ndarrays
            Should contain the raw data for which we want to visualize a smooth
            of the conditional expectation function, E[y | x].
        label : str or None, optional.
            Denotes the label for the plotted curve. Default is None.
        color : valid matplotlib color, optional.
            The color that is used for the plotted curve. Default is '#a6cee3'.
        alpha : positive float in [0.0, 1.0], or `None`, optional.
            Determines the opacity of the elements drawn on the plot.
            0.0 == transparent and 1.0 == opaque. Default == 0.5.

        Returns
        -------
        None.
        """
        # Get the smoothed x and y values to be plotted.
        plot_x, plot_y = self.smoother(X, Y)
        # Make the desired plot
        self.ax.plot(plot_x, plot_y, c=color, alpha=alpha, label=label)
        return None

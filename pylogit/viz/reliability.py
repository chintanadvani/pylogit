# -*- coding: utf-8 -*-
"""
Functions for plotting reliability diagrams: smooths of simulated vs observed
outcomes on the y-axis against predicted probabilities on the x-axis.
"""
from __future__ import absolute_import

import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

from .utils import progress
from .plot_utils import _label_despine_save_and_show_plot

# Set the plotting style
sbn.set_style('darkgrid')


def _determine_bin_obs(total, partitions):
    """
    Determines the number of observations that should be in a given partition.

    Parameters
    ----------
    total : positive int.
        Denotes the total number of observations that are to be partitioned.
    partitions : positive int.
        Denotes the number of partitions that are to be created. Should be
        less than or equal to `total`.

    Returns
    -------
    obs_per_partition : 1D ndarray of positive its.
        Denotes the number of observations to be placed in each partition.
        Will have one element per partition.
    """
    partitions_float = float(partitions)
    naive = int(total / partitions_float) * np.ones(partitions)
    correction = np.ones(partitions)
    correction[total % partitions:] = 0
    return (naive + correction).astype(int)


def _populate_bin_means_for_plots(x_vals,
                                  y_vals,
                                  obs_per_bin,
                                  mean_x,
                                  mean_y,
                                  auxillary_y=None,
                                  auxillary_mean=None):
    """
    Populate the mean per bin of predicted probabilities, observed outcomes,
    and simulated outcomes.

    Parameters
    ----------
    x_vals : 1D ndarray of floats.
        Elements should be the sorted values to be placed on the x-axis.
    y_vals : 1D ndarray of floats.
        Elements should be the values to be averaged and placed on the y-axis.
        Should have been sorted in the same order as `x_vals`.
    obs_per_bin : 1D ndarray of positive ints.
        There should be one element per bin. Each element should denote the
        number of observations to be used in each partition.
    mean_x, mean_y : 1D ndarray.
        `mean_x.size` and `mean_y.size` should equal `obs_per_bin.size`.
    auxillary_y : 1D ndarray or None, optional.
        Same as `y_vals` except these elements denote additional values to be
        plotted on the y-axis.
    auxillary_mean : 1D ndarray or None, optional.
        Same as `mean_probs` and `mean_obs`.

    Returns
    -------
    mean_x : 1D ndarray.
        Will have 1 element per partition. Each value will denote the mean of
        the `x_vals` for all observations in the partition.
    mean_y : 1D ndarray.
        Will have 1 element per partition. Each value will denote the mean of
        the `y_vals` for all observations in the partition.
    auxillary_mean : 1D ndarray or None.
        Will have 1 element per partition. Each value will denote the mean of
        the `auxillary_y` for all observations in the partition. If
        `auxillary_mean` was passed as None, it will be returned as None.
    """
    # Initialize a row counter
    row_counter = 0

    # Iterate over each of the partitions
    for i in range(obs_per_bin.size):
        # Get the upper and lower ranges of the slice
        lower_row = row_counter
        upper_row = row_counter + obs_per_bin[i]

        # Get the particular observations we care about
        rel_x = x_vals[lower_row:upper_row]
        rel_y = y_vals[lower_row:upper_row]

        # Store the mean probs and mean y
        mean_x[i] = rel_x.mean()
        mean_y[i] = rel_y.mean()

        # Store the mean simulated y per group
        if auxillary_y is not None:
            rel_auxillary_y = auxillary_y[lower_row:upper_row]
            auxillary_mean[i] = rel_auxillary_y.mean()

        # Update the row counter
        row_counter += obs_per_bin[i]

    return mean_x, mean_y, auxillary_mean


def _check_reliability_args(probs, choices, partitions, sim_y):
    """
    Ensures `probs` is a 1D or 2D ndarray, that `choices` is a 1D ndarray, that
    `partitions` is an int, and that `sim_y` is a ndarray of the same shape as
    `probs` or None.
    """
    if not isinstance(probs, np.ndarray):
        msg = '`probs` MUST be an ndarray.'
        raise ValueError(msg)
    if probs.ndim not in [1, 2]:
        msg = 'probs` MUST be a 1D or 2D ndarray.'
        raise ValueError(msg)
    if not isinstance(choices, np.ndarray):
        msg = '`choices` MUST be an ndarray.'
        raise ValueError(msg)
    if choices.ndim != 1:
        msg = '`choices` MUST be a 1D ndarray.'
        raise ValueError(msg)
    if not isinstance(partitions, int):
        msg = '`partitions` MUST be an int.'
        raise ValueError(msg)
    if not isinstance(sim_y, np.ndarray) and sim_y is not None:
        msg = '`sim_y` MUST be an ndarray or None.'
        raise ValueError(msg)
    sim_to_prob_conditions = probs.ndim != 1 and sim_y.shape != probs.shape
    if sim_y is not None and sim_to_prob_conditions:
        msg = ('`sim_y` MUST have the same shape as `probs` if '
               '`probs.shape[1] != 1`.')
        raise ValueError(msg)
    return None


def _plot_single_binned_reliability_curve(probs,
                                          choices,
                                          col,
                                          ax,
                                          line_label,
                                          line_color,
                                          alpha,
                                          obs_per_partition,
                                          mean_probs_per_group,
                                          mean_y_per_group):
    """
    Plots a single, binned reliability curve on the provided matplotlib Axes.

    Parameters
    ----------
    probs : 1D or 2D ndarray.
        Each element should be in [0, 1]. There should be 1 column for each
        set of predicted probabilities. These will be plotted on the x-axis.
    choices : 1D ndarray.
        Each element should be either a zero or a one. Elements should denote
        whether the alternative corresponding to the given row was chosen or
        not. A 'one' corresponds to a an outcome of 'success'.
    col : int
        The current column in `probs` to use when computing the x-axis values.
    ax : matplotlib Axes instance
        The Axes that the reliability curve should be plotted on.
    line_label : str or None.
        Denotes the label to be used for the lines relating the predicted
        probabilities and the binned, empirical probabilities.
    line_color : valid matplotlib color.
        Determines the color that is used to plot the predicted probabilities
        versus the observed choices.
    alpha : positive float in [0.0, 1.0], or `None`, optional.
        Determines the opacity of the elements drawn on the plot.
        0.0 == transparent and 1.0 == opaque. Default == 1.0.
    obs_per_partition : 1D ndarray of positive its.
        Denotes the number of observations to be placed in each partition.
        Will have one element per partition.
    mean_probs_per_group, mean_y_per_group : 1D ndarray of positive scalars.
        Denotes the mean of the probabilities and observed choices per 'bin'.

    Returns
    -------
    None. `ax` is modified in place: the line is plotted and the label added.
    """
    # Get the current line label and probabilities
    current_line_label = line_label if col == 0 else None
    current_probs = probs[:, col]

    # Sort the array of probs and choices
    sort_order = np.argsort(current_probs)
    current_probs = current_probs[sort_order]
    current_choices = choices[sort_order]

    # Populate the bin means of predicted probabilities,
    # observed choices, and simulated choices
    population_results =\
        _populate_bin_means_for_plots(
            current_probs, current_choices,
            obs_per_partition, mean_probs_per_group, mean_y_per_group)
    mean_probs_per_group = population_results[0]
    mean_y_per_group = population_results[1]

    # Plot the mean predicted probs per group versus
    # the mean observations per group
    ax.plot(mean_probs_per_group,
            mean_y_per_group,
            c=line_color,
            alpha=alpha,
            label=current_line_label)
    return


def add_ref_line(ax, ref_label="Perfect Calibration"):
    """
    Plots a diagonal line to show perfectly calibrated probabilities.

    Parameters
    ----------
    ax : matplotlib Axes instance
        The Axes that the reference line should be plotted on.
    ref_label : str, optional.
        The label to be applied to the reference line that is drawn.

    Returns
    -------
    None. `ax` is modified in place: the line is plotted and the label added.
    """
    # Determine the maximum value of the x-axis or y-axis
    max_ref_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    min_ref_val = max(ax.get_xlim()[0], ax.get_ylim()[0])
    # Determine the values to use to plot the reference line
    ref_vals = np.linspace(min_ref_val, max_ref_val, num=100)
    # Plot the reference line as a black dashed line
    ax.plot(ref_vals, ref_vals, 'k--', label=ref_label)
    return None


def plot_binned_reliability(probs,
                            choices,
                            partitions=10,
                            line_color='#1f78b4',
                            line_label='Observed vs Predicted',
                            alpha=None,
                            sim_y=None,
                            sim_line_color='#a6cee3',
                            sim_label=None,
                            sim_alpha=0.5,
                            x_label='Mean Predicted Probability',
                            y_label='Binned\nEmpirical\nProbability',
                            title=None,
                            fontsize=12,
                            ref_line=False,
                            figsize=(5, 3),
                            fig_and_ax=None,
                            legend=True,
                            progress_bar=True,
                            show=True,
                            output_file=None,
                            dpi=500):
    """
    Creates a binned reliability plot based on the given probability
    predictions and the given observed outcomes.

    Parameters
    ----------
    probs : 1D or 2D ndarray.
        Each element should be in [0, 1]. There should be 1 column for each
        set of predicted probabilities. These will be plotted on the x-axis.
    choices : 1D ndarray.
        Each element should be either a zero or a one. Elements should denote
        whether the alternative corresponding to the given row was chosen or
        not. A 'one' corresponds to a an outcome of 'success'.
    partitions : positive int.
        Denotes the number of partitions to split one's data into for binning.
    line_color : valid matplotlib color, optional.
        Determines the color that is used to plot the predicted probabilities
        versus the observed choices. Default is `'#1f78b4'`.
    line_label : str or None, optional.
        Denotes the label to be used for the lines relating the predicted
        probabilities and the binned, empirical probabilities. Default is
        'Observed vs Predicted'.
    alpha : positive float in [0.0, 1.0], or `None`, optional.
        Determines the opacity of the elements drawn on the plot.
        0.0 == transparent and 1.0 == opaque. Default == 1.0.
    sim_y : 2D ndarray or None, optional.
        Denotes the choices that were simulated based on `probs`. If passed,
        `sim_y.shape` MUST equal `probs.shape` in order to ensure that lines
        are plotted for the predicted probabilities versus simulated choices.
        This kwarg is useful because it shows one the reference distribution of
        predicted probabilities versus choices that actually come from one's
        postulated model.
    sim_line_color : valid matplotlib color, optional.
        Determines the color that is used to plot the predicted probabilities
        versus the simulated choices. Default is `'#a6cee3'`.
    sim_line_label : str, or None, optional.
        Denotes the label to be used for the lines relating the predicted
        probabilities and the binned, empirical probabilities based on the
        simulated choices. Default is None.
    sim_alpha : positive float in [0.0, 1.0], or `None`, optional.
        Determines the opacity of the simulated reliability curves.
        0.0 == transparent and 1.0 == opaque. Default == 0.5.
    x_label, y_label : str, optional.
        Denotes the label for the x-axis and y-axis, respectively. Defaults are
        'Mean Predicted Probability' and 'Binned\nEmpirical\nProbability' for
        the x-axis and y-axis, respectively.
    title : str, or None, optional.
        Denotes the title to be displayed for the plot. Default is None.
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
    ref_line : bool, optional.
        Determines whether a diagonal line, y = x, will be plotted to show the
        expected relationship. Default is True.
    figsize : 2-tuple of positive ints.
        Determines the size of the created figure. Default == (5, 3).
    fig_and_ax : list of matplotlib figure and axis, or `None`, optional.
        Determines whether a new figure will be created for the plot or whether
        the plot will be drawn on existing axes. If None, a new figure will be
        created. Default is `None`.
    legend : bool, optional.
        Determines whether a legend is printed for the plot. Default == True.
    progress_bar : bool, optional.
        Determines whether a progress bar is displayed while making the plot.
        Default == True.
    show : bool, optional.
        Determines whether the figure is shown after plotting is complete.
        Default == True.
    output_file : str, or None, optional.
        Denotes the relative or absolute filepath (including the file format)
        that is to be used to save the plot. If None, the plot will not be
        saved to file. Default is None.
    dpi : positive int, optional.
        Denotes the number of 'dots per inch' for the saved figure. Will only
        be used if `output_file is not None`. Default == 500.

    Returns
    -------
    None.
    """
    # Perform some basic argument checking
    _check_reliability_args(probs, choices, partitions, sim_y)

    # Make probs 2D if necessary
    probs = probs[:, None] if probs.ndim == 1 else probs

    # Create the figure and axes if need be
    if fig_and_ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
        fig_and_ax = [fig, ax]
    else:
        fig, ax = fig_and_ax

    # Create the progressbar iterator if desired
    if progress_bar and sim_y is not None:
        description = "Plotting" if sim_y is None else "Plotting Simulations"
        sim_iterator = progress(range(sim_y.shape[1]), desc=description)
    else:
        sim_iterator = range(probs.shape[1])

    # Determine the number of observations in each partition
    obs_per_partition = _determine_bin_obs(probs.shape[0], partitions)

    # Initialize an array for each group's mean probabilities and observations
    mean_probs_per_group = np.zeros(partitions)
    mean_y_per_group = np.zeros(partitions)

    # Create helper functions
    def get_current_probs(col):
        """
        Fetches the current probabilities when plotting the reliability curves.
        """
        current = probs[:, 0] if probs.shape[1] == 1 else probs[:, col]
        return current

    # Plot the simulated reliability curves, if desired
    if sim_y is not None:
        for i in sim_iterator:
            current_label = sim_label if i == 0 else None
            plot_binned_reliability(get_current_probs(i),
                                    sim_y[:, i],
                                    partitions=partitions,
                                    line_color=sim_line_color,
                                    line_label=current_label,
                                    alpha=sim_alpha,
                                    sim_y=None,
                                    sim_line_color=None,
                                    sim_label=None,
                                    title=None,
                                    fontsize=fontsize,
                                    ref_line=False,
                                    figsize=figsize,
                                    fig_and_ax=fig_and_ax,
                                    legend=False,
                                    progress_bar=False,
                                    show=False,
                                    output_file=None,
                                    dpi=dpi)

    # Create the progressbar iterator if desired
    if progress_bar:
        prob_iterator = progress(range(probs.shape[1]), desc="Plotting")
    else:
        prob_iterator = range(probs.shape[1])

    # Make the 'true' reliability plots
    for col in prob_iterator:
        _plot_single_binned_reliability_curve(
            probs, choices, col, ax, line_label, line_color,
            alpha, obs_per_partition, mean_probs_per_group, mean_y_per_group)

    # Create the reference line if desired
    if ref_line:
        add_ref_line(ax)

    # Make the legend, if desired
    if legend:
        ax.legend(loc='best', fontsize=fontsize)

    # Take care of boilerplate plotting necessities
    _label_despine_save_and_show_plot(
        x_label=x_label, y_label=y_label, fig_and_ax=fig_and_ax,
        fontsize=fontsize, y_rot=0, y_pad=40, title=title,
        output_file=output_file, show=show, dpi=dpi)
    return None

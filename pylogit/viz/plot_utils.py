"""
Helper functions for plotting.
"""
import numpy as np

# Use statsmodels for empirical cdf function
import statsmodels.tools as sm_tools
import statsmodels.distributions as sm_dist

# Alias the empirical cdf function
ECDF = sm_tools.tools.ECDF if hasattr(sm_tools.tools, 'ECDF') else sm_dist.ECDF

def _label_despine_save_and_show_plot(x_label,
                                      y_label,
                                      fig_and_ax,
                                      fontsize=12,
                                      y_rot=0,
                                      y_pad=40,
                                      title=None,
                                      output_file=None,
                                      show=True,
                                      dpi=500):
    """
    Adds the x-label, y-label, and title to the matplotlib Axes object. Also
    despines the figure, saves it (if desired), and shows it (if desired).

    Parameters
    ----------
    x_label, y_label : string.
        Determines the labels for the x-axis and y-axis respectively.
    fig_and_ax : list of matplotlib figure and axis.
        The matplotlib figure and axis that are being altered.
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
    y_rot : int in [0, 360], optional.
        Denotes the angle by which to rotate the text of the y-axis label.
        Default == 0.
    y_pad : int, optional.
        Denotes the amount by which the text of the y-axis label will be offset
        from the y-axis itself. Default == 40.
    title : string or None, optional.
        Denotes the title to be displayed for the plot. Default is None.
    output_file : str, or None, optional.
        Denotes the relative or absolute filepath (including the file format)
        that is to be used to save the plot. If None, the plot will not be
        saved to file. Default is None.
    show : bool, optional.
        Determines whether the figure is shown after plotting is complete.
        Default == True.
    dpi : positive int, optional.
        Denotes the number of 'dots per inch' for the saved figure. Will only
        be used if `output_file is not None`. Default == 500.
    """
    # Ensure seaborn is imported
    if 'sbn' not in globals():
        import seaborn as sbn

    # Get the figure and axis as separate objects
    fig, axis = fig_and_ax

    # Despine the plot
    sbn.despine()
    # Make plot labels
    axis.set_xlabel(x_label, fontsize=fontsize)
    axis.set_ylabel(y_label, fontsize=fontsize, rotation=y_rot, labelpad=y_pad)
    # Create the title
    if title is not None and title != '':
        if not isinstance(title, basestring):
            msg = "`title` MUST be a string."
            raise TypeError(msg)
        axis.set_title(title, fontsize=fontsize)

    # Save the plot if desired
    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')

    if show:
        fig.show()
    return None


def _choice_evaluator(choice_array, choice_condition):
    """
    Determines which rows in `choice_array` meet the given `choice_condition`,
    where `choice_condition` is in the set `{0.0, 1.0}`.

    Parameters
    ----------
    choice_array : 1D ndarray of ints that are either 0 or 1.
    choice_condition : int in `{0, 1}`.

    Returns
    -------
    bool_mask : 1D ndarray of bools.
        Equal to `choice_array == choice_condition`
    """
    if choice_condition in [0.0, 1.0]:
        return choice_array == choice_condition
    else:
        msg = 'choice_condition MUST be either a 0 or a 1'
        raise ValueError(msg)


def _thin_rows(sim_y, thin_pct):
    """
    Randomly select `thin_pct` percentage of rows to be used in plotting.

    Parameters
    ----------
    sim_y : 2D ndarray of zeros and ones.
        Each row should represent an alternative for a given choice situation.
        Each column should represent a given simulated set of choices.
    thin_pct : float in (0.0, 1.0) or None, optional.
        Determines the percentage of the data (rows) to be used for plotting.
        If None, the full dataset will be used. Default is None.

    Returns
    -------
    selected_rows : 1D ndarray of bools.
        Denotes the randomly selected rows to be used in plotting.
    """
    # Determine the number of rows to select
    num_selected_rows = int(thin_pct * sim_y.shape[0])
    # Randomly choose rows to retain.
    selected_rows =\
        np.random.choice(sim_y.shape[0],
                         size=num_selected_rows,
                         replace=False)
    return selected_rows


def _plot_single_cdf_on_axis(x_vals,
                             axis,
                             color='#a6bddb',
                             linestyle='-',
                             label=None,
                             alpha=0.1):
    """
    Plots a CDF of `x_vals` on `axis` with the desired color, linestyle, label,
    and transparency (alpha) level.
    """
    # Create a function that will take in an array of values and
    # return an array of the same length which contains the CDF
    # value at each corresponding value in the passed array.
    cdf_func = ECDF(x_vals)
    # Create a sorted list of all of the unique values that were
    # sampled for this variable
    sorted_samples = np.sort(np.unique(x_vals))
    # Get the CDF values for each of the sorted values
    cdf_values = cdf_func(sorted_samples)
    # Plot the sorted, unique values versus their CDF values
    axis.plot(sorted_samples,
              cdf_values,
              c=color,
              ls=linestyle,
              alpha=alpha,
              label=label,
              drawstyle='steps-post')
    return None


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
        Same as `mean_x` and `mean_y`.

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


def _plot_single_binned_x_vs_binned_y(x_vals,
                                      y_vals,
                                      ax,
                                      label,
                                      color,
                                      alpha,
                                      obs_per_partition,
                                      mean_x_per_group,
                                      mean_y_per_group):
    """
    Plots a single, binned reliability curve on the provided matplotlib Axes.

    Parameters
    ----------
    x_vals : 1D ndarray.
        A 'continuous' variable that is to be plotted on the x-axis.
    y_vals : 1D ndarray.
        Each element should be either a zero or a one. Elements should denote
        whether the alternative corresponding to the given row was chosen or
        not. A 'one' corresponds to a an outcome of 'success'.
    ax : matplotlib Axes instance
        The Axes that the reliability curve should be plotted on.
    label : str or None.
        Denotes the label to be used for the lines relating the predicted
        probabilities and the binned, empirical probabilities.
    color : valid matplotlib color.
        Determines the color that is used to plot the predicted probabilities
        versus the observed choices.
    alpha : positive float in [0.0, 1.0], or `None`.
        Determines the opacity of the elements drawn on the plot.
        0.0 == transparent and 1.0 == opaque.
    obs_per_partition : 1D ndarray of positive its.
        Denotes the number of observations to be placed in each partition.
        Will have one element per partition.
    mean_probs_per_group, mean_y_per_group : 1D ndarray of positive scalars.
        Denotes the mean of the probabilities and observed choices per 'bin'.

    Returns
    -------
    None. `ax` is modified in place: the line is plotted and the label added.
    """
    # Populate the bin means of predicted probabilities,
    # observed choices, and simulated choices
    population_results =\
        _populate_bin_means_for_plots(x_vals, y_vals, obs_per_partition,
                                      mean_x_per_group, mean_y_per_group)
    mean_x_per_group = population_results[0]
    mean_y_per_group = population_results[1]

    # Plot the mean predicted probs per group versus
    # the mean observations per group
    ax.plot(mean_x_per_group, mean_y_per_group,
            c=color, alpha=alpha, label=label)
    return None

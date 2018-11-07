"""
Helper functions for the rest of the visualization module. Externally, this
file gives users a way to:
- create prorgress bars without worrying about the python environment
- determine if a given vector contains data that is in some way categorical
- simulate from their model's predicted probabilities.
"""
import sys

import numpy as np
from scipy.stats import itemfreq

from tqdm import tqdm, tqdm_notebook


def _is_kernel():
    """
    Determines whether or not one's code is executed inside of an ipython
    notebook environment.

    Returns
    -------
    in_kernel : bool
        True if one's code is in an ipython environment. False otherwise.
    """
    return bool(any([x in sys.modules for x in ['ipykernel', 'IPython']]))


def progress(*args, **kwargs):
    """
    Creates a tqdm progressbar iterable based on whether one is in ipython.
    In ipython it will return a `tqdm_notebook` iterable. Else, it returns a
    `tqdm` iterable.

    Parameters
    ----------
    args, kwargs: passed directly to `tqdm` and `tqdm_notebook`.
    """
    if _is_kernel():
        return tqdm_notebook(*args, **kwargs)
    return tqdm(*args, **kwargs)


def _prep_categorical_return(truth, description, verbose):
    """
    Return `truth` and `description` if `verbose is True` else return
    `description` by itself.
    """
    if verbose:
        return truth, description
    return truth


def is_categorical(vector,
                   solo_threshold=0.1,
                   group_threshold=0.5,
                   group_num=10,
                   verbose=False):
    """
    Determines if a given vector of variables is categorical (or mixed
    categorical and continuous) or not.

    Parameters
    ----------
    vector : 1D ndarray.
        Contains the data to be checked for categorical status.
    solo_threshold : float in (0.0, 1.0), optional.
        If a single unique value in `vector` makes up more than or equal to
        this fraction of the values, then the vector is considered to be
        categorical. Default == 0.1.
    group_threshold : float in (0.0, 1.0), optional.
        If a group of `group_num` unique values in `vector` makes up more than
        this fraction of the values, then the vector is considered to be
        categorical. Default == 0.5.
    group_num : int, optional.
        Denotes the size of the group that is used when judging if the vector
        is categorical or not. Default == 10.
    verbose : bool, optional.
        Determines whether the function will return a description of the 'type'
        of categorical variable this vector is deemed to be. Default is False.

    Returns
    -------
    bool, or (bool, str) if `verbose == True`.

    Examples
    --------
    >>> import numpy as np
    >>> is_categorical(np.arange(30))
    False

    >>> is_categorical(np.arange(30), group_num=15, group_threshold=0.5)
    True

    15 values make up 50% of the data so the second example evaluates to True

    >>> x = np.tile(np.array([1, 2, 3]), 10)
    >>> x
    array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2,
           3, 1, 2, 3, 1, 2, 3])

    >>> is_categorical(x, group_num=2, group_threshold=0.75)
    True

    Even though 2 values make up only 2/3 of the data, a single value (e.g. 1)
    makes up 10% of the data, thus reaching the `solo_threshold`

    >>> is_categorical(x, solo_threshold=0.15,
    >>>                group_num=2, group_threshold=0.75)
    False

    Since the `solo_threshold` is now 15%, `x` is no longer deemed categorical.
    """
    # Figure out how many observations are in `vector`
    num_observations = float(vector.shape[0])
    # Get the count of each unique value in `vector`
    item_frequencies = itemfreq(vector)
    # Sort the item frequencies by the second column
    item_frequencies =\
        item_frequencies[np.argsort(item_frequencies[:, 1])[::-1], :]
    # Get the percentage of `vector` made up by each unique value
    individual_percents = item_frequencies[:, 1] / num_observations
    # Get the cumulative density function of `vector`.
    cumulative_percents = np.cumsum(item_frequencies[:, 1]) / num_observations
    # Check for 'categorical' nature of `vector`
    if item_frequencies.shape[0] <= group_num:
        truth = True
        description = 'categorical'
    elif cumulative_percents[group_num] >= group_threshold:
        truth = True
        description = 'group'
    elif (individual_percents > solo_threshold).any():
        truth = True
        description = 'solo'
    else:
        truth = False
        description = None

    return _prep_categorical_return(truth, description, verbose)


def _simulate_wide_binary_choices(predictions, rseed=None):
    """
    Take vectorized random draws over many bernoulli random variables with
    different probabilities of success. This function is faster than using a
    for-loop and repeated calls to `np.random.choice`.
    """
    # Initialize the simulated choices
    choice_vec = np.zeros(predictions.shape, dtype=int)

    # Set the random seed if desired
    if rseed is not None:
        np.random.seed(rseed)

    # Generate uniform random variates
    uniform_draws =\
        np.random.uniform(size=predictions.shape)

    # Determine which predictions led to 'successful' observations
    choice_vec[np.where(uniform_draws <= predictions)] = 1
    return choice_vec


def _simulate_choices_for_1obs(obs_id,
                               rows_per_obs,
                               predicted_probs):
    """
    Generates the chosen rows for each simulated choice situation for the given
    decision maker.

    Parameters
    ----------
    obs_id : positive int.
        The identification number of a given decision maker.
    rows_per_obs : dict.
        Keys should be integers, including `obs_id`. Values should be a list of
        ints, where each int is a row of `predicted_probs` that is associated
        with the decision maker identified by `obs_id`.
    predicted_probs : 2D ndarray of floats in (0.0, 1.0).
        Each row should correspond to a particular alternative for a particular
        observation. Each column should correspond to a sampled parameter
        vector. Finally, each element should denote the probability of that
        alternative being chosen by that decision maker, given their
        explanatory variables and the sampled model parameters.

    Returns
    -------
    chosen_rows : 1D ndarray of ints.
        Should have shape `(predicted_probs.shape[1],)`. There will be one
        value for each simulated choice situation, i.e. each column of
        `predicted_probs`. Each value will represent the row that correspondes
        to the chosen alternative for the corresponding choice situation.
    """
    # Get the rows belonging to this observation
    obs_rows = rows_per_obs[obs_id]

    # Get the current probabilities
    current_long_probs = predicted_probs[obs_rows, :]

    # Get the 'cdf' of each alternative
    current_cdf = np.cumsum(current_long_probs, axis=0)

    # Draw random uniform values for each probability vector
    uniform_draws = np.random.uniform(size=predicted_probs.shape[1])

    # Determine which alternative's 'bucket' the random value
    # might have fallen into.
    possible_alts =\
        (np.arange(1, obs_rows.size + 1)[:, None] *
         (current_cdf >= uniform_draws[None, :]))
    # Give a 'big' value to alternatives that are not chosen
    possible_alts[np.where(possible_alts == 0)] = obs_rows.size + 10
    # Figure out the exact rows/alternatives that were chosen
    chosen_rows = obs_rows[np.argmin(possible_alts, axis=0)]
    return chosen_rows


def simulate_choice_vector(predicted_probs,
                           observation_ids,
                           wide_binary=False,
                           rseed=None):
    """
    Simulates choice outcomes based on the predicted probabilities of each
    alternative for each observation.

    Parameters
    ----------
    predicted_probs : 2D ndarray of floats in (0.0, 1.0).
        Each row should correspond to a particular alternative for a particular
        observation. Each column should correspond to a sampled parameter
        vector. Finally, each element should denote the probability of that
        alternative being chosen by that decision maker, given their
        explanatory variables and the sampled model parameters.
    observation_ids : 1D ndarray of ints.
        Each element should represent an obervation id. Should have
        `observation_ids.shape[0] == predicted_probs.shape[0]`.
    wide_binary : bool, optional.
        Denotes whether `predicted_probs` are for a wide-format dataset of
        binary choices or not.
    rseed : positive int or None, optional.
        The random seed used to simulate the choices. Use when one wants to
        reproduce particular simulations. Default is None.

    Returns
    -------
    simulated_y : 2D ndarray of zeros and ones.
        Each row should correspond to a particular alternative for a particular
        observation. Each column should correspond to a sampled parameter
        vector. Finally, each element will be a one if that row's alternative
        was chosen by that row's decision-maker for that columns simulated
        parameter vector. Otherwise, the element will be a zero. When
        `wide_binary == True`, each element in `simulated_y` will indicate
        whether that row's observation had `y == 1` for that simulation or not.
    """
    # Make predicted_probs 2D
    if predicted_probs.ndim == 1:
        predicted_probs = predicted_probs[:, None]
    elif predicted_probs.ndim > 2:
        msg = 'predicted_probs should have 1 or 2 dimensions.'
        raise ValueError(msg)

    # Make the wide-format binary simulations if necessary
    if wide_binary:
        return _simulate_wide_binary_choices(predicted_probs, rseed=rseed)

    # Determine the unique values in observation_ids
    unique_idx = np.sort(np.unique(observation_ids, return_index=True)[1])
    unique_obs = observation_ids[unique_idx]

    # Determine the rows belonging to each observation
    rows_per_obs = {k: np.where(observation_ids == k)[0] for k in unique_obs}

    # Initialize an array of simulated choices
    choice_vec = np.zeros(predicted_probs.shape, dtype=int)

    # Create an index for the columns
    col_idx = np.arange(predicted_probs.shape[1])

    # Set the seed if desired
    if isinstance(rseed, int):
        np.random.seed(rseed)

    # Populate the array
    for obs_id in progress(unique_obs.tolist(), desc='Simulating Choices'):
        # Determine the exact rows/alternatives chosen in each situation
        chosen_rows =\
            _simulate_choices_for_1obs(obs_id, rows_per_obs, predicted_probs)

        # Store the simulated choice
        choice_vec[chosen_rows, col_idx] = 1

    return choice_vec

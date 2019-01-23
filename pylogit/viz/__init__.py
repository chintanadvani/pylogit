# -*- coding: utf-8 -*-
"""
Visualization module for PyLogit.
"""
from __future__ import absolute_import

from .utils import progress, is_categorical, simulate_choice_vector
from .sim_cdf import plot_simulated_cdf_traces
from .sim_kde import plot_simulated_kde_traces
from .cont_scalars import plot_continous_scalars
from .disc_scalars import plot_discrete_scalars

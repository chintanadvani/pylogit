# -*- coding: utf-8 -*-
"""
Functions for plotting reliability diagrams: smooths of simulated vs observed
outcomes on the y-axis against predicted probabilities on the x-axis.
"""
from __future__ import absolute_import

from copy import deepcopy
from numbers import Number

import scipy.stats
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

from .plot_utils import _label_despine_save_and_show_plot

# Set the plotting style
sbn.set_style('darkgrid')

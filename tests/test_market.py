# -*- coding: utf-8 -*-
"""
Tests for viz.market.py
"""
import unittest
import mock

import numpy as np
import pandas as pd
import numpy.testing as npt

import pylogit.viz.market as market


class MarketShareTests(unittest.TestCase):
    """
    Tests for `market.plot_simulated_market_shares` and supporting functions.
    """
    def setup(self):
        # Set the names of the x and y columns for the dataset
        self.x_label, self.y_label = 'x', 'y'
        original_cols = [self.x_label, self.y_label]
        # Set the x and y values for the tests
        self.MARKET_X = np.tile(np.arange(1, 4), 3)
        self.MARKET_Y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        # This set of "simulated" y-values has 3 ones, 4 twos, and 2 threes
        self.MARKET_SIM_Y =\
            np.array([[1, 0, 1],
                      [0, 1, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0],
                      [1, 0, 1],
                      [0, 0, 1],
                      [1, 1, 0],
                      [0, 0, 0]
                     ])
        # Set the expected boxplot dataframe for the tests
        # Column 2 should denote, with one row per simulation (or observed
        # choices), the number of times the value in column 1 was seen.
        self.BOXPLOT_DF =\
            pd.DataFrame([[1, 1],
                          [1, 0],
                          [1, 2],
                          [2, 1],
                          [2, 3],
                          [2, 0],
                          [3, 1],
                          [3, 0],
                          [3, 1]], columns=original_cols)
        # Set the expected dataframe of observed boxplot values for the tests
        # Note the index has to be str(x) for x in OBS_DF['x'].values
        self.OBS_DF =\
            pd.DataFrame([[1, 1], [2, 1], [3, 1]],
                         columns=original_cols,
                         index=[str(1), str(2), str(3)])
        # Set a display dictionary for tests
        self.display_dict = {1: "one", 2: "two", 3: "three"}
        return None

    # First test the desired functionality with hand-crafted, correct inputs
    # Second, test the desired functionality with hand-crafted, incorrect inputs
    # Third, find the code's limits and test the desired functionality with
    # property-based testing.
    def test_get_objects_for_market_share_plot(self, display=False):
        self.setup()
        # Alias the function being tested
        func = market._get_objects_for_market_share_plot

        # Note the arguments for the various functions being tested
        args_1 = (
            self.MARKET_X, self.MARKET_SIM_Y, self.MARKET_Y,
            self.x_label, self.y_label)
        kwargs_2 = {'display_dict': self.display_dict}

        # Perform the desired tests
        if display:
            # Rename the columns of the expected dataframes
            self.BOXPLOT_DF[self.x_label] =\
                self.BOXPLOT_DF[self.x_label].map(self.display_dict)
            self.OBS_DF[self.x_label] =\
                self.OBS_DF[self.x_label].map(self.display_dict)

            # Get the function results
            boxplot_df, obs_df = func(*args_1, **kwargs_2)
        else:
            # Get the function results
            boxplot_df, obs_df = func(*args_1)

        # Ensure the function results are as expected
        self.assertTrue(isinstance(boxplot_df, pd.DataFrame))
        self.assertTrue(isinstance(obs_df, pd.DataFrame))

        self.assertEqual(len(boxplot_df.columns), 2)
        self.assertEqual(len(obs_df.columns), 2)

        testing_func = npt.assert_equal if display else npt.assert_allclose

        testing_func(
            boxplot_df[self.BOXPLOT_DF.columns.tolist()].values,
            self.BOXPLOT_DF.values)
        testing_func(
            obs_df[self.OBS_DF.columns.tolist()].values,
            self.OBS_DF.values)
        return None

    def test_display_in_get_objects_for_market_share_plot(self):
        self.test_get_objects_for_market_share_plot(display=True)
        return None


    @mock.patch('pylogit.viz.market._label_despine_save_and_show_plot')
    @mock.patch('pylogit.viz.market.plt.Axes')
    @mock.patch('pylogit.viz.market._get_objects_for_market_share_plot')
    @mock.patch('pylogit.viz.market.sbn.stripplot')
    @mock.patch('pylogit.viz.market.sbn.boxplot')
    def test_plot_simulated_market_shares(self,
                                          mock_sbn_box,
                                          mock_sbn_strip,
                                          mock_objects,
                                          mock_ax,
                                          mock_label,
                                         ):
        self.setup()
        # Alias the function being tested
        func = market.plot_simulated_market_shares

        # Set values for the plot
        figsize = (5, 3)
        fig_and_ax =[None, mock_ax]
        fontsize = 14
        title = 'Test Plot'
        box_color = 'grey'
        obs_color = '#a6cee3'
        obs_marker = '+'
        obs_size = 13
        obs_label = 'Test Observed'
        output_file = './test_file.png'
        dpi = 550
        show = False

        # set return values for the tests
        mock_objects.return_value = (self.BOXPLOT_DF, self.OBS_DF)
        mock_ax.return_value = (None, None)
        setattr(mock_ax,
                'get_legend_handles_labels',
                mock.MagicMock(return_value=((None,), (None,))))

        # For the xtick labels, we'll need a two-level, nested Mock
        # We will proceed backwards, first setting the mocks for the objects
        # (i.e. the 'v') that will be returned in the list from get_xticklabels
        xtick_return_values = []
        values_seen = []
        for value in self.OBS_DF[self.x_label].values:
            if value not in values_seen:
                current_mock_v = mock.MagicMock()
                # Note that these lower level mocks will need to have
                # get_text methods that return the correct values.
                current_mock_text = mock.MagicMock(return_value=str(value))
                setattr(current_mock_v, 'get_text', current_mock_text)
                # Add the lower level mock to the return list from the upper
                # level mock
                xtick_return_values.append(current_mock_v)
                values_seen.append(value)
        # Once we have the xticklabels, set them as the return level for the
        # upper level mock.
        setattr(mock_ax,
                'get_xticklabels',
                lambda : xtick_return_values)

        # Determine function arguments and keyword arguments
        args = (self.MARKET_X, self.MARKET_SIM_Y, self.MARKET_Y)
        kwargs = {'x_label': self.x_label,
                  'y_label': self.y_label,
                  'display_dict': self.display_dict,
                  'fig_and_ax': fig_and_ax,
                  'figsize': figsize,
                  'fontsize': fontsize,
                  'title': title,
                  'box_color': box_color,
                  'obs_color': obs_color,
                  'obs_marker': obs_marker,
                  'obs_size': obs_size,
                  'obs_label': obs_label,
                  'output_file': output_file,
                  'dpi': dpi,
                  'show': show}

        # Execute the function and perform the desired tests
        func(*args, **kwargs)
        mock_objects.assert_called_once_with(
            self.MARKET_X,
            self.MARKET_SIM_Y,
            self.MARKET_Y,
            self.x_label,
            self.y_label,
            display_dict=self.display_dict)
        mock_sbn_box.assert_called_once_with(
            x=self.x_label,
            y=self.y_label,
            data=self.BOXPLOT_DF,
            color=box_color,
            ax=mock_ax)
        # mock_sbn_strip.assert_called_once_with(
        #     x=self.x_label,
        #     y=self.y_label,
        #     ax=mock_ax,
        #     color=obs_color,
        #     s=obs_size,
        #     marker=obs_marker,
        #     label=obs_label)
        mock_label.assert_called_once_with(
            x_label=self.x_label,
            y_label=self.y_label,
            fig_and_ax=fig_and_ax,
            fontsize=fontsize,
            y_rot=0,
            y_pad=40,
            title=title,
            output_file=output_file,
            show=show,
            dpi=dpi)
        return None

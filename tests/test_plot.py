# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for module plot.py. Module matplotlib is not required as it's mocked accordingly.
"""


from __future__ import absolute_import, division, print_function

from datetime import datetime, timedelta

import mock
import pytest
from numpy.testing import assert_array_equal
from pandas import Timestamp

import causalimpact.plot as plot
from causalimpact import CausalImpact


def test_plot_original_panel(rand_data, pre_int_period, post_int_period, monkeypatch):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    ax_mock = mock.Mock()
    plotter_mock = mock.Mock()
    plotter_mock.subplot.return_value = ax_mock
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)

    ci.plot(panels=['original'])
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(15, 12))
    plotter_mock.subplot.assert_any_call(1, 1, 1)
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(ci.data.iloc[:, 0], ax_args[0][0][0])
    assert ax_args[0][0][1] == 'k'
    assert ax_args[0][1] == {'label': 'y'}

    inferences = ci.inferences.iloc[1:, :]

    assert_array_equal(inferences['preds'], ax_args[1][0][0])
    assert ax_args[1][0][1] == 'b--'
    assert ax_args[1][1] == {'label': 'Predicted'}

    ax_mock.axvline.assert_called_with(ci.post_period[0] - 1, c='k', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], inferences['preds'].index)
    assert_array_equal(ax_args[0][1], inferences['preds_lower'])
    assert_array_equal(ax_args[0][2], inferences['preds_upper'])
    assert ax_args[1] == {'facecolor': 'blue', 'interpolate': True, 'alpha': 0.25}

    ax_mock.grid.assert_called_with(True, linestyle='--')
    ax_mock.legend.assert_called()

    plotter_mock.show.assert_called_once()


def test_plot_original_panel_date_index(date_rand_data, pre_str_period, post_str_period,
                                        monkeypatch):
    ci = CausalImpact(date_rand_data, pre_str_period, post_str_period)
    ax_mock = mock.Mock()
    plotter_mock = mock.Mock()
    plotter_mock.subplot.return_value = ax_mock
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)

    ci.plot(panels=['original'])
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(15, 12))
    plotter_mock.subplot.assert_any_call(1, 1, 1)
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(ci.data.iloc[:, 0], ax_args[0][0][0])
    assert ax_args[0][0][1] == 'k'
    assert ax_args[0][1] == {'label': 'y'}

    inferences = ci.inferences.iloc[1:, :]

    assert_array_equal(inferences['preds'], ax_args[1][0][0])
    assert ax_args[1][0][1] == 'b--'
    assert ax_args[1][1] == {'label': 'Predicted'}

    date_ = datetime.strptime(ci.post_period[0], "%Y%m%d")
    date_ = date_ + timedelta(days=-1)
    date_ = Timestamp(date_.strftime("%Y-%m-%d %H:%M:%S"))
    ax_mock.axvline.assert_called_with(date_, c='k', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], inferences['preds'].index)
    assert_array_equal(ax_args[0][1], inferences['preds_lower'])
    assert_array_equal(ax_args[0][2], inferences['preds_upper'])
    assert ax_args[1] == {'facecolor': 'blue', 'interpolate': True, 'alpha': 0.25}

    ax_mock.grid.assert_called_with(True, linestyle='--')
    ax_mock.legend.assert_called()

    plotter_mock.show.assert_called_once()


def test_plot_original_panel_date_index_no_freq(date_rand_data, pre_str_period,
                                                post_str_period, monkeypatch):
    dd = date_rand_data.copy()
    dd.drop(dd.index[10:20])
    ci = CausalImpact(dd, pre_str_period, post_str_period)
    ax_mock = mock.Mock()
    plotter_mock = mock.Mock()
    plotter_mock.subplot.return_value = ax_mock
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)

    ci.plot(panels=['original'])
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(15, 12))
    plotter_mock.subplot.assert_any_call(1, 1, 1)
    ax_args = ax_mock.plot.call_args_list

    assert_array_equal(ci.data.iloc[:, 0], ax_args[0][0][0])
    assert ax_args[0][0][1] == 'k'
    assert ax_args[0][1] == {'label': 'y'}

    inferences = ci.inferences.iloc[1:, :]

    assert_array_equal(inferences['preds'], ax_args[1][0][0])
    assert ax_args[1][0][1] == 'b--'
    assert ax_args[1][1] == {'label': 'Predicted'}

    date_ = datetime.strptime(ci.post_period[0], "%Y%m%d")
    date_ = date_ + timedelta(days=-1)
    date_ = Timestamp(date_.strftime("%Y-%m-%d %H:%M:%S"))
    ax_mock.axvline.assert_called_with(date_, c='k', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], inferences['preds'].index)
    assert_array_equal(ax_args[0][1], inferences['preds_lower'])
    assert_array_equal(ax_args[0][2], inferences['preds_upper'])
    assert ax_args[1] == {'facecolor': 'blue', 'interpolate': True, 'alpha': 0.25}

    ax_mock.grid.assert_called_with(True, linestyle='--')
    ax_mock.legend.assert_called()

    plotter_mock.show.assert_called_once()


def test_plot_pointwise_panel(rand_data, pre_int_period, post_int_period, monkeypatch):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    ax_mock = mock.Mock()
    plotter_mock = mock.Mock()
    plotter_mock.subplot.return_value = ax_mock
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)

    ci.plot(panels=['pointwise'])
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(15, 12))
    plotter_mock.subplot.assert_any_call(1, 1, 1, sharex=ax_mock)
    ax_args = ax_mock.plot.call_args

    inferences = ci.inferences.iloc[1:, :]

    assert_array_equal(inferences['point_effects'], ax_args[0][0])
    assert ax_args[0][1] == 'b--'
    assert ax_args[1] == {'label': 'Point Effects'}

    ax_mock.axvline.assert_called_with(ci.post_period[0] - 1, c='k', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], inferences['point_effects'].index)
    assert_array_equal(ax_args[0][1], inferences['point_effects_lower'])
    assert_array_equal(ax_args[0][2], inferences['point_effects_upper'])
    assert ax_args[1] == {'facecolor': 'blue', 'interpolate': True, 'alpha': 0.25}

    ax_mock.axhline.assert_called_with(y=0, color='k', linestyle='--')

    ax_mock.grid.assert_called_with(True, linestyle='--')
    ax_mock.legend.assert_called()

    plotter_mock.show.assert_called_once()


def test_plot_pointwise_panel_date_index(date_rand_data, pre_str_period, post_str_period,
                                         monkeypatch):
    ci = CausalImpact(date_rand_data, pre_str_period, post_str_period)
    ax_mock = mock.Mock()
    plotter_mock = mock.Mock()
    plotter_mock.subplot.return_value = ax_mock
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)

    ci.plot(panels=['pointwise'])
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(15, 12))
    plotter_mock.subplot.assert_any_call(1, 1, 1, sharex=ax_mock)
    ax_args = ax_mock.plot.call_args

    inferences = ci.inferences.iloc[1:, :]

    assert_array_equal(inferences['point_effects'], ax_args[0][0])
    assert ax_args[0][1] == 'b--'
    assert ax_args[1] == {'label': 'Point Effects'}

    date_ = datetime.strptime(ci.post_period[0], "%Y%m%d")
    date_ = date_ + timedelta(days=-1)
    date_ = Timestamp(date_.strftime("%Y-%m-%d %H:%M:%S"))
    ax_mock.axvline.assert_called_with(date_, c='k', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], inferences['point_effects'].index)
    assert_array_equal(ax_args[0][1], inferences['point_effects_lower'])
    assert_array_equal(ax_args[0][2], inferences['point_effects_upper'])
    assert ax_args[1] == {'facecolor': 'blue', 'interpolate': True, 'alpha': 0.25}

    ax_mock.axhline.assert_called_with(y=0, color='k', linestyle='--')

    ax_mock.grid.assert_called_with(True, linestyle='--')
    ax_mock.legend.assert_called()

    plotter_mock.show.assert_called_once()


def test_plot_pointwise_panel_date_index_no_freq(date_rand_data, pre_str_period,
                                                 post_str_period, monkeypatch):
    dd = date_rand_data.copy()
    dd.drop(dd.index[10:20])
    ci = CausalImpact(date_rand_data, pre_str_period, post_str_period)
    ax_mock = mock.Mock()
    plotter_mock = mock.Mock()
    plotter_mock.subplot.return_value = ax_mock
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)

    ci.plot(panels=['pointwise'])
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(15, 12))
    plotter_mock.subplot.assert_any_call(1, 1, 1, sharex=ax_mock)
    ax_args = ax_mock.plot.call_args

    inferences = ci.inferences.iloc[1:, :]

    assert_array_equal(inferences['point_effects'], ax_args[0][0])
    assert ax_args[0][1] == 'b--'
    assert ax_args[1] == {'label': 'Point Effects'}

    date_ = datetime.strptime(ci.post_period[0], "%Y%m%d")
    date_ = date_ + timedelta(days=-1)
    date_ = Timestamp(date_.strftime("%Y-%m-%d %H:%M:%S"))
    ax_mock.axvline.assert_called_with(date_, c='k', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], inferences['point_effects'].index)
    assert_array_equal(ax_args[0][1], inferences['point_effects_lower'])
    assert_array_equal(ax_args[0][2], inferences['point_effects_upper'])
    assert ax_args[1] == {'facecolor': 'blue', 'interpolate': True, 'alpha': 0.25}

    ax_mock.axhline.assert_called_with(y=0, color='k', linestyle='--')

    ax_mock.grid.assert_called_with(True, linestyle='--')
    ax_mock.legend.assert_called()

    plotter_mock.show.assert_called_once()


def test_plot_cumulative_panel(rand_data, pre_int_period, post_int_period, monkeypatch):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    ax_mock = mock.Mock()
    plotter_mock = mock.Mock()
    plotter_mock.subplot.return_value = ax_mock
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)

    ci.plot(panels=['cumulative'])
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(15, 12))
    plotter_mock.subplot.assert_any_call(1, 1, 1, sharex=ax_mock)
    ax_args = ax_mock.plot.call_args

    inferences = ci.inferences.iloc[1:, :]

    assert_array_equal(inferences['post_cum_effects'], ax_args[0][0])
    assert ax_args[0][1] == 'b--'
    assert ax_args[1] == {'label': 'Cumulative Effect'}

    ax_mock.axvline.assert_called_with(ci.post_period[0] - 1, c='k', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], inferences['post_cum_effects'].index)
    assert_array_equal(ax_args[0][1], inferences['post_cum_effects_lower'])
    assert_array_equal(ax_args[0][2], inferences['post_cum_effects_upper'])
    assert ax_args[1] == {'facecolor': 'blue', 'interpolate': True, 'alpha': 0.25}

    ax_mock.axhline.assert_called_with(y=0, color='k', linestyle='--')

    ax_mock.grid.assert_called_with(True, linestyle='--')
    ax_mock.legend.assert_called()

    plotter_mock.show.assert_called_once()


def test_plot_cumulative_panel_date_index(date_rand_data, pre_str_period, post_str_period,
                                          monkeypatch):
    ci = CausalImpact(date_rand_data, pre_str_period, post_str_period)
    ax_mock = mock.Mock()
    plotter_mock = mock.Mock()
    plotter_mock.subplot.return_value = ax_mock
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)

    ci.plot(panels=['cumulative'])
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(15, 12))
    plotter_mock.subplot.assert_any_call(1, 1, 1, sharex=ax_mock)
    ax_args = ax_mock.plot.call_args

    inferences = ci.inferences.iloc[1:, :]

    assert_array_equal(inferences['post_cum_effects'], ax_args[0][0])
    assert ax_args[0][1] == 'b--'
    assert ax_args[1] == {'label': 'Cumulative Effect'}

    date_ = datetime.strptime(ci.post_period[0], "%Y%m%d")
    date_ = date_ + timedelta(days=-1)
    date_ = Timestamp(date_.strftime("%Y-%m-%d %H:%M:%S"))
    ax_mock.axvline.assert_called_with(date_, c='k', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], inferences['post_cum_effects'].index)
    assert_array_equal(ax_args[0][1], inferences['post_cum_effects_lower'])
    assert_array_equal(ax_args[0][2], inferences['post_cum_effects_upper'])
    assert ax_args[1] == {'facecolor': 'blue', 'interpolate': True, 'alpha': 0.25}

    ax_mock.axhline.assert_called_with(y=0, color='k', linestyle='--')

    ax_mock.grid.assert_called_with(True, linestyle='--')
    ax_mock.legend.assert_called()

    plotter_mock.show.assert_called_once()


def test_plot_cumulative_panel_date_index_no_freq(date_rand_data, pre_str_period,
                                                  post_str_period, monkeypatch):
    ci = CausalImpact(date_rand_data, pre_str_period, post_str_period)
    dd = date_rand_data.copy()
    dd.drop(dd.index[10:20])
    ax_mock = mock.Mock()
    plotter_mock = mock.Mock()
    plotter_mock.subplot.return_value = ax_mock
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)

    ci.plot(panels=['cumulative'])
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(15, 12))
    plotter_mock.subplot.assert_any_call(1, 1, 1, sharex=ax_mock)
    ax_args = ax_mock.plot.call_args

    inferences = ci.inferences.iloc[1:, :]

    assert_array_equal(inferences['post_cum_effects'], ax_args[0][0])
    assert ax_args[0][1] == 'b--'
    assert ax_args[1] == {'label': 'Cumulative Effect'}

    date_ = datetime.strptime(ci.post_period[0], "%Y%m%d")
    date_ = date_ + timedelta(days=-1)
    date_ = Timestamp(date_.strftime("%Y-%m-%d %H:%M:%S"))
    ax_mock.axvline.assert_called_with(date_, c='k', linestyle='--')

    ax_args = ax_mock.fill_between.call_args_list[0]
    assert_array_equal(ax_args[0][0], inferences['post_cum_effects'].index)
    assert_array_equal(ax_args[0][1], inferences['post_cum_effects_lower'])
    assert_array_equal(ax_args[0][2], inferences['post_cum_effects_upper'])
    assert ax_args[1] == {'facecolor': 'blue', 'interpolate': True, 'alpha': 0.25}

    ax_mock.axhline.assert_called_with(y=0, color='k', linestyle='--')

    ax_mock.grid.assert_called_with(True, linestyle='--')
    ax_mock.legend.assert_called()

    plotter_mock.show.assert_called_once()


def test_plot_multi_panels(rand_data, pre_int_period, post_int_period, monkeypatch):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    ax_mock = mock.Mock()
    ax_mock.get_xticklabels.return_value = 'xticklabels'
    plotter_mock = mock.Mock()
    plotter_mock.subplot.return_value = ax_mock
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)

    ci.plot(panels=['original', 'pointwise'], figsize=(10, 10))
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(10, 10))
    plotter_mock.subplot.assert_any_call(2, 1, 1)
    plotter_mock.subplot.assert_any_call(2, 1, 2, sharex=ax_mock)
    plotter_mock.setp.assert_called_once_with('xticklabels', visible=False)
    assert ax_mock.plot.call_count == 3
    plotter_mock.show.assert_called_once()

    ax_mock.reset_mock()
    plot_mock.reset_mock()
    plot_mock.reset_mock()

    ci.plot(panels=['original', 'cumulative'], figsize=(10, 10))
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(10, 10))
    plotter_mock.subplot.assert_any_call(2, 1, 1)
    plotter_mock.subplot.assert_any_call(2, 1, 2, sharex=ax_mock)
    plotter_mock.setp.assert_called_once_with('xticklabels', visible=False)
    assert ax_mock.plot.call_count == 3
    plotter_mock.show.assert_called_once()

    ax_mock.reset_mock()
    plot_mock.reset_mock()
    plot_mock.reset_mock()

    ci.plot(panels=['pointwise', 'cumulative'], figsize=(10, 10))
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(10, 10))
    plotter_mock.subplot.assert_any_call(2, 1, 1, sharex=ax_mock)
    plotter_mock.subplot.assert_any_call(2, 1, 2, sharex=ax_mock)
    plotter_mock.setp.assert_called_once_with('xticklabels', visible=False)
    assert ax_mock.plot.call_count == 2
    plotter_mock.show.assert_called_once()

    ax_mock.reset_mock()
    plot_mock.reset_mock()
    plot_mock.reset_mock()

    ci.plot(panels=['pointwise', 'cumulative', 'original'], figsize=(10, 10))
    plot_mock.assert_called_once()
    plotter_mock.figure.assert_called_with(figsize=(10, 10))
    plotter_mock.subplot.assert_any_call(3, 1, 1)
    plotter_mock.subplot.assert_any_call(3, 1, 2, sharex=ax_mock)
    plotter_mock.subplot.assert_any_call(3, 1, 3, sharex=ax_mock)
    plotter_mock.setp.assert_called_with('xticklabels', visible=False)
    assert ax_mock.plot.call_count == 4
    plotter_mock.show.assert_called_once()


def test_plot_raises_when_not_initialized(rand_data, pre_int_period, post_int_period,
                                          monkeypatch):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    ci.summary_data = None
    plotter_mock = mock.Mock()
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)
    with pytest.raises(RuntimeError):
        ci.plot()


def test_plot_raises_wrong_input_panel(rand_data, pre_int_period, post_int_period,
                                       monkeypatch):
    ci = CausalImpact(rand_data, pre_int_period, post_int_period)
    plotter_mock = mock.Mock()
    plot_mock = mock.Mock(return_value=plotter_mock)
    monkeypatch.setattr(plot.Plot, '_get_plotter', plot_mock)
    with pytest.raises(ValueError) as excinfo:
        ci.plot(panels=['test'])
    assert str(excinfo.value) == (
        '"test" is not a valid panel. Valid panels are: '
        '"original", "pointwise", "cumulative".'
    )

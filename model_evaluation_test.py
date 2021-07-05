# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for model_evaluation."""

import unittest

from google.cloud import bigquery
import mock

from solutions.crystalvalue import model_evaluation

_NUMBER_BINS = 5


class MockModelLtvTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_client = mock.patch.object(bigquery, 'Client', autospec=True)
    self.mock_client.project = 'test_project'
    self.fetch_test_set_predictions_from_bigquery = mock.patch.object(
        model_evaluation,
        '_fetch_test_set_predictions_from_bigquery',
        autospec=True).start()
    self.mock_calculate_bin_averages = mock.patch.object(
        model_evaluation, '_calculate_bin_averages', autospec=True).start()
    self.mock_calculate_normalized_mae = mock.patch.object(
        model_evaluation, '_calculate_normalized_mae', autospec=True).start()
    self.mock_compute_gini = mock.patch.object(
        model_evaluation, '_compute_gini', autospec=True).start()
    self.mock_compute_gini_normalized = mock.patch.object(
        model_evaluation, '_compute_gini_normalized', autospec=True).start()
    self.mock_plot_summary_stats = mock.patch.object(
        model_evaluation, '_plot_summary_stats', autospec=True).start()
    self.mock_create_summary_stats_data = mock.patch.object(
        model_evaluation, '_create_summary_stats_data', autospec=True).start()
    self.dataset_name = 'dataset_name'
    self.predictions_table = 'predictions_table'
    self.model_display_name = 'model_display_name'
    self.table_evaluation_stats = 'table_evaluation_stats'
    self.number_bins = _NUMBER_BINS

  @mock.patch.object(model_evaluation, '_load_table_to_bigquery', autospec=True)
  def test_client_called_once(self, mock_load_table_to_bigquery):
    model_evaluation.evaluate_model_predictions(
        bigquery_client=self.mock_client,
        dataset_name=self.dataset_name,
        predictions_table=self.predictions_table,
        model_display_name=self.model_display_name,
        table_evaluation_stats=self.table_evaluation_stats,
        number_bins=self.number_bins)

    mock_load_table_to_bigquery.assert_called_once()


if __name__ == '__main__':
  unittest.main()

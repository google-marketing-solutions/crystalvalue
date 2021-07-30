# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for crystalvalue.feature_engineering."""

import unittest

from google.cloud import bigquery
import mock

from crystalvalue import feature_engineering


class FeatureEngineeringTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_client = mock.patch.object(
        bigquery, 'Client', autospec=True)
    self.mock_client.project = 'test_project'
    self.mock_read = mock.patch.object(
        feature_engineering, '_read_file', autospec=True).start()
    self.mock_write = mock.patch.object(
        feature_engineering, '_write_file', autospec=True).start()
    self.dataset_id = 'test_dataset_id'
    self.transaction_table_name = 'test_transaction_table_name'
    self.destination_table_name = 'test_destination_table_name'
    self.query_template_train_file = 'template_query.txt'
    self.write_executed_query_file = 'generated_query.txt'
    self.numerical_features = frozenset(['numerical_column'])
    self.non_numerical_features = frozenset(['categorical_columm'])

  @mock.patch.object(feature_engineering, '_run_query', autospec=True)
  def test_client_is_called_once(self, mock_run_query):

    feature_engineering.build_train_data(
        bigquery_client=self.mock_client,
        dataset_id=self.dataset_id,
        transaction_table_name=self.transaction_table_name,
        destination_table_name=self.destination_table_name,
        numerical_features=self.numerical_features,
        non_numerical_features=self.non_numerical_features,
        query_template_train_file=self.query_template_train_file,
        write_executed_query_file=self.write_executed_query_file)

    mock_run_query.assert_called_once()

if __name__ == '__main__':
  unittest.main()

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

import unittest

from google.cloud import bigquery
import mock

from src import feature_engineering


class FeatureEngineeringTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_client = mock.create_autospec(bigquery.Client, instance=True)
    self.mock_client.project = 'test_project'
    self.mock_read = mock.patch.object(
        feature_engineering, '_read_file', autospec=True).start()
    self.mock_write = mock.patch.object(
        feature_engineering, '_write_file', autospec=True).start()
    self.dataset_id = 'test_dataset_id'
    self.read_query_file = 'query.txt'
    self.transaction_table_name = 'test_transaction_table_name'
    self.destination_table_name = 'test_destination_table_name'
    self.input_data_types = {
        'numeric': ['numerical_column'],
        'boolean': ['boolean_column'],
        'string_or_categorical': ['string_column']
    }

  def test_build_train_query_reads_query(self):

    feature_engineering.build_query(
        bigquery_client=self.mock_client,
        dataset_id=self.dataset_id,
        transaction_table_name=self.transaction_table_name,
        input_data_types=self.input_data_types)

    self.mock_read.assert_called_once()

  def test_run_query_writes_file(self):

    feature_engineering.run_query(
        bigquery_client=self.mock_client,
        dataset_id=self.dataset_id,
        query_file=self.read_query_file,
        destination_table_name=self.destination_table_name)

    self.mock_read.assert_called_once()

  def test_build_query_substitutions(self):

    with mock.patch.object(
        feature_engineering, '_read_file', return_value='{features_sql}'
    ):
      query, _ = feature_engineering.build_query(
          bigquery_client=self.mock_client,
          dataset_id=self.dataset_id,
          transaction_table_name=self.transaction_table_name,
          input_data_types=self.input_data_types)

      expected_numeric_str = 'MAX(numerical_column) AS max_numerical_column'
      expected_boolean_str = ('MAX(CAST(boolean_column AS INT))'
                              ' AS max_boolean_column')
      expected_string_str = ('COUNT (DISTINCT string_column)'
                             ' AS n_distinct_string_column')

      self.assertIn(expected_numeric_str, query)
      self.assertIn(expected_boolean_str, query)
      self.assertIn(expected_string_str, query)


if __name__ == '__main__':
  unittest.main()

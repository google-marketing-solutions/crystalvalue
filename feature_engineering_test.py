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
    self.mock_client = mock.create_autospec(bigquery.Client, instance=True)
    self.mock_client.project = 'test_project'
    self.mock_read = mock.patch.object(
        feature_engineering, '_read_file', autospec=True).start()
    self.mock_write = mock.patch.object(
        feature_engineering, '_write_file', autospec=True).start()
    self.dataset_id = 'test_dataset_id'
    self.transaction_table_name = 'test_transaction_table_name'
    self.destination_table_name = 'test_destination_table_name'
    self.query_template_train_sql = 'SELECT {features_sql} FROM Dataset'
    self.write_executed_query_file = 'generated_query.txt'
    self.numerical_features = frozenset(['numerical_column'])
    self.string_or_categorical_features = frozenset(['categorical_column'])
    self.bool_features = frozenset(['bool_column'])
    self.array_features = frozenset(['array_column'])

  def test_build_query_function_substitutions(self):
    query = feature_engineering.build_query_function(
        bigquery_client=self.mock_client,
        transaction_table_name=self.transaction_table_name,
        numerical_features=self.numerical_features,
        string_or_categorical_features=self.string_or_categorical_features,
        dataset_id=self.dataset_id,
        query_template_train_sql=self.query_template_train_sql,
        write_executed_query_file=self.write_executed_query_file)

    self.assertIn('MAX(numerical_column) as max_numerical_column', query)
    self.assertIn('SUM(numerical_column) as sum_numerical_column', query)
    self.assertIn('MIN(numerical_column) as min_numerical_column', query)
    self.assertIn('AVG(numerical_column) as avg_numerical_column', query)
    self.assertIn(
        'TRIM(STRING_AGG(DISTINCT categorical_column, " " ORDER BY categorical_column)) AS categorical_column',
        query)

  def test_build_query_function_array_bool_substitutions(self):
    query = feature_engineering.build_query_function(
        bigquery_client=self.mock_client,
        transaction_table_name=self.transaction_table_name,
        numerical_features=self.numerical_features,
        string_or_categorical_features=self.string_or_categorical_features,
        bool_features=self.bool_features,
        array_features=self.array_features,
        dataset_id=self.dataset_id,
        query_template_train_sql=self.query_template_train_sql,
        write_executed_query_file=self.write_executed_query_file)

    self.assertIn('COUNTIF(bool_column) AS bool_column', query)
    self.assertIn('COUNTIF(NOT bool_column) AS not_bool_column', query)
    self.assertIn('ARRAY_CONCAT_AGG(array_column) AS array_column', query)

  def test_run_query_reads_file(self):

    feature_engineering.run_query(
        bigquery_client=self.mock_client,
        dataset_id=self.dataset_id,
        query_sql=self.query_template_train_sql,
        query_file=self.write_executed_query_file,
        destination_table_name=self.destination_table_name)

    self.mock_read.assert_called_once()


if __name__ == '__main__':
  unittest.main()

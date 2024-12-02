# Copyright 2024 Google LLC
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

import tempfile
import unittest

import mock

from src import ga4_data


class GA4DataTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.ga4_project_id = 'ga4_project'
    self.ga4_dataset_id = 'ga4_dataset'
    self.ga4_events_table_name = 'ga4_events'

  def test_read_yaml_file(self):
    yaml_content = """
    event_name:
      - test_event
    event_params:
      test_event:
        - test_event_param
    """
    expected_dict = {
        'event_name': ['test_event'],
        'event_params': {'test_event': ['test_event_param']},
    }
    with tempfile.NamedTemporaryFile() as temp_file:

      with open(temp_file.name, 'w') as f:
        f.write(yaml_content)

      read_from_file = ga4_data._read_yaml_file(temp_file.name)
      self.assertEqual(read_from_file, expected_dict)

  def test_ga4_query_substitutions(self):
    sample_yaml = {
        'event_name': ['test_event'],
        'event_params': {'test_event': ['test_event_param']},
        'user_properties': ['test_user_property'],
        'other': ['test.other']
    }

    with mock.patch.object(
        ga4_data, '_read_yaml_file', return_value=sample_yaml
    ):
      query = ga4_data.build_ga4_query(
          ga4_project_id=self.ga4_project_id,
          ga4_dataset_id=self.ga4_dataset_id,
          ga4_events_table_name=self.ga4_events_table_name,
      )
      expected_event_name_str = ('event_name = "test_event" '
                                 'AS event_is_test_event')
      expected_event_params_str = """
          (
            IF(
              event_name = 'test_event',
              (
                SELECT
                  ep.value.string_value
                FROM UNNEST(event_params) AS ep
                WHERE ep.key = 'test_event_param'
              ),
            NULL
            )
          ) AS test_event_test_event_param"""

      expected_user_properties_str = """
          IFNULL(
            (
              SELECT
                CONCAT(up.key, '_', up.value.string_value)
              FROM UNNEST(user_properties) AS up
              WHERE up.key = 'test_user_property'
            ),
            '(not set)') AS user_test_user_property"""

      expected_other_str = 'test.other AS test_other'

      # Remove extra whitespace because of indentation before comparing strings.
      self.assertIn(
          ' '.join(expected_event_name_str.split()),
          ' '.join(query.split())
      )
      self.assertIn(
          ' '.join(expected_event_params_str.split()),
          ' '.join(query.split())
      )
      self.assertIn(
          ' '.join(expected_user_properties_str.split()),
          ' '.join(query.split())
      )
      self.assertIn(
          ' '.join(expected_other_str.split()),
          ' '.join(query.split())
      )

  def test_write_file(self):
    sample_query = """
    SELECT
      test_column
    FROM
      `test_project.test_dataset.test_table`
    """

    with tempfile.NamedTemporaryFile() as temp_file:
      ga4_data._write_file(sample_query, temp_file.name)
      with open(temp_file.name, 'r') as f:
        read_from_file = f.read()
      self.assertEqual(sample_query, read_from_file)


if __name__ == '__main__':
  unittest.main()

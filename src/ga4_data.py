# Copyright 2024 Google LLC.
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
"""Module for creating a flat table from GA4 data."""

import logging
from typing import Any, Optional

import yaml

logging.getLogger().setLevel(logging.INFO)

_GA4_YAML_FILE = 'src/ga4_features.yaml'


def _read_yaml_file(file_path: str) -> dict[str, Any]:
  """Reads a YAML file and returns its contents as a Python dictionary.

  Args:
    file_path: The path to the YAML file

  Returns:
    A Python dictionary containing the YAML file's contents
  """
  with open(file_path, 'r') as f:
    return yaml.safe_load(f)


def _write_file(query: str, file_name: str) -> None:
  """Writes query to file."""
  with open(file_name, 'w') as f:
    f.write(query)


def build_ga4_query(
    ga4_project_id: str,
    ga4_dataset_id: str,
    ga4_events_table_name: str,
    write_query_to_file: Optional[str] = None,
) -> str:
  """Creates a GA4 query based on the inputs.

  Args:
    ga4_project_id: The project ID with GA4 data
    ga4_dataset_id: The GA4 dataset ID
    ga4_events_table_name: The GA4 events table name to use for the query
    write_query_to_file: If provided, the query will be written to the file

  Returns:
    Query with substituted values
  """

  ga4_sql_template = """
  SELECT
    user_pseudo_id AS customer_id,
    PARSE_DATE('%Y%m%d', event_date) AS date,
    IFNULL(event_value_in_usd, 0) AS value,
    {ga4_features}
  FROM `{project_id}.{dataset_id}.{table_name}`
  """

  features_input = _read_yaml_file(_GA4_YAML_FILE)

  logging.info(
      'Using the following inputs for building GA4 query \n %s', features_input
  )

  # Build the list of features to be used in the query based on the inputs
  features_list = []
  for feature in features_input.get('event_name', []):
    features_list.append(f'event_name = "{feature}" AS event_is_{feature}')

  for event in features_input.get('event_params', []):
    for param in features_input['event_params'][event]:
      features_list.append(f"""
          (
            IF(
              event_name = '{event}',
              (
                SELECT
                  ep.value.string_value
                FROM UNNEST(event_params) AS ep
                WHERE ep.key = '{param}'
              ),
            NULL
            )
          ) AS {event}_{param}
      """)

  for feature in features_input.get('user_properties', []):
    features_list.append(f"""
        IFNULL(
          (
            SELECT
              CONCAT(up.key, '_', up.value.string_value)
            FROM UNNEST(user_properties) AS up
            WHERE up.key = '{feature}'
          ),
          '(not set)') AS user_{feature}
    """)

  for feature in features_input.get('other', []):
    features_list.append(f'{feature} AS {feature.replace(".", "_")}')

  substituted_query = ga4_sql_template.format(
      project_id=ga4_project_id,
      dataset_id=ga4_dataset_id,
      table_name=ga4_events_table_name,
      ga4_features=', \n'.join(features_list),
  )

  if write_query_to_file:
    _write_file(substituted_query, write_query_to_file)
    logging.info(
        'Query successfully written to: "%r"', write_query_to_file
    )

  return substituted_query


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
"""Module for feature engineering via BigQuery for crystalvalue pipeline."""

import datetime
import logging
from typing import Collection, Dict, List, Mapping, Optional

from google.cloud import bigquery
import pandas as pd

logging.getLogger().setLevel(logging.INFO)

# BigQuery aggregate functions to apply to numerical and boolean columns.
# https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate_functions
_NUMERICAL_TRANSFORMATIONS = frozenset(['AVG', 'MAX', 'MIN', 'SUM'])

# SQL templates library
_TRAIN_QUERY_TEMPLATE_FILES = {
    'train_user_split': './sql_templates/train_user_split.sql'
}


def _detect_feature_types(bigquery_client: bigquery.Client, dataset_id: str,
                          table_name: str,
                          ignore_columns: List[str]) -> Dict[str, List[str]]:
  """Detects the feature types using the schema in BigQuery.

  Args:
    bigquery_client: BigQuery client.
    dataset_id: The Bigquery dataset_id.
    table_name: The Bigquery table name with transactions.
    ignore_columns: The column names to ignore to identify features.

  Returns:
    A mapping of features types (boolean, numerical, string or categorical) to
    feature names.
  """
  dataset_reference = bigquery_client.dataset(
      dataset_id, project=bigquery_client.project)
  table_reference = dataset_reference.table(table_name)
  table = bigquery_client.get_table(table_reference)
  logging.info(
      'Detecting features types in project_id %r in dataset %r in table %r',
      bigquery_client.project, dataset_id, table_name)

  features = {}
  for feature in table.schema:
    if feature.name not in ignore_columns:
      if feature.mode == 'REPEATED':
        features.setdefault('array', []).append(feature.name)
      elif feature.field_type == 'BOOLEAN':
        features.setdefault('boolean', []).append(feature.name)
      elif feature.field_type in ['INTEGER', 'FLOAT', 'NUMERIC']:
        features.setdefault('numeric', []).append(feature.name)
      else:
        features.setdefault('string_or_categorical', []).append(feature.name)
  return features


def _read_file(file_name: str) -> str:
  """Reads file."""
  with open(file_name, 'r') as f:
    return f.read()


def _write_file(query: str, file_name: str) -> None:
  """Writes query to file."""
  with open(file_name, 'w') as f:
    f.write(query)


def build_train_query(
    bigquery_client: bigquery.Client,
    dataset_id: str,
    transaction_table_name: str,
    query_type: str = 'train_user_split',
    numerical_transformations: Collection[str] = _NUMERICAL_TRANSFORMATIONS,
    write_executed_query_file: Optional[str] = None,
    features: Optional[Mapping[str, str]] = None,
    ignore_columns: Optional[List[str]] = None,
    days_lookback: int = 365,
    days_lookahead: int = 365,
    customer_id_column: str = 'customer_id',
    date_column: str = 'date',
    value_column: str = 'value',
    window_date: Optional[str] = None) -> str:
  """Builds training data from transaction data through BigQuery.

  This function takes a transaction dataset (a BigQuery table that includes
  information about purchases) and creates an SQL query to generate a machine
  learning-ready dataset that can be ingested by AutoML. The SQL query can be
  written to the file path `write_executed_query_file` for manual modifications.
  Data types will be automatically detected from the BigQuery schema if
  `features` argument is not provided. Columns can be REPEATED, however note
  that RECORD type (i.e. SQL STRUCT type) is currently not supported.

  Args:
    bigquery_client: BigQuery client.
    dataset_id: The Bigquery dataset_id.
    transaction_table_name: The Bigquery table name with transactions.
    query_type: The query type. Has to be one of the keys in
      _TRAIN_QUERY_TEMPLATE_FILES. See README for more information.
    numerical_transformations: The types of transformations for numerical
      features.
    write_executed_query_file: File path to write the generated SQL query.
    features: The mapping of feature types to feature names.
    ignore_columns: Column name to ignore when automatically detecting features.
    days_lookback: The number of days to look back to create features.
    days_lookahead: The number of days to look ahead to predict value.
    customer_id_column: The name of the customer id column.
    date_column: The name of the date column.
    value_column: The name of the value column.
    window_date: The date to create 'customer-windows'

  Returns:
    The SQL script to generate training data ready for machine learning.
  """
  if query_type not in _TRAIN_QUERY_TEMPLATE_FILES:
    raise ValueError(
        f'{query_type} not one of {_TRAIN_QUERY_TEMPLATE_FILES.keys()}')

  if not features:
    if not ignore_columns:
      ignore_columns = [customer_id_column, date_column]
    else:
      ignore_columns = [customer_id_column, date_column] + ignore_columns
    features = _detect_feature_types(
        bigquery_client,
        dataset_id,
        transaction_table_name,
        ignore_columns=ignore_columns)
    if not features:
      raise ValueError('No features detected')
    for feature_type in features:
      for feature in features[feature_type]:
        logging.info('Detected %r feature %r', feature_type, feature)

  features_list = []
  if 'numeric' in features:
    for feature in features['numeric']:
      for transformation in numerical_transformations:
        features_list.append(f'{transformation}({feature}) as '
                             f'{transformation.lower()}_{feature}')

  if 'boolean' in features:
    for feature in features['boolean']:
      for transformation in numerical_transformations:
        features_list.append(f'{transformation}(CAST({feature} AS INT)) as '
                             f'{transformation.lower()}_{feature}')

  if 'string_or_categorical' in features:
    for feature in features['string_or_categorical']:
      features_list.append(f'TRIM(STRING_AGG(DISTINCT {feature}, " " ORDER BY '
                           f'{feature})) AS {feature}')

  if not window_date:
    window_date = (datetime.date.today() -
                   datetime.timedelta(days=days_lookback)).strftime('%Y-%m-%d')
    logging.info('Using window date %r', window_date)
    logging.info('Using a lookback window of %r days to create features',
                 days_lookback)
    logging.info('Using a lookahead window of %r days to predict value',
                 days_lookahead)

  query_template_train = _read_file(_TRAIN_QUERY_TEMPLATE_FILES[query_type])

  substituted_query = query_template_train.format(
      project_id=bigquery_client.project,
      dataset_id=dataset_id,
      table_name=transaction_table_name,
      window_date=window_date,
      customer_id_column=customer_id_column,
      date_column=date_column,
      value_column=value_column,
      days_lookback=days_lookback,
      days_lookahead=days_lookahead,
      features_sql=', \n'.join(features_list))

  if write_executed_query_file:
    _write_file(substituted_query, write_executed_query_file)
    logging.info('Query successfully written to: "%r"',
                 write_executed_query_file)

  return substituted_query


def run_query(
    bigquery_client: bigquery.Client,
    dataset_id: str,
    destination_table_name: str = 'training_data',
    query_sql: Optional[str] = None,
    query_file: Optional[str] = None,
    location: str = 'europe-west4',
) -> pd.DataFrame:
  """Runs a query in BigQuery and returns the result.

  Args:
    bigquery_client: BigQuery client.
    dataset_id: The Bigquery dataset_id.
    destination_table_name: The table to write to.
    query_sql: The SQL query to execute.
    query_file: Path to the SQL query to execute.
    location: The location to write the table in BigQuery.

  Returns:
    The result of the executed query as a Pandas DataFrame.
  """
  if query_file:
    query = _read_file(query_file)
  elif query_sql:
    query = query_sql
  else:
    raise ValueError('One of `query_sql` or `query_file` must be provided')

  table_id = f'{bigquery_client.project}.{dataset_id}.{destination_table_name}'
  job_config = bigquery.QueryJobConfig(
      destination=table_id,
      write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
  data = bigquery_client.query(
      query, job_config=job_config, location=location).result().to_dataframe()
  logging.info('Created table %r in location %r', table_id, location)
  return data

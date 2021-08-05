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
"""Module for feature engineering via BigQuery for crystal value pipeline."""

import datetime
import logging
from typing import FrozenSet, List, Optional, Tuple

from google.cloud import bigquery
import pandas as pd

logging.getLogger().setLevel(logging.INFO)

# BigQuery aggregate functions to apply to numerical columns.
# https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate_functions
_NUMERICAL_TRANSFORMATIONS = frozenset(['AVG', 'MAX', 'MIN', 'SUM'])


def _detect_feature_types(
    bigquery_client: bigquery.Client, dataset_id: str, table_name: str,
    ignore_columns: List[str]
) -> Tuple[List[str], List[str], List[str], List[str]]:
  """Detects the features types using the schema in BigQuery.

  Args:
    bigquery_client: BigQuery client.
    dataset_id: The Bigquery dataset_id.
    table_name: The Bigquery table name with transactions.
    ignore_columns: The column names to ignore to create features.

  Returns:
    Tuple of a list of numerical features, non numerical, bool, array features.
  """

  numerical_features = []
  string_or_categorical_features = []
  bool_features = []
  array_features = []

  dataset_reference = bigquery_client.dataset(
      dataset_id, project=bigquery_client.project)
  table_reference = dataset_reference.table(table_name)
  table = bigquery_client.get_table(table_reference)

  for feature in table.schema:
    if feature.name not in ignore_columns:
      if feature.mode == 'REPEATED':
        array_features.append(feature.name)
      elif feature.field_type == 'BOOLEAN':
        bool_features.append(feature.name)
      elif feature.field_type in ['INTEGER', 'FLOAT', 'NUMERIC']:
        numerical_features.append(feature.name)
      else:
        string_or_categorical_features.append(feature.name)
  return numerical_features, string_or_categorical_features, bool_features, array_features


def _read_file(file_name: str) -> str:
  """Reads file."""
  with open(file_name, 'r') as f:
    return f.read()


def _write_file(query: str, file_name: str) -> None:
  """Writes query to file."""
  with open(file_name, 'w') as f:
    f.write(query)
  logging.info('Wrote generated query to %r', file_name)


def build_query_function(
    bigquery_client: bigquery.Client,
    dataset_id: str,
    transaction_table_name: str,
    query_template_train_file: Optional[str] = None,
    query_template_train_sql: Optional[str] = None,
    write_executed_query_file: Optional[str] = None,
    numerical_features: Optional[FrozenSet[str]] = frozenset([]),
    string_or_categorical_features: Optional[FrozenSet[str]] = frozenset([]),
    bool_features: Optional[FrozenSet[str]] = frozenset([]),
    array_features: Optional[FrozenSet[str]] = frozenset([]),
    days_look_back: int = 365,
    days_look_ahead: int = 365,
    customer_id_column: str = 'customer_id',
    date_column: str = 'date',
    value_column: str = 'value',
    numerical_transformations: FrozenSet[str] = _NUMERICAL_TRANSFORMATIONS,
    window_date: Optional[str] = None) -> Optional[pd.DataFrame]:
  """Builds training data from transaction data through BigQuery.

  This function takes a transaction dataset (a BigQuery table that includes
  information about purchases) and creates a SQL query to generate a machine
  learning-ready dataset that can be ingested by AutoML. The SQL query can be
  written to the file path `write_executed_query_file` for manual modifications.
  Data types will be automatically detected from the BigQuery schema if
  `numerical_features` and `string_or_categorical_features` are not provided.
  Columns can be REPEATED, however note that RECORD type (i.e. SQL STRUCT type)
  is currently not supported.

  Args:
    bigquery_client: BigQuery client.
    dataset_id: The Bigquery dataset_id.
    transaction_table_name: The Bigquery table name with transactions.
    query_template_train_file: File path with the template SQL query. Must be
      provided if query_template_train_sql is not provided.
    query_template_train_sql: SQL with the template query. Must be provided if
      query_template_train_file is not provided.
    write_executed_query_file: File path to write the generated SQL query.
    numerical_features: The names of numerical features to be processed.
    string_or_categorical_features: The names of non-numerical features to be
      processed. These should be either categorical or text features.
    bool_features: The names of bool features to be processed.
    array_features: The names of array features to be processed.
    days_look_back: The number of days to look back to create features.
    days_look_ahead: The number of days to look ahead to predict value.
    customer_id_column: The name of the customer id column.
    date_column: The name of the date column.
    value_column: The name of the value column.
    numerical_transformations: The types of transformations for numerical
      features.
    window_date: The date to create 'customer-windows'

  Returns:
    The SQL script to generate training data ready for machine learning.
  """
  # TODO(): Refactor _detect_feature_types
  if not numerical_features and not string_or_categorical_features:
    numerical_features, string_or_categorical_features, bool_features, array_features = _detect_feature_types(
        bigquery_client,
        dataset_id,
        transaction_table_name,
        ignore_columns=[customer_id_column, date_column])
    logging.info('Detected Numerical Features:')
    for feature in numerical_features:
      logging.info({feature})
    logging.info('Detected Non-Numerical Features:')
    for feature in string_or_categorical_features:
      logging.info({feature})
    logging.info('Detected Bool Features:')
    for feature in bool_features:
      logging.info({feature})
    logging.info('Detected Array Features:')
    for feature in array_features:
      logging.info({feature})

  numerical_features_sql = []
  for feature in numerical_features:
    for transformation in numerical_transformations:
      numerical_features_sql.append((f'{transformation}({feature}) as '
                                     f'{transformation.lower()}_{feature}'))
  numerical_features_sql = ', \n'.join(numerical_features_sql)

  string_or_categorical_features_sql = []
  for feature in string_or_categorical_features:
    string_or_categorical_features_sql.append(
        (f'TRIM(STRING_AGG(DISTINCT {feature}, " " '
         f'ORDER BY {feature})) AS {feature}'))
  string_or_categorical_features_sql = ', \n'.join(
      string_or_categorical_features_sql)

  bool_features_sql = []
  for feature in bool_features:
    bool_features_sql.append((f'COUNTIF({feature}) AS {feature}'))
    bool_features_sql.append((f'COUNTIF(NOT {feature}) AS not_{feature}'))
  bool_features_sql = ', \n'.join(bool_features_sql)

  array_agg_features_sql = []
  dedup_sql = []
  for feature in array_features:
    array_agg_features_sql.append((f'ARRAY_CONCAT_AGG({feature}) AS {feature}'))
    dedup_sql.append(f'Dedup({feature}) AS {feature}')
  array_features_sql = ', \n'.join(array_features)
  array_agg_features_sql = ', \n'.join(array_agg_features_sql)
  array_dedup_features_sql = ', \n'.join(dedup_sql)

  if not window_date:
    window_date = (datetime.date.today() -
                   datetime.timedelta(days=365)).strftime('%Y-%m-%d')

  if query_template_train_file:
    query_template_train = _read_file(query_template_train_file)
  elif query_template_train_sql:
    query_template_train = query_template_train_sql
  else:
    raise ValueError(
        'One of `query_template_train_sql` or `query_template_train_file`'
        'must be provided')

  features_sql = ', \n'.join([
      numerical_features_sql, string_or_categorical_features_sql,
      bool_features_sql, array_agg_features_sql
  ])

  substituted_query = query_template_train.format(
      project_id=bigquery_client.project,
      dataset_id=dataset_id,
      table_name=transaction_table_name,
      window_date=window_date,
      customer_id_column=customer_id_column,
      date_column=date_column,
      value_column=value_column,
      days_look_back=days_look_back,
      days_look_ahead=days_look_ahead,
      features_sql=features_sql,
      array_features_sql=array_features_sql,
      array_dedup_features_sql=array_dedup_features_sql)

  if write_executed_query_file:
    _write_file(substituted_query, write_executed_query_file)
    logging.info('Query successfully written to: "%s"',
                 write_executed_query_file)

  return substituted_query


def run_query(
    bigquery_client: bigquery.Client,
    query_sql: Optional[str],
    query_file: Optional[str],
    dataset_id: str,
    destination_table_name: str = 'training_data',
    location: str = 'europe-west4',
) -> pd.DataFrame:
  """Runs a query in BigQuery and returns the result.

  Args:
    bigquery_client: BigQuery client.
    query_sql: The SQL query to execute. Either query or query_file MUST be
      specified.
    query_file: Path to the SQL query to execute. Either query or query_file
      MUST be specified.
    dataset_id: The Bigquery dataset_id.
    destination_table_name: The table to write to.
    location: The location to write the table in BigQuery.

  Returns:
    The result of the executed query as a Pandas DataFrame.
  """
  if query_file:
    query = _read_file(query_file)
  elif query_sql:
    query = query_sql
  else:
    raise ValueError('One of `query` or `query_file` must be provided')

  table_id = f'{bigquery_client.project}.{dataset_id}.{destination_table_name}'
  job_config = bigquery.QueryJobConfig(
      destination=table_id,
      write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
  data = bigquery_client.query(
      query, job_config=job_config, location=location).result().to_dataframe()
  logging.info('Created table %r', table_id)
  return data

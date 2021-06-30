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
_NUMERICAL_TRANSFORMATIONS = frozenset(['AVG', 'MAX', 'MIN'])


def _detect_feature_types(
    bigquery_client: bigquery.Client,
    dataset_id: str,
    table_name: str,
    ignore_columns: List[str]) -> Tuple[List[str], List[str]]:
  """Detects the features types using the schema in BigQuery.

  Args:
    bigquery_client: BigQuery client.
    dataset_id: The Bigquery dataset_id.
    table_name: The Bigquery table name with transactions.
    ignore_columns: The column names to ignore to create features.

  Returns:
    Tuple of a list of numerical features and a list of non numerical features.
  """

  numerical_features = []
  non_numerical_features = []

  dataset_reference = bigquery_client.dataset(
      dataset_id, project=bigquery_client.project)
  table_reference = dataset_reference.table(table_name)
  table = bigquery_client.get_table(table_reference)

  for feature in table.schema:
    if feature.name not in ignore_columns:
      if feature.field_type in ['INTEGER', 'FLOAT']:
        numerical_features.append(feature.name)
      else:
        non_numerical_features.append(feature.name)
  return numerical_features, non_numerical_features


def _read_file(file_name: str) -> str:
  """Reads file."""
  with open(file_name, 'r') as f:
    return f.read()


def _write_file(query: str, file_name: str) -> None:
  """Writes query to file."""
  with open(file_name, 'w') as f:
    f.write(query)
  logging.info('Wrote generated query to %r', file_name)


def _run_query(
    bigquery_client: bigquery.Client,
    query: str,
    dataset_id: str,
    destination_table_name: str,
    location: str) -> pd.DataFrame:
  """Runs a query in BigQuery and returns the result.

  Args:
    bigquery_client: BigQuery client.
    query: The SQL query to execute.
    dataset_id: The Bigquery dataset_id.
    destination_table_name: The table to write to.
    location: The location to write the table in BigQuery.

  Returns:
    The result of the executed query as a Pandas DataFrame.
  """
  table_id = f'{bigquery_client.project}.{dataset_id}.{destination_table_name}'
  job_config = bigquery.QueryJobConfig(
      destination=table_id,
      write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
  data = bigquery_client.query(
      query, job_config=job_config,
      location=location).result().to_dataframe()
  logging.info('Created table %r', table_id)
  return data


def build_train_data(
    bigquery_client: bigquery.Client,
    dataset_id: str,
    transaction_table_name: str,
    destination_table_name: str,
    query_template_train_file: Optional[str] = None,
    query_template_train_sql: Optional[str] = None,
    write_executed_query_file: Optional[str] = None,
    numerical_features: Optional[FrozenSet[str]] = None,
    non_numerical_features: Optional[FrozenSet[str]] = None,
    days_look_back: int = 365,
    days_look_ahead: int = 365,
    customer_id_column: str = 'customer_id',
    date_column: str = 'date',
    value_column: str = 'value',
    location: str = 'europe-west4',
    numerical_transformations: FrozenSet[str] = _NUMERICAL_TRANSFORMATIONS,
    window_date: Optional[str] = None) -> pd.DataFrame:
  """Builds training data from transaction data through BigQuery.

  This function takes a transaction dataset (a BigQuery table that includes
  information about purchases) and creates a machine learning-ready dataset
  that can be ingested by AutoML. It will first create an SQL query (and
  write it to the file path `write_executed_query_file` for debugging purposes)
  and then execute it. Data types will be automatically detected from the
  BigQuery schema
  if `numerical_features` and `non_numerical_features` are not provided. Columns
  should not be nested.

  Args:
    bigquery_client: BigQuery client.
    dataset_id: The Bigquery dataset_id.
    transaction_table_name: The Bigquery table name with transactions.
    destination_table_name: The Bigquery table name to write to.
    query_template_train_file: File path with the template SQL query. Must be
      provided if query_template_train_sql is not provided.
    query_template_train_sql: SQL with the template query. Must be provided if
      query_template_train_file is not provided.
    write_executed_query_file: File path to write the generated SQL query.
    numerical_features: The names of numerical features to be processed.
    non_numerical_features: The names of non-numerical features to be processed.
      These should be either categorical or text features.
    days_look_back: The number of days to look back to create features.
    days_look_ahead: The number of days to look ahead to predict value.
    customer_id_column: The name of the customer id column.
    date_column: The name of the date column.
    value_column: The name of the value column.
    location: The location to write the table in BigQuery.
    numerical_transformations: The types of transformations for numerical
      features.
    window_date: The date to create 'customer-windows'

  Returns:
    Training data ready for machine learning.
  """
  if not numerical_features and not non_numerical_features:
    numerical_features, non_numerical_features = _detect_feature_types(
        bigquery_client, dataset_id, transaction_table_name,
        ignore_columns=[customer_id_column, date_column])
    logging.info('Detected Numerical Features:')
    for feature in numerical_features:
      logging.info({feature})
    logging.info('Detected Non-Numerical Features:')
    for feature in non_numerical_features:
      logging.info({feature})

  numerical_features_sql = []
  for feature in numerical_features:
    for transformation in numerical_transformations:
      numerical_features_sql.append(
          (f'{transformation}({feature}) as '
           f'{transformation.lower()}_{feature}')
          )
  numerical_features_sql = ', \n'.join(numerical_features_sql)

  non_numerical_features_sql = []
  for feature in non_numerical_features:
    non_numerical_features_sql.append(
        (f'TRIM(STRING_AGG(DISTINCT {feature}, " " '
         f'order by {feature})) AS {feature} \n')
        )
  non_numerical_features_sql = ', \n'.join(non_numerical_features_sql)

  if not window_date:
    window_date = (datetime.date.today() -
                   datetime.timedelta(days=365)).strftime('%Y-%m-%d')

  if query_template_train_file:
    query_template_train = _read_file(query_template_train_file)
  elif query_template_train_sql:
    query_template_train = query_template_train_sql
  else:
    raise ValueError('One of `query_template_train` or `query_template_train`'
                     'must be provided')
  substituted_query = f'{query_template_train}'.format(
      project_id=bigquery_client.project,
      dataset_id=dataset_id,
      table_name=transaction_table_name,
      window_date=window_date,
      customer_id_column=customer_id_column,
      date_column=date_column,
      value_column=value_column,
      days_look_back=days_look_back,
      days_look_ahead=days_look_ahead,
      numerical_features_sql=numerical_features_sql,
      non_numerical_features_sql=non_numerical_features_sql)

  if write_executed_query_file:
    _write_file(substituted_query, write_executed_query_file)
  return _run_query(bigquery_client, substituted_query, dataset_id,
                    destination_table_name, location)

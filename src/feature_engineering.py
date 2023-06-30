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

import collections
import datetime
import logging
from typing import Collection, Dict, List, Mapping, Optional, Tuple

from google.cloud import bigquery
import numpy as np
import pandas as pd


logging.getLogger().setLevel(logging.INFO)

# BigQuery aggregate functions to apply to numerical and boolean columns.
# https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate_functions
_NUMERICAL_TRANSFORMATIONS = frozenset(['AVG', 'MAX', 'MIN', 'SUM'])

# SQL templates library
_QUERY_TEMPLATE_FILES = {
    'train_query': 'src/sql_templates/train_query.sql',
    'predict_query': 'src/sql_templates/predict_query.sql',
}
_STATIC_NUMERIC_FEATURES = (
    'days_since_last_transaction',
    'days_since_first_transaction',
    'count_transactions',
)


def run_load_table_to_bigquery(
    data: pd.DataFrame,
    bigquery_client: bigquery.Client,
    dataset_id: str,
    table_name: str,
    location: str = 'europe-west4',
    write_disposition: str = bigquery.WriteDisposition.WRITE_TRUNCATE,
) -> None:
  """Loads a Pandas Dataframe to Bigquery."""
  table_id = f'{bigquery_client.project}.{dataset_id}.{table_name}'
  job_config = bigquery.job.LoadJobConfig(write_disposition=write_disposition)
  if write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE:
    logging.info('Creating table %r in location %r', table_id, location)
  else:
    logging.info('Appending to table %r in location %r', table_id, location)

  data.columns = data.columns.astype(str)
  bigquery_client.load_table_from_dataframe(
      dataframe=data,
      destination=table_id,
      job_config=job_config,
      location=location,
  ).result()


def run_data_checks(
    bigquery_client: bigquery.Client,
    dataset_id: str,
    table_name: str,
    location: str,
    days_lookback: int = 365,
    days_lookahead: int = 365,
    customer_id_column: str = 'customer_id',
    date_column: str = 'date',
    value_column: str = 'value',
    round_decimal_places: int = 2,
    summary_table_name: str = 'summary_statistics',
) -> pd.DataFrame:
  """Raises exceptions for data issues and outputs summary table to Bigquery.

  Args:
    bigquery_client: BigQuery client.
    dataset_id: The Bigquery dataset_id.
    table_name: The name of the transaction table.
    location: The data location.
    days_lookback: The number of days to look back to create features.
    days_lookahead: The number of days to look ahead to predict value.
    customer_id_column: The name of the customer id column.
    date_column: The name of the date column.
    value_column: The name of the value column.
    round_decimal_places: Then number of decimal places to round to.
    summary_table_name: The name of the summary table to create in Bigquery.

  Raises:
    ValueError: If there are too rows in the dataset.
    ValueError: If there analysis period isn't long enough relative to the
     chosen lookahead and lookback days.

  Returns:
    Summary table.
  """

  query = f"""
  WITH base_table AS (
    SELECT
      {customer_id_column},
      {date_column},
      DATE_ADD(DATE({date_column}), INTERVAL 1 day) AS lookahead_start,
      DATE_ADD(DATE({date_column}), INTERVAL {days_lookahead} day) AS lookahead_stop,
      {value_column}
    FROM
      {bigquery_client.project}.{dataset_id}.{table_name}
  ),

  customer_dates AS (
    SELECT DISTINCT
      {customer_id_column},
      {date_column}
    FROM
      {bigquery_client.project}.{dataset_id}.{table_name}
  )

  SELECT
    base_table.*,
    COUNT(DISTINCT customer_dates.{date_column}) AS future_purchase_dates
  FROM
    base_table
  LEFT JOIN
    customer_dates
  ON (
    base_table.{customer_id_column} = customer_dates.{customer_id_column}
    AND DATE(customer_dates.{date_column}) BETWEEN base_table.lookahead_start AND base_table.lookahead_stop)
  GROUP BY 1,2,3,4,5
  """
  data = bigquery_client.query(query, location=location).result().to_dataframe()

  data[date_column] = pd.to_datetime(data[date_column]).dt.date
  max_date = data[date_column].max()
  min_date = data[date_column].min()

  last_day_with_lookahead = data[date_column].max() - datetime.timedelta(
      days=days_lookahead
  )
  data_with_lookahead = data[data[date_column] <= last_day_with_lookahead]
  lookahead_customer_return_rate = (
      data_with_lookahead['future_purchase_dates'].astype(bool).mean()
  )
  lookahead_customer_mean_returns = data_with_lookahead[
      'future_purchase_dates'
  ].mean()
  lookahead_customer_conditional_mean_returns = data_with_lookahead.loc[
      data_with_lookahead['future_purchase_dates'] > 0, 'future_purchase_dates'
  ].mean()

  summary_data = pd.Series({
      'number_of_rows': len(data),
      'number_of_customers': data[customer_id_column].nunique(),
      'number_of_transactions': len(data[data[value_column] > 0]),
      'total_analysis_days': (max_date - min_date).days,
      'number_of_days_with_data': data[date_column].nunique(),
      'max_transaction_date': max_date.strftime('%Y-%m-%d'),
      'min_transaction_date': min_date.strftime('%Y-%m-%d'),
      'lookahead_customer_return_rate': np.round(
          lookahead_customer_return_rate, round_decimal_places
      ),
      'lookahead_customer_mean_returns': np.round(
          lookahead_customer_mean_returns, round_decimal_places
      ),
      'lookahead_customer_conditional_mean_returns': np.round(
          lookahead_customer_conditional_mean_returns, round_decimal_places
      ),
  })

  value_summary = data[value_column].describe()[1:].round(round_decimal_places)
  value_summary.index = [
      f'{value_column}_{statistic}'
      for statistic in value_summary.index.str.replace('%', '_quantile')
  ]

  transactions_summary = (
      data.groupby(customer_id_column)
      .size()
      .describe()[1:]
      .round(round_decimal_places)
  )
  transactions_summary.index = [
      f'transactions_per_customer_{statistic}'
      for statistic in transactions_summary.index.str.replace('%', '_quantile')
  ]

  summary_data = pd.concat([summary_data, value_summary, transactions_summary])

  if summary_data['number_of_rows'] < 1000:
    raise ValueError(
        f'{summary_data["number_of_rows"]} is too few data points to '
        'build an LTV model.'
    )

  if summary_data['total_analysis_days'] < days_lookback + days_lookahead:
    raise ValueError(
        f'Insufficient analysis days ({summary_data["total_analysis_days"]})'
        f'for selected lookahead ({days_lookahead} days) and '
        f'lookback ({days_lookback} days) windows'
    )

  summary_data = summary_data.to_frame('statistics').T
  run_load_table_to_bigquery(
      summary_data,
      bigquery_client,
      dataset_id,
      table_name=summary_table_name,
      location=location,
  )
  return summary_data


def detect_column_types(
    bigquery_client: bigquery.Client,
    dataset_id: str,
    table_name: str,
    ignore_columns: List[str],
) -> Dict[str, List[str]]:
  """Detects the column types using the schema in BigQuery.

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
      dataset_id, project=bigquery_client.project
  )
  table_reference = dataset_reference.table(table_name)
  table = bigquery_client.get_table(table_reference)
  logging.info(
      'Detecting features types in project_id %r in dataset %r in table %r',
      bigquery_client.project,
      dataset_id,
      table_name,
  )

  input_data_types = collections.defaultdict(list)
  for feature in table.schema:
    if feature.name not in ignore_columns:
      if feature.field_type == 'BOOLEAN':
        input_data_types['boolean'].append(feature.name)
      elif feature.field_type in ['INTEGER', 'FLOAT', 'NUMERIC']:
        input_data_types['numeric'].append(feature.name)
      elif feature.field_type == 'STRING':
        input_data_types['string_or_categorical'].append(feature.name)
  return input_data_types


def _read_file(file_name: str) -> str:
  """Reads file."""
  with open(file_name, 'r') as f:
    return f.read()


def _write_file(query: str, file_name: str) -> None:
  """Writes query to file."""
  with open(file_name, 'w') as f:
    f.write(query)


def build_query(
    bigquery_client: bigquery.Client,
    dataset_id: str,
    transaction_table_name: str,
    input_data_types: Mapping[str, List[str]],
    query_type: str = 'train_query',
    numerical_transformations: Collection[str] = _NUMERICAL_TRANSFORMATIONS,
    write_executed_query_file: Optional[str] = None,
    days_lookback: int = 365,
    days_lookahead: int = 365,
    customer_id_column: str = 'customer_id',
    date_column: str = 'date',
    value_column: str = 'value',
) -> Tuple[str, Mapping[str, List[str]]]:
  """Builds training or prediction query from transaction data through BigQuery.

  This function takes a transaction dataset (a BigQuery table that includes
  information about purchases) and creates an SQL query to generate a machine
  learning-ready dataset that can be ingested by AutoML either for training
  or prediction (determined by the parameter `query_type`. The SQL query can be
  written to the file path `write_executed_query_file` for manual modifications.
  Data types will be automatically detected from the BigQuery schema if
  `features_types` argument is not provided.

  Args:
    bigquery_client: BigQuery client.
    dataset_id: The Bigquery dataset_id.
    transaction_table_name: The Bigquery table name with transactions.
    input_data_types: The mapping of feature types to feature names.
    query_type: The query type. Has to be one of the keys in
      _QUERY_TEMPLATE_FILES.
    numerical_transformations: The types of transformations for numerical
      features.
    write_executed_query_file: File path to write the generated SQL query.
    days_lookback: The number of days to look back to create features.
    days_lookahead: The number of days to look ahead to predict value.
    customer_id_column: The name of the customer id column.
    date_column: The name of the date column.
    value_column: The name of the value column.

  Returns:
    The SQL script to generate training data ready for machine learning along
    side with a mapping of the column type and list of columns of that type
    thave have been created in the training table.
  """
  if query_type not in _QUERY_TEMPLATE_FILES:
    raise ValueError(f'{query_type} not one of {_QUERY_TEMPLATE_FILES.keys()}')
  features_types = collections.defaultdict(list)
  features_list = []
  if 'numeric' in input_data_types:
    for feature in input_data_types['numeric']:
      for transformation in numerical_transformations:
        feature_name = f'{transformation.lower()}_{feature}'
        features_list.append(f'{transformation}({feature}) AS {feature_name}')
        features_types['numeric'].append(feature_name)

  if 'boolean' in input_data_types:
    for feature in input_data_types['boolean']:
      for transformation in numerical_transformations:
        feature_name = f'{transformation.lower()}_{feature}'
        features_list.append(
            f'{transformation}(CAST({feature} AS INT)) AS {feature_name}'
        )
        features_types['boolean'].append(feature_name)

  if 'string_or_categorical' in input_data_types:
    for feature in input_data_types['string_or_categorical']:
      features_list.append(
          f'TRIM(STRING_AGG(DISTINCT {feature}, " " ORDER BY '
          f'{feature})) AS {feature}'
      )
      features_types['string_or_categorical'].append(feature)

  features_types['numeric'].extend(list(_STATIC_NUMERIC_FEATURES))
  query_template = _read_file(_QUERY_TEMPLATE_FILES[query_type])

  substituted_query = query_template.format(
      project_id=bigquery_client.project,
      dataset_id=dataset_id,
      table_name=transaction_table_name,
      customer_id_column=customer_id_column,
      date_column=date_column,
      value_column=value_column,
      days_lookback=days_lookback,
      days_lookahead=days_lookahead,
      features_sql=', \n'.join(features_list),
  )

  if write_executed_query_file:
    _write_file(substituted_query, write_executed_query_file)
    logging.info(
        'Query successfully written to: "%r"', write_executed_query_file
    )
  return substituted_query, features_types


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
      write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
  )
  data = (
      bigquery_client.query(query, job_config=job_config, location=location)
      .result()
      .to_dataframe()
  )
  logging.info('Created table %r in location %r', table_id, location)
  return data

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

"""Module for creating synthetic data for testing crystal value pipeline."""

import logging

from typing import Optional

from google.cloud import bigquery
import numpy as np
import pandas as pd


logging.getLogger().setLevel(logging.INFO)

_CUSTOMER_SEARCHES = frozenset([
    'cool', 'car', 'amazing', 'beach', 'blue', 'car', 'drive', 'to', 'the',
    'seaside', 'shiny'
])


def _load_table_to_bigquery(
    data: pd.DataFrame,
    bigquery_client: bigquery.Client,
    dataset_id: str,
    table_name: str,
    bigquery_location: str = 'EU') -> None:
  """Loads a Pandas Dataframe to Bigquery."""
  table_id = f'{bigquery_client.project}.{dataset_id}.{table_name}'
  job_config = bigquery.job.LoadJobConfig(
      write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
  bigquery_client.load_table_from_dataframe(
      dataframe=data,
      destination=table_id,
      job_config=job_config,
      location=bigquery_location).result()
  logging.info('Created table %r', table_id)


def create_synthetic_data(
    bigquery_client: Optional[bigquery.Client] = None,
    dataset_id: Optional[str] = None,
    table_name: Optional[str] = 'synthetic_data',
    row_count: int = 100000,
    start_date: str = '2018-01-01',
    end_date: str = '2021-01-01',
    load_table_to_bigquery: bool = False,
    bigquery_location: str = 'EU') -> pd.DataFrame:
  """Creates a synthetic transaction dataset with an option to load to Bigquery.

  The transaction dataset contains customer ids which can make multiple
  transactions. There is also additional information from the transaction
  including a numerical, categorical and a text column.

  Args:
    bigquery_client: The Bigquery client.
    dataset_id: The Bigquery dataset_id within the project_id to load data to.
    table_name: The Bigquery table name to load data to.
    row_count: The number of rows in the dataset to generate.
    start_date: The start date of the transactions.
    end_date: The end date of the transactions.
    load_table_to_bigquery: Whether to load the data to Bigquery.
    bigquery_location: The location of the dataset to load in Bigquery.

  Returns:
    The created dataset.
  """

  date_list = pd.date_range(
      start=start_date,
      end=end_date,
      freq='1D').strftime('%Y-%m-%d').tolist()
  date_list = date_list * int((row_count / 100))

  data = pd.DataFrame({
      'customer_id': np.random.poisson(row_count / 3, size=row_count),
      'date': date_list[:row_count],
      'numeric_column': np.random.exponential(5, size=row_count),
      'categorical_column': np.random.poisson(3, size=row_count),
      'text_column': [' '.join(np.random.choice(list(_CUSTOMER_SEARCHES), 3))
                      for i in range(row_count)]
  }).round(3)
  data['value'] = data['numeric_column'] * data['categorical_column']
  data['categorical_column'] = [f'category_ {str(i)}'
                                for i in data['categorical_column']]

  if load_table_to_bigquery:
    _load_table_to_bigquery(
        data, bigquery_client, dataset_id, table_name, bigquery_location)

  return data

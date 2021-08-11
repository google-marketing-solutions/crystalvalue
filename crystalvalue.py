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
"""Main module to train and predict a CrystalValue LTV model.

The CrystalValue LTV model uses Vertex AI AutoML Tables.


Minimal example:

from google.cloud import bigquery

import crystalvalue


# Create BigQuery client for cloud authentication.
bigquery_client = bigquery.Client()

# Initiate the CrystalValue class.
# The Google Cloud Platform project will be identified using Bigquery client.
pipeline = crystalvalue.CrystalValue(
    bigquery_client=bigquery_client,
    dataset_id='an_existing_dataset')

# (Optional) If you are just testing out CrystalValue, use this method to
# create a synthetic transaction dataset and load it to BigQuery.
data = pipeline.create_synthetic_data(table_name='synthetic_data')

# (Optional) Run automated data checks on your input data to ensure
# that you have sufficient data. This also outputs a summary table in your
# Bigquery dataset which can be useful to spot outliers.
summary_statistics = pipeline.run_data_checks(
    transaction_table_name='synthetic_data')

# Perform feature engineering using BigQuery.
# CrystalValue automatically detects data types and applies transformations.
# CrystalValue by default will predict 1 year ahead using data
# accumulated from 1 year before (configurable).
training_data_query = pipeline.feature_engineer(
    transaction_table_name='synthetic_data')

# Creates AI Platform Dataset and trains AutoML model.
pipeline.train()

# Check out your trained AutoML model in the Google Cloud Platform UI!
# The feature importance graphs and statistics can be viewed in the UI.
# Statistics on your features can also be viewed here.
# https://console.cloud.google.com//vertex-ai/models

# Now create LTV predictions using the model and input data.
pipeline.predict(
  input_table_name='predict_data',
  model_resource_name='5537281294968291328')

"""

import dataclasses
from typing import Collection, List, Mapping, Optional

from absl import logging
from google.cloud import bigquery
import pandas as pd

from crystalvalue import automl
from crystalvalue import feature_engineering
from crystalvalue import synthetic_data


@dataclasses.dataclass
class CrystalValue:
  """Class to train and predict LTV model.

  Attributes:
    bigquery_client: BigQuery client.
    dataset_id: The Bigquery dataset_id.
    training_table_name: The name of the training table to be created.
    customer_id_column: The name of the customer id column.
    date_column: The name of the date column.
    value_column: The name of the future value column.
    features_types: A mapping of the feature type to a list of columns of such
      type. For example:
      {'numeric': ['transaction_value', 'number_products'],
      'boolean': ['is_registered_customer'],
      'string_or_categorical': ['transaction_type', 'payment_method']}
    ignore_columns: Columns to ignore from the original dataset. For example:
      ['webpage_id_column'].
    location: The Bigquery and Vertex AI location for processing (e.g.
      'europe-west4' or 'us-east-4')
    window_date: The date to create 'customer-windows'. CrystalValue will train
      a model using data from 1 year before this date to predict value from 1
      year after this date. If `None` (default), then CrystalValue will set the
      window_date to 365 days ago.
    days_lookback: The number of days to look back to create features.
    days_lookahead: The number of days to look ahead to predict value.
  """
  bigquery_client: bigquery.Client
  dataset_id: str
  training_table_name: str = 'training_data'
  customer_id_column: str = 'customer_id'
  date_column: str = 'date'
  value_column: str = 'value'
  features_types: Optional[Mapping[str, List[str]]] = None
  ignore_columns: Optional[Collection[str]] = None
  location: str = 'europe-west4'
  window_date: Optional[str] = None
  days_lookback: int = 365
  days_lookahead: int = 365

  def create_synthetic_data(self,
                            table_name: str = 'synthetic_data',
                            row_count: int = 100000,
                            start_date: str = '2018-01-01',
                            end_date: str = '2021-01-01') -> pd.DataFrame:
    """Creates a synthetic transaction dataset and loads to Bigquery.

    The transaction dataset contains customer ids which can make multiple
    transactions. There is also additional information from the transaction
    including a numerical, categorical and a text column.

    Args:
      table_name: The Bigquery table name to load data to.
      row_count: The number of rows in the dataset to generate.
      start_date: The start date of the transactions.
      end_date: The end date of the transactions.

    Returns:
      The created dataset.
    """
    return synthetic_data.create_synthetic_data(
        bigquery_client=self.bigquery_client,
        dataset_id=self.dataset_id,
        table_name=table_name,
        row_count=row_count,
        start_date=start_date,
        end_date=end_date,
        load_table_to_bigquery=True,
        location=self.location)

  def run_data_checks(self,
                      transaction_table_name: str,
                      round_decimal_places: int = 2,
                      summary_table_name: str = 'summary_statistics'
                      ) -> pd.DataFrame:
    """Runs data checks on transaction data.

    Args:
      transaction_table_name: The name of the table with the data.
      round_decimal_places: The number of decimal places to round to.
      summary_table_name: The name of the statistics table to output.

    Returns:
      Summary table.
    """
    return feature_engineering.run_data_checks(
        bigquery_client=self.bigquery_client,
        dataset_id=self.dataset_id,
        table_name=transaction_table_name,
        summary_table_name=summary_table_name,
        days_lookback=self.days_lookback,
        days_lookahead=self.days_lookahead,
        customer_id_column=self.customer_id_column,
        date_column=self.date_column,
        value_column=self.value_column,
        round_decimal_places=round_decimal_places,
        location=self.location)

  def feature_engineer(
      self,
      transaction_table_name: str,
      write_executed_query_file: Optional[str] = None) -> pd.DataFrame:
    """Builds training data from transaction data through BigQuery.

    This function takes a transaction dataset (a BigQuery table that includes
    information about purchases) and creates a script to generate a machine
    learning-ready dataset that can be ingested by AutoML.The SQL query can be
    written to the file path `write_executed_query_file` for manual
    modifications. Data types will be automatically detected from the BigQuery
    schema if `numerical_features` and `string_or_categorical_features` are not
    provided. By default, the model will use features from between 2 and 1 years
    ago to predict value from between 1 year ago and now. This is configurable
    using the `window_date` class attribute and the days_look_back and
    days_look_ahead method arguments.

    Args:
      transaction_table_name: The Bigquery table name with transactions.
      write_executed_query_file: File path to write the generated SQL query.

    Returns:
      The SQL script to generate training data ready for machine learning.
    """
    if not self.features_types:
      if not self.ignore_columns:
        self.ignore_columns = [self.customer_id_column, self.date_column]
      else:
        self.ignore_columns = [self.customer_id_column, self.date_column
                              ] + self.ignore_columns
      self.features_types = feature_engineering.detect_feature_types(
          bigquery_client=self.bigquery_client,
          dataset_id=self.dataset_id,
          table_name=transaction_table_name,
          ignore_columns=self.ignore_columns)
      if not self.features_types:
        raise ValueError('No features detected')
      for feature_type in self.features_types:
        for feature in self.features_types[feature_type]:
          logging.info('Detected %r feature %r', feature_type, feature)

    query = feature_engineering.build_train_query(
        bigquery_client=self.bigquery_client,
        dataset_id=self.dataset_id,
        transaction_table_name=transaction_table_name,
        features_types=self.features_types,
        write_executed_query_file=write_executed_query_file,
        days_lookback=self.days_lookback,
        days_lookahead=self.days_lookahead,
        customer_id_column=self.customer_id_column,
        date_column=self.date_column,
        value_column=self.value_column,
        window_date=self.window_date)

    return self.run_query(query_sql=query)

  def run_query(self,
                query_sql: Optional[str] = None,
                query_file: Optional[str] = None) -> pd.DataFrame:
    """Runs a query in Bigquery either using a file or a query string.

    One of query_sql or query_file must be provided.

    Args:
      query_sql: The SQL query to execute.
      query_file: Path to the SQL query to execute.

    Returns:
      Training data ready for machine learning.
    """
    return feature_engineering.run_query(
        bigquery_client=self.bigquery_client,
        query_sql=query_sql,
        query_file=query_file,
        dataset_id=self.dataset_id,
        destination_table_name=self.training_table_name,
        location=self.location)

  def train(self,
            dataset_display_name: str = 'crystalvalue_dataset',
            model_display_name: str = 'crystalvalue_model',
            predefined_split_column_name: str = 'predefined_split_column',
            target_column: str = 'future_value',
            optimization_objective: str = 'minimize-rmse',
            budget_milli_node_hours: int = 1000) -> None:
    """Creates Vertex AI Dataset and trains an AutoML Tabular model.

    An AutoML Dataset is required before training a model. See
    https://cloud.google.com/vertex-ai/docs/datasets/create-dataset-api
    https://cloud.google.com/vertex-ai/docs/training/automl-api

    Args:
      dataset_display_name: The display name of the Dataset to create.
      model_display_name: The display name of the Model to create.
      predefined_split_column_name: A name of one of the Dataset's columns. The
        values of the column must be one of {``training``, ``validation``,
        ``test``}, and it defines to which set the given piece of data is
        assigned. If for a piece of data the key is not present or has an
        invalid value, that piece is ignored by the pipeline.
      target_column: The target to predict.
      optimization_objective: Objective function the Model is to be optimized
        towards. The training task creates a Model that maximizes/minimizes the
        value of the objective function over the validation set. "minimize-rmse"
        (default) - Minimize root-mean-squared error (RMSE). "minimize-mae" -
        Minimize mean-absolute error (MAE). "minimize-rmsle" - Minimize
        root-mean-squared log error (RMSLE).
      budget_milli_node_hours: The number of node hours to use to train the
        model (times 1000), 1000 milli node hours is 1 mode hour.
    """

    aiplatform_dataset = automl.create_automl_dataset(
        project_id=self.bigquery_client.project,
        dataset_id=self.dataset_id,
        table_name=self.training_table_name,
        dataset_display_name=dataset_display_name,
        location=self.location)

    automl.train_automl_model(
        project_id=self.bigquery_client.project,
        aiplatform_dataset=aiplatform_dataset,
        model_display_name=model_display_name,
        predefined_split_column_name=predefined_split_column_name,
        target_column=target_column,
        optimization_objective=optimization_objective,
        budget_milli_node_hours=budget_milli_node_hours,
        location=self.location)

  def predict(self,
              input_table_name: str,
              model_resource_name: str,
              model_name: str = 'crystalvalue_model',
              destination_table: str = 'predictions'):
    """Creates predictions using Vertex AI model into destination table.

    Args:
      input_table_name: The table containing features to predict with.
      model_resource_name: The resource name of the Vertex AI model e.g.
        '553728129496821'
      model_name: The name of the Vertex AI trained model e.g.
        'crystalvalue_model'.
      destination_table: The table to either create (if it doesn't exist) or
        append predictions to within your dataset.
    """

    batch_predictions = automl.create_batch_predictions(
        project_id=self.bigquery_client.project,
        dataset_id=self.dataset_id,
        model_resource_name=model_resource_name,
        table_name=input_table_name,
        location=self.location)

    automl.load_predictions_to_table(
        bigquery_client=self.bigquery_client,
        dataset_id=self.dataset_id,
        batch_predictions=batch_predictions,
        location=self.location,
        destination_table=destination_table,
        model_name=model_name)

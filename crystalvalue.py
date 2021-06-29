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

The CrystalValue LTV model uses AutoML Tables through the AI Platform (Unified).


Example usage:

# Import libraries
from google.cloud import bigquery
from google.cloud.aiplatform import aiplatform

import crystalvalue


# Create clients for cloud authentication.
bigquery_client = bigquery.Client()
aiplatform_dataservice_client = aiplatform.gapic.DatasetServiceClient()
aiplatform_pipeline_client = aiplatform.gapic.PipelineServiceClient()


# Initiate the CrystalValue class.
crystal_value = crystalvalue.CrystalValue(
    bigquery_client=bigquery_client,
    aiplatform_dataservice_client=aiplatform_dataservice_client,
    aiplatform_pipeline_client=aiplatform_pipeline_client,
    dataset_id='ltv_dataset')


# (Optional) If you are just testing out CrystalValue, use this method to
# create a synthetic transaction dataset and load it to BigQuery.
data = crystal_value.create_synthetic_data(table_name='synthetic_data')


# Perform feature engineering using BigQuery.
# CrystalValue automatically detects data types and applies transformations.
# CrystalValue by default will predict 1 year ahead (configurable)
training_data = crystal_value.feature_engineer(
    transaction_table_name='synthetic_data',
    query_template_train_file='./sql_templates/train_user_split.sql')

# Creates AI Platform Dataset and trains AutoML model.
crystal_value.train(training_budget_milli_node_hours=1000)

# Check out your trained AutoML model in the Google Cloud Platform UI!
# https://console.cloud.google.com/ai-platform/models
"""

from typing import FrozenSet, Optional

import dataclasses
from google.cloud import aiplatform
from google.cloud import bigquery
import pandas as pd

from solutions.crystalvalue import automl
from solutions.crystalvalue import feature_engineering
from solutions.crystalvalue import synthetic_data


@dataclasses.dataclass
class CrystalValue:
  """Class to train and predict LTV model.

  Attributes:
    bigquery_client: BigQuery client.
    aiplatform_dataservice_client: AI Platform client for creating dataset.
    aiplatform_pipeline_client: AI Platform client for training pipeline.
    dataset_id: The Bigquery dataset_id.
    training_dataset_name: The name of the training data table to be created.
    customer_id_column: The name of the customer id column.
    date_column: The name of the date column.
    value_column: The name of the future value column.
    numerical_features: The names of numerical features to be processed.
    non_numerical_features: The names of non-numerical features to be processed.
      These should be either categorical or text features.
    bigquery_location: The location to write the table in BigQuery.
    aiplatform_location: The location for the AI Platforms dataset and model.
    window_date: The date to create 'customer-windows'
  """
  bigquery_client: bigquery.Client
  aiplatform_dataservice_client: aiplatform.gapic.DatasetServiceClient
  aiplatform_pipeline_client: aiplatform.gapic.PipelineServiceClient
  dataset_id: str
  training_dataset_name: str = 'training_data'
  customer_id_column: str = 'customer_id'
  date_column: str = 'date'
  value_column: str = 'value'
  numerical_features: Optional[FrozenSet[str]] = None
  non_numerical_features: Optional[FrozenSet[str]] = None
  bigquery_location: str = 'EU'
  aiplatform_location: str = 'europe-west4'
  window_date: Optional[str] = None

  def create_synthetic_data(
      self,
      table_name: str = 'synthetic_data',
      row_count: int = 50000,
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
        bigquery_location=self.bigquery_location)

  def feature_engineer(
      self,
      transaction_table_name: str,
      query_template_train_file: Optional[str] = None,
      query_template_train_sql: Optional[str] = None,
      write_executed_query_file: Optional[str] = None,
      days_look_back: int = 365,
      days_look_ahead: int = 365) -> pd.DataFrame:
    """Builds training data from transaction data through BigQuery.

    This function takes a transaction dataset (a BigQuery table that includes
    information about purchases) and creates a machine learning-ready dataset
    that can be ingested by AutoML. It will first create an SQL query (and
    write it to the file path `write_executed_query_file` for debugging
    purposes) and then execute it. Data types will be automatically detected
    from the BigQuery schema if `numerical_features` and
    `non_numerical_features` are not provided. Columns should not be nested.
    By default, the model will use features from between 2 and 1 years ago to
    predict value from between 1 year ago and now. This is configurable using
    the `window_date` class attribute and the days_look_back and days_look_ahead
    method arguments.

    Args:
      transaction_table_name: The Bigquery table name with transactions.
      query_template_train_file: File path with the template SQL query. Must be
        provided if query_template_train_sql is not provided.
      query_template_train_sql: SQL with the template query. Must be provided if
        query_template_train_file is not provided.
      write_executed_query_file: File path to write the generated SQL query.
      days_look_back: The number of days to look back to create features.
      days_look_ahead: The number of days to look ahead to predict value.

    Returns:
      Training data ready for machine learning.
    """
    return feature_engineering.build_train_data(
        bigquery_client=self.bigquery_client,
        dataset_id=self.dataset_id,
        transaction_table_name=transaction_table_name,
        destination_table_name=self.training_dataset_name,
        query_template_train_file=query_template_train_file,
        query_template_train_sql=query_template_train_sql,
        write_executed_query_file=write_executed_query_file,
        numerical_features=self.numerical_features,
        non_numerical_features=self.non_numerical_features,
        days_look_back=days_look_back,
        days_look_ahead=days_look_ahead,
        customer_id_column=self.customer_id_column,
        date_column=self.date_column,
        value_column=self.value_column,
        bigquery_location=self.bigquery_location,
        window_date=self.window_date)

  def train(
      self,
      training_budget_milli_node_hours: int = 1000,
      timeout: int = 300) -> None:
    """Runs feature engineering and AutoML training.

    Args:
      training_budget_milli_node_hours: The number of node hours to use to train
        the model (times 1000), 1000 milli node hours is 1 mode hour.
      timeout: The timeout in seconds for creating the Dataset.
    """
    automl.create_automl_dataset(
        aiplatform_client=self.aiplatform_dataservice_client,
        project_id=self.bigquery_client.project,
        dataset_id=self.dataset_id,
        table_name=self.training_dataset_name,
        dataset_display_name=self.training_dataset_name,
        aiplatform_location=self.aiplatform_location,
        timeout=timeout)

    features = list(self.numerical_features) + list(self.non_numerical_features)
    automl.train_automl_model(
        aiplatform_client=self.aiplatform_pipeline_client,
        project_id=self.bigquery_client.project,
        dataset_id=self.dataset_id,
        dataset_display_name=self.training_dataset_name,
        model_display_name=self.training_dataset_name,
        features=features,
        target_column=self.value_column,
        training_budget_milli_node_hours=training_budget_milli_node_hours,
        aiplatform_location=self.aiplatform_location)

  # TODO() Create AI Platform module for predicting through AutoML
  def predict(self):
    ...

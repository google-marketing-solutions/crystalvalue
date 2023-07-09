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

"""Module for training and predicting using AutoML on AI Platform (Unified).

Example use:

project_id = 'my_project'
dataset_id = 'dataset_ltv'
table_name = 'training_data'
target_column = 'future_value'

aiplatform_dataset = automl.create_automl_dataset(
    project_id=project_id,
    dataset_id=dataset_id,
    table_name=table_name)

automl.train_automl_model(
    project_id=project_id,
    aiplatform_dataset=aiplatform_dataset,
    target_column=target_column)
"""

import logging
import re
from typing import List

from google.cloud import bigquery
from google.cloud import aiplatform
import pandas as pd


logging.getLogger().setLevel(logging.INFO)

_NON_FEATURES = [
    'customer_id',
    'window_date',
    'lookback_start',
    'lookahead_start',
    'lookahead_stop',
    'future_value',
    'predefined_split_column',
]


def create_automl_dataset(
    project_id: str,
    dataset_id: str,
    table_name: str = 'training_data',
    dataset_display_name: str = 'crystalvalue_dataset',
    location: str = 'europe-west4',
) -> aiplatform.datasets.tabular_dataset.TabularDataset:
  """Creates AutoML Dataset in the AI Platform.

  An AutoML Dataset is required before training a model. See
  https://cloud.google.com/vertex-ai/docs/datasets/create-dataset-api

  Args:
    project_id: The Bigquery project_id.
    dataset_id: The Bigquery dataset_id.
    table_name: The Bigquery training dataset name to use for AutoML.
    dataset_display_name: The display name of the AutoML Dataset to be created.
    location: The location of the AutoML Dataset to be created.

  Returns:
    The AI Platform AutoML dataset.
  """
  logging.info(
      'Creating Vertex AI Dataset with display name %r', dataset_display_name
  )
  bigquery_uri = f'bq://{project_id}.{dataset_id}.{table_name}'

  aiplatform.init(project=project_id, location=location)
  dataset = aiplatform.TabularDataset.create(
      display_name=dataset_display_name, bq_source=bigquery_uri
  )

  dataset.wait()
  return dataset


def train_automl_model(
    project_id: str,
    aiplatform_dataset: aiplatform.TabularDataset,
    model_display_name: str = 'crystalvalue_model',
    predefined_split_column_name: str = 'predefined_split_column',
    target_column: str = 'future_value',
    optimization_objective: str = 'minimize-rmse',
    optimization_prediction_type: str = 'regression',
    budget_milli_node_hours: int = 1000,
    location: str = 'europe-west4',
) -> aiplatform.models.Model:
  """Trains an AutoML model given an AutoML Dataset.

  See:
  https://cloud.google.com/vertex-ai/docs/training/automl-api

  Args:
    project_id: The Bigquery project_id.
    aiplatform_dataset: The dataset in the AI Platform used for AutoML.
    model_display_name: The name of the AutoML model to create.
    predefined_split_column_name: The key is a name of one of the Dataset's data
      columns. The value of the key (either the label's value or value in the
      column) must be one of {``training``, ``validation``, ``test``}, and it
      defines to which set the given piece of data is assigned. If for a piece
      of data the key is not present or has an invalid value, that piece is
      ignored by the pipeline.
    target_column: The target to predict.
    optimization_objective: Objective function the Model is to be optimized
      towards. The training task creates a Model that maximizes/minimizes the
      value of the objective function over the validation set. "minimize-rmse"
      (default) - Minimize root-mean-squared error (RMSE). "minimize-mae" -
      Minimize mean-absolute error (MAE). "minimize-rmsle" - Minimize
      root-mean-squared log error (RMSLE). for classification use:
      "maximize-au-prc" - Maximize precision-recall area under curve.
      "maximize-au-roc" - Maximize ROC area under curve. "minimize-log-loss"
      Minimize log-loss.
    optimization_prediction_type: Prediction type for training. The possible
      training tasks are: "regression" (default) "classification"
    budget_milli_node_hours: The number of node hours to use to train the model
      (times 1000), 1000 milli node hours is 1 mode hour.
    location: The location to train the AutoML model.

  Returns:
    Vertex AI AutoML model.
  """
  logging.info(
      'Creating Vertex AI AutoML model with display name %r', model_display_name
  )

  transformations = [
      {'auto': {'column_name': f'{feature}'}}
      for feature in aiplatform_dataset.column_names
      if feature not in _NON_FEATURES
  ]

  aiplatform.init(project=project_id, location=location)
  job = aiplatform.AutoMLTabularTrainingJob(
      display_name=model_display_name,
      optimization_prediction_type=optimization_prediction_type,
      optimization_objective=optimization_objective,
      column_transformations=transformations,
  )

  model = job.run(
      dataset=aiplatform_dataset,
      target_column=target_column,
      budget_milli_node_hours=budget_milli_node_hours,
      model_display_name=model_display_name,
      predefined_split_column_name=predefined_split_column_name,
  )

  model.wait()
  logging.info(
      'Created AI Platform Model with display name %r', model.display_name
  )
  return model


def create_batch_predictions(
    project_id: str,
    dataset_id: str,
    table_name: str,
    model_id: str,
    location: str,
) -> aiplatform.BatchPredictionJob:
  """Creates batch prediction job.

  Args:
    project_id: The Bigquery project_id.
    dataset_id: The Bigquery dataset_id containing the data to create
      predictions with.
    table_name: The table name containing the data to create predictions with.
    model_id: The name of the Vertex AI AutoML model.
    location: The location of the Vertex AI AutoML model.

  Returns:
    Vertex AI batch prediction object.
  """
  bigquery_uri = f'bq://{project_id}.{dataset_id}.{table_name}'

  batch_prediction = aiplatform.BatchPredictionJob.create(
      job_display_name='crystalvalue_job',
      model_name=model_id,
      bigquery_source=bigquery_uri,
      bigquery_destination_prefix=project_id,
      location=location,
  )

  return batch_prediction


def load_predictions_to_table(
    bigquery_client: bigquery.Client,
    dataset_id: str,
    batch_predictions: aiplatform.BatchPredictionJob,
    location: str = 'europe-west4',
    destination_table: str = 'predictions',
    model_name: str = 'crystalvalue_model',
) -> None:
  """Extracts data from Vertex AI prediction dataset and into workspace.

  The Vertex AI AutoML automatically puts predictions in a new dataset. This
  function takes those predictions and puts them into the workspace dataset.

  If any predictions are negative they will be set to 0.

  Args:
    bigquery_client: BigQuery client.
    dataset_id: The Bigquery dataset_id.
    batch_predictions: Vertex AI batch predictions object.
    location: Data processing location.
    destination_table: Table to load predictions to within the dataset_id.
    model_name: The name of the trained model.
  """
  output_dataset = re.findall(
      'bq://(.*?)"', str(batch_predictions.output_info)
  )[0]
  timestamp = output_dataset.split(model_name)[1]
  output_table = output_dataset + f'.predictions{timestamp}'

  prediction_table = (
      f'{bigquery_client.project}.{dataset_id}.{destination_table}'
  )

  tables = [table.table_id for table in bigquery_client.list_tables(dataset_id)]
  if prediction_table.split('.')[-1] in tables:
    bigquery_option = f'INSERT INTO {prediction_table}'
  else:
    bigquery_option = f'CREATE TABLE {prediction_table} AS'

  query = f"""
  {bigquery_option}
  SELECT
    customer_id,
    lookahead_start,
    lookahead_stop,
    IF(predicted_future_value.value < 0,
       0,
       predicted_future_value.value) AS future_value_predicted
  FROM `{output_table}`
  """
  bigquery_client.query(query, location=location).result()


def deploy_model(
    bigquery_client: bigquery.Client,
    model_id: str,
    machine_type: str = 'n1-standard-2',
    location: str = 'europe-west4',
) -> aiplatform.Model:
  """Creates an endpoint and deploys Vertex AI Tabular AutoML model.

  Args:
    bigquery_client: BigQuery client.
    model_id: The ID of the model e.g. '6314539258782679041'.
    machine_type: The machine type to deploy to.
    location: The location of the model.

  Returns:
    Deployed model object.
  """
  aiplatform.init(project=bigquery_client.project, location=location)
  model = aiplatform.Model(model_name=model_id)
  model.deploy(machine_type=machine_type)
  model.wait()
  logging.info('Deployed model with display name %r', model.display_name)
  return model


def predict_using_deployed_model(
    project_id: str,
    endpoint: str,
    features: pd.DataFrame,
    location: str = 'europe-west4',
) -> List[float]:
  """Create predictions using a deployed model from a Pandas DataFrame.

  Args:
    project_id: The Google project ID.
    endpoint: The ID of the endpoint e.g. '4749679125759787009'.
    features: The features to be used to create predictions.
    location: The location of the model.

  Returns:
    Prediction values.
  """

  aiplatform.init(project=project_id, location=location)
  endpoint = aiplatform.Endpoint(endpoint)

  # Ensure objects such as dates are strings and not datetime.
  string_columns = features.select_dtypes([object, 'dbdate']).columns.to_list()
  features[string_columns] = features[string_columns].astype(str)

  # There seems to be a bug for recognising integers
  # so we have to convert integer columns to strings.
  integers = ['int16', 'int32', 'int64']
  integer_columns = features.select_dtypes(include=integers).columns.to_list()
  features[integer_columns] = features[integer_columns].astype(str)

  # Gets around the 100 record prediction limit.
  predictions = []
  for i in range(0, len(features), 100):
    response = endpoint.predict(
        instances=features[i : (i + 100)].to_dict('records')
    )
    if isinstance(response.predictions, list) and isinstance(
        response.predictions[0], float
    ):
      predictions.extend(response.predictions)
    else:
      predictions.extend([record['value'] for record in response.predictions])

  return predictions

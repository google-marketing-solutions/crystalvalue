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

from google.cloud import aiplatform


logging.getLogger().setLevel(logging.INFO)

_NON_FEATURES = [
    'customer_id', 'window_date', 'lookback_start', 'lookahead_start',
    'lookahead_stop', 'future_value', 'predefined_split_column'
]


def create_automl_dataset(
    project_id: str,
    dataset_id: str,
    table_name: str = 'training_data',
    dataset_display_name: str = 'crystalvalue_dataset',
    location: str = 'europe-west4'
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
  bigquery_uri = f'bq://{project_id}.{dataset_id}.{table_name}'

  aiplatform.init(project=project_id, location=location)
  dataset = aiplatform.TabularDataset.create(
      display_name=dataset_display_name, bq_source=bigquery_uri)

  dataset.wait()
  logging.info('Created AI Platform Dataset with display name %r',
               dataset_display_name)
  return dataset


def train_automl_model(
    project_id: str,
    aiplatform_dataset: str,
    model_display_name: str = 'crystalvalue_model',
    predefined_split_column_name: str = 'predefined_split_column',
    target_column: str = 'future_value',
    optimization_objective: str = 'minimize-rmse',
    budget_milli_node_hours: int = 1000,
    location: str = 'europe-west4'
    ) -> aiplatform.models.Model:
  """Trains an AutoML model given an AutoML Dataset.

  See:
  https://cloud.google.com/vertex-ai/docs/training/automl-api

  Args:
    project_id: The Bigquery project_id.
    aiplatform_dataset: The dataset in the AI Platform used for AutoML.
    model_display_name: The name of the AutoML model to create.
    predefined_split_column_name: The key is a name of one of the Dataset's data
        columns. The value of the key (either the label's value or
        value in the column) must be one of {``training``,
        ``validation``, ``test``}, and it defines to which set the
        given piece of data is assigned. If for a piece of data the
        key is not present or has an invalid value, that piece is
        ignored by the pipeline.
    target_column: The target to predict.
    optimization_objective: Objective function the Model is to be optimized
      towards. The training task creates a Model that maximizes/minimizes the
      value of the objective function over the validation set.
      "minimize-rmse" (default) - Minimize root-mean-squared error (RMSE).
      "minimize-mae" - Minimize mean-absolute error (MAE).
      "minimize-rmsle" - Minimize root-mean-squared log error (RMSLE).
    budget_milli_node_hours: The number of node hours to use to train the model
      (times 1000), 1000 milli node hours is 1 mode hour.
    location: The location to train the AutoML model.

  Returns:
    Vertex AI AutoML model.
  """
  transformations = [{'auto': {'column_name': f'{feature}'}}
                     for feature in aiplatform_dataset.column_names
                     if feature not in _NON_FEATURES]

  aiplatform.init(project=project_id, location=location)
  job = aiplatform.AutoMLTabularTrainingJob(
      display_name=model_display_name,
      optimization_prediction_type='regression',
      optimization_objective=optimization_objective,
      column_transformations=transformations)

  model = job.run(
      dataset=aiplatform_dataset,
      target_column=target_column,
      budget_milli_node_hours=budget_milli_node_hours,
      model_display_name=model_display_name,
      predefined_split_column_name=predefined_split_column_name)

  model.wait()
  logging.info('Created AI Platform Model with display name %r',
               model.display_name)
  return model

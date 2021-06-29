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
model_display_name = 'model1'

dataset_id = automl.create_automl_dataset(
    project_id=project_id,
    dataset_id=dataset_id,
    table_name=table_name,
    dataset_display_name=table_name)

response = automl.train_automl_model(
    project_id=project_id,
    dataset_id=dataset_id,
    dataset_display_name=table_name,
    model_display_name=table_name)
"""

import logging
from typing import List

from google.cloud import aiplatform

from google3.google.protobuf import struct_pb2
from google3.net.proto2.python.public import json_format

# External use: from google.protobuf import json_format


logging.getLogger().setLevel(logging.INFO)


def create_automl_dataset(
    aiplatform_client: aiplatform.gapic.DatasetServiceClient,
    project_id: str,
    dataset_id: str,
    table_name: str,
    dataset_display_name: str,
    aiplatform_location: str = 'europe-west4',
    timeout: int = 300) -> str:
  """Creates AutoML Dataset in the AI Platform.

  An AutoML Dataset is required before training a model. See
  https://cloud.google.com/ai-platform-unified/docs/datasets/create-dataset-api#tabular_1

  Args:
    aiplatform_client: AutoML client.
    project_id: The Bigquery project_id.
    dataset_id: The Bigquery dataset_id.
    table_name: The Bigquery training dataset name to use for AutoML.
    dataset_display_name: The display name of the AutoML Dataset to be created.
    aiplatform_location: The location of the AutoML Dataset to be created.
    timeout: The timeout in seconds for creating the Dataset.

  Returns:
    The ID of the dataset.
  """
  bigquery_uri = f'bq://{project_id}.{dataset_id}.{table_name}'
  metadata_dict = {'input_config': {'bigquery_source': {'uri': bigquery_uri}}}
  metadata = json_format.ParseDict(metadata_dict, struct_pb2.Value())

  dataset = {
      'display_name': dataset_display_name,
      'metadata_schema_uri': ('gs://google-cloud-aiplatform/schema/dataset/'
                              'metadata/tabular_1.0.0.yaml'),
      'metadata': metadata,
  }
  parent = f'projects/{project_id}/locations/{aiplatform_location}'
  response = aiplatform_client.create_dataset(parent=parent, dataset=dataset)
  create_dataset_response = response.result(timeout=timeout)
  logging.info('Created AutoML AI Platform Dataset with display name %r',
               dataset_display_name)
  dataset_id = create_dataset_response.name.split('/')[-1]
  return dataset_id


def train_automl_model(
    aiplatform_client: aiplatform.gapic.DatasetServiceClient,
    project_id: str,
    dataset_id: str,
    dataset_display_name: str,
    model_display_name: str,
    features: List[str],
    target_column: str = 'target',
    training_budget_milli_node_hours: int = 1000,
    aiplatform_location: str = 'europe-west4') -> None:
  """Trains an AutoML model given an AutoML Dataset.

  See:
  https://github.com/googleapis/python-aiplatform/blob/master/samples/snippets/create_training_pipeline_tabular_regression_sample.py
  https://cloud.google.com/ai-platform-unified/docs/training/automl-api#training_an_automl_model_using_the_api

  Args:
    aiplatform_client: AutoML client.
    project_id: The Bigquery project_id.
    dataset_id: The Bigquery dataset_id.
    dataset_display_name: The display name of the AutoML Dataset to use.
    model_display_name: The name of the AutoML model to create.
    features: The features to use in the model.
    target_column: The target to predict.
    training_budget_milli_node_hours: The number of node hours to use to train
      the model (times 1000), 1000 milli node hours is 1 mode hour.
    aiplatform_location: The location to train the AutoML model.
  """
  # Set the columns used for training and their data types.
  # 'auto' option allows AutoML to detect data types automatically.
  transformations = []
  for feature in features:
    transformations.append({'auto': {'column_name': f'{feature}'}})
  training_task_inputs_dict = {
      'targetColumn': target_column,
      'predictionType': 'regression',
      'transformations': transformations,
      'trainBudgetMilliNodeHours': training_budget_milli_node_hours,
      'disableEarlyStopping': False,
      # Optimisation objectives: minimize-rmse, minimize-mae, minimize-rmsle.
      'optimizationObjective': 'minimize-rmse',
  }
  training_task_inputs = json_format.ParseDict(training_task_inputs_dict,
                                               struct_pb2.Value())

  training_pipeline = {
      'display_name': dataset_display_name,
      'training_task_definition': ('gs://google-cloud-aiplatform/schema/'
                                   'trainingjob/definition/automl_tabular_1.0'
                                   '.0.yaml'),
      'training_task_inputs': training_task_inputs,
      'input_data_config': {
          'dataset_id': dataset_id,
          'fraction_split': {
              'training_fraction': 0.8,
              'validation_fraction': 0.1,
              'test_fraction': 0.1,
          },
      },
      'model_to_upload': {
          'display_name': model_display_name
      },
  }
  parent = f'projects/{project_id}/locations/{aiplatform_location}'
  aiplatform_client.create_training_pipeline(
      parent=parent, training_pipeline=training_pipeline)

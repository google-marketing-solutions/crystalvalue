# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to allow for custom modelling in Crystalvalue."""

import os
import subprocess
from typing import Any, List, Mapping, Optional, Sequence, Union

from google.cloud import bigquery
from google.cloud import storage
from google.cloud import aiplatform
import joblib
import numpy as np
import pandas as pd
from sklearn import base
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing
import tensorflow as tf
from typing_extensions import Protocol

_MODEL_DIR = 'crystalvalue/custom_model/'
_MODEL_FILENAME = os.path.join(_MODEL_DIR, 'model.joblib')
_EMBEDDING_MODEL_FILENAME = os.path.join(_MODEL_DIR, 'embedding_model.h5')
_CONTAINER_IMAGE = 'gcr.io/prem-data-science/custom_model:latest'
_DOCKER_BUILD_COMMAND = f'docker build -t custom_model {_MODEL_DIR}'
_DOCKER_TAG_COMMAND = f'docker tag custom_model {_CONTAINER_IMAGE}'
_DOCKER_PUSH_COMMAND = 'docker push gcr.io/prem-data-science/custom_model'


class Model(Protocol):
  """Class for structural subtyping of any sklearn regressor or classifier."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    ...

  def fit(self, features: Union[pd.DataFrame, np.ndarray],
          target: Union[pd.Series, np.ndarray], **kwargs: Any) -> None:
    ...

  def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    ...


# ColumnTransformer gives some errors when pickleing a FunctionTransformer that
# contain/use a Keras model. That is why we need to use a custom made
# ColumnSelector.
class ColumnSelector(base.BaseEstimator, base.TransformerMixin):
  """Select only specified columns.

  Attributes:
    columns: List of columns to select from the raw data for the next step in
      the pipeline.
  """

  def __init__(self, columns: Sequence[str]) -> None:
    """Constructor of the ColumnSelector class.

    Args:
      columns:
    """
    self.columns = columns

  def fit(self, data: pd.DataFrame, target: np.ndarray) -> 'ColumnSelector':
    """This fit method does not perform any operations.

    Args:
      data: Input data parameter.
      target: Target data parameter.

    Returns:
      The transformer class being fit.
    """
    return self

  def transform(self, data: pd.DataFrame) -> np.ndarray:
    """Transforms the input data based on the columns to select.

    Args:
      data: Data to trasnform.

    Returns:
      Data columns selected
    """
    return data[self.columns]


class OrdinalEncoderPlusOne(preprocessing.OrdinalEncoder):
  """OrdinalEncoder class that adds 1 to the output when transforming.

  This is needed since currently Scikit-Learn does not allow to have multiple
  values (one per column) for 'unkown_value'. Therefore we need to use -1 and
  then add +1 so it can fit properly into the tf.keras.layers.Embedding later.
  """

  def __init__(self,
               categories='auto',
               dtype=np.float64,
               handle_unknown='error',
               unknown_value=None) -> None:
    """Constructor for OrdinalEncoderPlusOne.

    Please visit the following documentation site for detailed information:
    https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/preprocessing/_encoders.py#L649

    Args:
      categories: Sequence of categories or 'auto' for automatically determining
        the categories.
      dtype: Data type desired.
      handle_unknown: Either {'error', 'use_encoded_value'}. If
        use_encoded_value then 'unknown_value' also needs to be provided.
      unknown_value: Value to use for unknown categories.
    """
    super().__init__(
        categories=categories,
        dtype=dtype,
        handle_unknown=handle_unknown,
        unknown_value=unknown_value)

  def fit(self,
          data: pd.DataFrame,
          target: Optional[np.ndarray] = None) -> 'OrdinalEncoderPlusOne':
    """Fits the regular OrdinalEncoder based on the given data.

    Args:
      data: Input data parameter.
      target: Target data parameter.

    Returns:
      The transformer class being fit.
    """
    return super().fit(X=data, y=target)

  def transform(self, data: pd.DataFrame) -> np.ndarray:
    """Transforms the given data with OrdinalEncoder and then adds one.

    Args:
      data: Data to trasnform.

    Returns:
      Data columns selected
    """
    return super().transform(X=data) + 1


def _save_model_locally_and_gcp(base_path: str,
                                model_name: str,
                                memory_object: Any,
                                bucket_name: str,
                                model_filename: str = 'model.joblib') -> str:
  """Saves a model both locally and then uploads it to GCP bucket.

  Args:
    base_path: Base path where model folders should go.
    model_name: Name of the model. This will create a folder with the model
      name.
    memory_object: Model to save.
    bucket_name: Name of the Google Storage bucket.
    model_filename: Name of the pickled file to be saved. By default is
      'model.joblib' as required by VertexAI pre built containers.

  Returns:
    The path of the folder where the model will be saved, but not the path to
    the model itself. If model is saved in
    'folder1/folder2/model_name/model.joblib' then the returned path will be:
    'folder1/folder2/model_name/'.
  """
  model_dir = os.path.join(base_path, model_name)
  if not os.path.exists(model_dir):
    os.mkdir(model_dir)

  model_path = os.path.join(model_dir, model_filename)
  joblib.dump(value=memory_object, filename=model_path)

  storage_uri = f'gs://{bucket_name}/{model_path}'
  blob = storage.blob.Blob.from_string(storage_uri, client=storage.Client())
  blob.upload_from_filename(model_path)

  return f'gs://{bucket_name}/{model_dir}'


def _embedding_layer(n_categories: int) -> tf.keras.Sequential:
  """Builds a tf.keras.Sequential model with an embeddign and a flatten layer.

  It is meant to be used as a helper function to re-create this embedding model
  for multiple categorical columns.

  Args:
    n_categories: Number of unique categories in the data column/array.

  Returns:
    The sequential model for creating embedding.
  """
  embedding = tf.keras.Sequential([
      tf.keras.layers.Embedding(
          input_dim=n_categories + 1,
          output_dim=int(tf.math.ceil(n_categories**0.5)),
          input_length=1),
      tf.keras.layers.Flatten(),
  ])
  return embedding


def _train_categories_embedding(
    data: pd.DataFrame,
    categorical_features: Sequence[str],
    target_column: str,
    epochs: int,
    embedding_model_path: str = _EMBEDDING_MODEL_FILENAME,
    learning_rate: float = 0.1) -> str:
  """Trains the embedding model for creating embeddings for categorical columns.

  It creates a model that will generate a embedding for each categorical column
  given. The returned model is a part of the model trained and is not compiled.

  Args:
    data: Data to use for training the embedding model.
    categorical_features: Sequence of columns containing the categories.
    target_column: Target column to use for fitting the model.
    epochs: Number of epochs to train for.
    embedding_model_path: Path to save the embedding model.
    learning_rate: Learning rate to use in the Adam optimizer.

  Returns:
    The trained embedding model.
  """
  ordinal_encoder = OrdinalEncoderPlusOne(
      handle_unknown='use_encoded_value', unknown_value=-1)
  categorical_data = pd.DataFrame(
      data=ordinal_encoder.fit_transform(data[categorical_features]),
      columns=categorical_features,
      index=data.index)
  embedding_inputs = [
      tf.keras.layers.Input(shape=(1,), dtype=np.int64)
      for feature in categorical_features
  ]
  embedding_outputs = []
  for feature, embedding_input in zip(categorical_features, embedding_inputs):
    # Plus 1 is always needed and the second + 1 is for the offset we are
    # introducing in OrdinalEncoderPlusOne.
    n_categories = int(categorical_data[feature].max() + 2)
    embedding_output = _embedding_layer(n_categories=n_categories)(
        embedding_input)
    embedding_outputs.append(embedding_output)

  embeddings_concat = tf.keras.layers.concatenate(embedding_outputs)
  model_output = tf.keras.layers.Dense(1)(embeddings_concat)
  full_model = tf.keras.Model(inputs=embedding_inputs, outputs=model_output)
  full_model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss='mse')

  embedding_model = tf.keras.Model(
      inputs=embedding_inputs, outputs=embeddings_concat)
  full_model.fit(
      x=[categorical_data.iloc[:, i] for i in range(categorical_data.shape[1])],
      y=data[target_column],
      epochs=epochs)
  tf.keras.models.save_model(
      model=embedding_model, filepath=embedding_model_path)

  return embedding_model_path


def _get_data_embedding(data: np.ndarray,
                        embedding_model_path: str) -> np.ndarray:
  """Get the embedding prediction for the given data based on given model.

  Args:
    data: Input data to generate the embeddings from.
    embedding_model_path: Path to the embedding model to use for generating the
      embeddings.

  Returns:
    The array containing the embedding for the given data/categories.
  """
  embedding_model = tf.keras.models.load_model(embedding_model_path)
  data_inputs = [data[:, i] for i in range(data.shape[1])]
  return embedding_model.predict(data_inputs)


def _passthrough(data: np.ndarray) -> np.ndarray:
  return data


def _build_sklearn_pipeline(
    feature_types: Mapping[str, List[str]], custom_model: Model,
    embedding_model_path: Optional[str]) -> pipeline.Pipeline:
  """Builds a sklearn pipeline with preprocessing and a given model.

  This pipelines tries to provide a similar (but simpler) interface to what
  using AutoML in Crystalvalue looks like. For that reason it includes null
  inputing and preprocessing for numerical and categorical features.

  Args:
    feature_types: Mapping of feature types along side with the list of columns
      of that type.
    custom_model: Custom model to include in the pipeline.
    embedding_model_path: Path to the embeddign model to use for categorical
      variables.

  Returns:
    A sklearn pipeline to use in Crystalvalue.
  """
  if embedding_model_path:
    embedding_transformer = preprocessing.FunctionTransformer(
        func=_get_data_embedding, validate=False, check_inverse=False,
        kw_args={'embedding_model_path': embedding_model_path})
  else:
    embedding_transformer = preprocessing.FunctionTransformer(
        func=_passthrough)

  numeric_pipeline = pipeline.Pipeline(
      steps=[('selector', ColumnSelector(feature_types['numeric'])),
             ('numeric_imputer', impute.SimpleImputer(strategy='median')),
             ('scaler', preprocessing.StandardScaler())])
  categorical_pipeline = pipeline.Pipeline(
      steps=[('selector',
              ColumnSelector(feature_types['string_or_categorical'])),
             ('category_imputer',
              impute.SimpleImputer(strategy='most_frequent')),
             ('ordinal_encoder',
              OrdinalEncoderPlusOne(
                  handle_unknown='use_encoded_value',
                  unknown_value=-1)
             ), ('embeddings', embedding_transformer)])
  boolean_selector = ColumnSelector(feature_types['boolean'])
  preprocessor = pipeline.FeatureUnion([('categorical', categorical_pipeline),
                                        ('numeric', numeric_pipeline),
                                        ('boolean', boolean_selector)])
  return pipeline.Pipeline(
      steps=[('preprocessor', preprocessor),
             ('regressor', custom_model)], verbose=True)


def _build_and_push_docker_image() -> None:
  subprocess.run(_DOCKER_BUILD_COMMAND.split(), check=True, shell=False)
  subprocess.run(_DOCKER_TAG_COMMAND.split(), check=True, shell=False)
  subprocess.run(_DOCKER_PUSH_COMMAND.split(), check=True, shell=False)


def train_custom_model(custom_model: Model,
                       model_name: str,
                       bigquery_client: bigquery.Client,
                       feature_types: Mapping[str, List[str]],
                       target_column: str,
                       dataset_id: str,
                       table_name: str,
                       location: str) -> aiplatform.Model:
  """Trains a custom model and uploads it to VertexAI.

  Args:
    custom_model: Model object. Must follow a sklearn like approach.
    model_name: Name to use to save the model.
    bigquery_client: Bigquery client to fetch the training data.
    feature_types: Mapping of feature types along side with the list of columns
      of that type.
    target_column: Name of the target column to use.
    dataset_id: GCP dataset id.
    table_name: Name of the table containing the training data.
    location: Location of the resources, make sure they are all in the same
      region.

  Returns:
    Object of aiplatform.Model as uploaded in GCP.
  """
  query_sql = f"""
      SELECT *
      FROM `{bigquery_client.project}.{dataset_id}.{table_name}`
      WHERE predefined_split_column = 'TRAIN'
      """
  data = bigquery_client.query(
      query_sql, location=location).result().to_dataframe()

  embedding_model_path = None
  if feature_types['string_or_categorical']:
    embedding_model_path = _train_categories_embedding(
        data=data,
        categorical_features=feature_types['string_or_categorical'],
        target_column=target_column,
        epochs=5)

  # Train and save locally custom model
  custom_pipeline = _build_sklearn_pipeline(
      feature_types=feature_types,
      custom_model=custom_model,
      embedding_model_path=embedding_model_path)
  custom_pipeline.fit(data, data[target_column])
  custom_pipeline.column_order = data.columns.to_list()
  joblib.dump(value=custom_pipeline,
              filename=_MODEL_FILENAME)
  # Build docker image containing trained model along side with prediction code.
  _build_and_push_docker_image()
  # Upload model to VertexAI
  aiplatform.init(project=bigquery_client.project, location=location)
  model = aiplatform.Model.upload(
      display_name=model_name,
      serving_container_image_uri=_CONTAINER_IMAGE,
      serving_container_predict_route='/predict',
      serving_container_health_route='/health_check',
  )
  model.wait()
  return model


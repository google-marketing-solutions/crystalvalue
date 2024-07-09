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

"""Custom utilities needed for the unpickled model to use on the server side.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn import base
from sklearn import preprocessing
import tensorflow as tf

_MODEL_LOCAL_PATH = '/model.joblib'
_EMBEDDING_MODEL_FILEPATH = '/embedding_model.h5'


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
      columns: Columns that will be selected.
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


# Unused parameter needs to stay as this function needs to have the same
# interface than it did at training time but we override the path of the
# embedding model to use.
def _get_data_embedding(
    data: np.ndarray,
    embedding_model_path: str) -> np.ndarray:
  """Get the embedding prediction for the given data based on given model.

  Args:
    data: Input data to generate the embeddings from.
    embedding_model_path: Path to the embedding model to use for generating the
      embeddings.

  Returns:
    The array containing the embedding for the given data/categories.
  """
  embedding_model = tf.keras.models.load_model(_EMBEDDING_MODEL_FILEPATH)
  data_inputs = [data[:, i] for i in range(data.shape[1])]
  return embedding_model.predict(data_inputs)


def _passthrough(data: np.ndarray) -> np.ndarray:
  return data


class OrdinalEncoderPlusOne(preprocessing.OrdinalEncoder):
  """OrdinalEncoder class that adds 1 to the output when transforming."""

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


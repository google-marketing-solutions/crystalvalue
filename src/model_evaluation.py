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
"""Module for evaluating models built via CrystalValue pipeline."""

import logging

from google.cloud import bigquery
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn import metrics

from src import automl
from src import feature_engineering


def get_test_set(bigquery_client: bigquery.Client,
                 dataset_id: str,
                 table_name: str = 'training_data',
                 predefined_split_column: str = 'predefined_split_column',
                 location: str = 'europe-west4') -> pd.DataFrame:
  """Get test set from Bigquery table for model evaluation.

  Args:
    bigquery_client: BigQuery client.
    dataset_id: BigQuery dataset.
    table_name: The table containing the training data.
    predefined_split_column: The column continaing the "TEST" label.
    location: Bigquery data location.

  Returns:
    Dataframe with test data.
  """
  query = f"""
  SELECT *
  FROM {bigquery_client.project}.{dataset_id}.{table_name}
  WHERE {predefined_split_column} = 'TEST'
  """
  return bigquery_client.query(query, location=location).result().to_dataframe()


def _calculate_normalized_mae(y_actual: pd.Series,
                              y_predicted: pd.Series) -> np.float:
  """Helper function for calculating bin level normalized mean absolute error.

  Args:
    y_actual: The series with actual LTV for test set customers.
    y_predicted: The series with predicted LTV for test set customers.

  Returns:
    Normalized Mean Absolute Error.
  """
  return np.divide(
      metrics.mean_absolute_error(y_actual, y_predicted), np.mean(y_actual))


def _gini(series1: pd.Series, series2: pd.Series) -> np.float64:
  """Returns Gini coefficient between two series.

  Args:
    series1: First series for Gini calculation (typically actual label).
    series2: Second series for Gini calculation  (typically predicted).

  Returns:
    Gini coefficient between the two series.
  """
  y_all = np.asarray(
      np.c_[series1, series2, np.arange(len(series1))], dtype=np.float)
  y_all_sorted_desc = y_all[np.lexsort((y_all[:, 2], -1 * y_all[:, 1]))]
  total_series1 = y_all_sorted_desc[:, 0].sum()
  gini_raw_numerator = y_all_sorted_desc[:, 0].cumsum().sum()
  gini_sum = gini_raw_numerator / total_series1
  gini_sum_adjusted = gini_sum - (len(series1) + 1) / 2.
  return gini_sum_adjusted / len(series1)


def _compute_gini_normalized(y_actual: pd.Series,
                             y_predicted: pd.Series) -> np.float:
  """Produces normalized Gini coefficient between actual and predicted LTV.

  Args:
    y_actual: The series with actual LTV for test set customers.
    y_predicted: The series with predicted LTV for test set customers.

  Returns:
    Normalized Gini Coefficent i.e. Model Gini: Label Gini.
  """
  return np.divide(_gini(y_actual, y_predicted), _gini(y_actual, y_actual))


def _plot_summary_stats(bin_data: pd.DataFrame) -> None:
  """Creates plots with key evaluation metrics for LTV model.

  This function creates visualization with bar charts of bin (default decile)
  level average predicted and actual LTV as well as provides overall
  Spearman Rank Correlation, Normalized Gini coefficient for the model.

  Args:
    bin_data: Dataframe with bin number, Predicted vs. actual bin average LTV.s

  Returns:
      Plot with Average predicted and actual LTV by decile.
  """
  plot_data = bin_data.melt(id_vars='bin').rename(columns=str.title)
  fig, ax1 = plt.subplots(figsize=(10, 7))
  p1 = sns.barplot(x='Bin', y='Value', hue='Variable', data=plot_data)
  ax1.set_title(
      'Model Evaluation  - Bin level average and predicted LTV', fontsize=15)
  p1.set_xlabel('Prediction Bin', fontsize=9)
  p1.set_ylabel('Average LTV', fontsize=9)
  p1.legend(loc='upper right')
  file_path = 'bin_predictions_vs_actuals.png'
  logging.info('Saving figure in %r', file_path)
  fig.savefig(file_path)
  sns.despine(fig)


def evaluate_model_predictions(
    bigquery_client: bigquery.Client,
    dataset_id: str,
    endpoint: str,
    model_id: str,
    training_data_name: str = 'crystalvalue_train_data',
    table_evaluation_stats: str = 'test_set_evaluation',
    location: str = 'europe-west4',
    number_bins: int = 10,
    round_decimal_places: int = 2) -> pd.DataFrame:
  """Creates a plot and BigQuery table with evaluation metrics for LTV model.

  This function creates plots and a table with date of running, model id and
  bin level average predicted and actual LTV, Spearman Rank Correlation,
  Normalized Gini coefficient for the model.
  To ensure consistency and comparision across model runs, these outputs are
  sent to a Big Query table that can capture changes in model performance over
  all iterations.

  Args:
    bigquery_client: Name of Big Query Client.
    dataset_id: Input Big Query Dataset with predictions from CrystalValue.
    endpoint: The Endpoint name.
    model_id: Model ID for the AutoML model.
    training_data_name: The name of the training dataset.
    table_evaluation_stats: Destination Big Query Table to store model results.
    location: Bigquery data location.
    number_bins: Number of bins to split the LTV predictions into for
      evaluation. The default split is into deciles.
    round_decimal_places: The number of decimal places to round to.

  Returns:
    Dataframe with statistics on the model.

  Raises:
    ValueError if the number of bins is less than 2.
  """
  if number_bins < 2:
    raise ValueError('Number of bins should be 2 or more')

  data = get_test_set(
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name=training_data_name,
      location=location)

  data['predicted_value'] = automl.predict_using_deployed_model(
      project_id=bigquery_client.project,
      endpoint=endpoint,
      features=data,
      location=location)

  spearman_correlation = round(
      stats.spearmanr(data['future_value'], data['predicted_value'])[0],
      round_decimal_places)
  gini_normalized = round(
      _compute_gini_normalized(data['future_value'], data['predicted_value']),
      round_decimal_places)
  normalised_mae = round(
      _calculate_normalized_mae(data['future_value'], data['predicted_value']),
      round_decimal_places)

  data['bin'] = pd.qcut(
      data['predicted_value'].rank(method='first'),
      number_bins,
      labels=np.arange(number_bins, 0, -1)).astype(int)

  revenue_shares = pd.DataFrame().append(
      pd.Series(dtype='object'), ignore_index=True)
  total_value = data['future_value'].sum()
  data = data.sort_values('predicted_value', ascending=False)
  for i in [0.01, 0.05, 0.10]:
    number_of_rows = int(i * len(data))
    revenue_shares[
        f'top_{int(i * 100)}_percent_predicted_customers_value_share'] = np.divide(
            data[:number_of_rows]['future_value'].sum(), total_value)
  revenue_shares = revenue_shares.round(round_decimal_places)

  bin_summary = data.groupby('bin')[['predicted_value', 'future_value'
                                    ]].mean().round(round_decimal_places)
  bin_summary.columns = [f'mean_{column}' for column in bin_summary.columns]
  bin_summary = bin_summary.reset_index()
  _plot_summary_stats(bin_data=bin_summary)

  model_summary_statistics = pd.DataFrame.from_records([{
      'time_run': pd.to_datetime('now').strftime('%d/%m/%Y %H:%M:%S'),
      'model_id': model_id,
      'test_set_rows': len(data),
      'spearman_correlation': spearman_correlation,
      'gini_normalized': gini_normalized,
      'normalised_mae': normalised_mae
  }])
  model_summary_statistics = pd.concat(
      [model_summary_statistics, revenue_shares], axis=1)

  feature_engineering.run_load_table_to_bigquery(
      data=model_summary_statistics,
      bigquery_client=bigquery_client,
      dataset_id=dataset_id,
      table_name=table_evaluation_stats,
      location=location,
      write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
  return model_summary_statistics

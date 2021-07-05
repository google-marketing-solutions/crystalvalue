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
from google.cloud import bigquery
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn import metrics


def _fetch_test_set_predictions_from_bigquery(bigquery_client: bigquery.Client,
                                              dataset_name: str,
                                              predictions_table: str,
                                              location: str) -> pd.DataFrame:
  """Helper function for fetching model predictions from Big Query.

  Args:
    bigquery_client: BigQuery client.
    dataset_name: Dataset with LTV predictions from CrystalValue.
    predictions_table: Dataset with LTV predictions from CrystalValue.
    location: Bigquery data location.

  Returns:
      Dataframe with customer id,actual and predicted LTV.
  """
  query = f"""
    SELECT customer_id,
    actual_future_value,
    tables.value as predicted_future_value
    FROM `{bigquery_client.project}.{dataset_name}.{predictions_table}`,
    UNNEST (predicted_subsequent_ad_revenue)
  """
  return bigquery_client.query(query, location=location).result().to_dataframe()


def _calculate_normalized_mae(data: pd.DataFrame) -> np.float:
  """Helper function for calculating bin level normalized mean absolute error.

  Args:
    data: Dataframe with actual and predicted LTV.

  Returns:
      Normalized Mean Absolute Error.
  """
  if np.mean(data.ltv_actual) == 0:
    return np.nan
  else:
    return np.round(
        metrics.mean_absolute_error(data.ltv_actual, data.ltv_predicted) /
        np.mean(data.ltv_actual), 3)


def _calculate_bin_averages(y_actual: pd.Series, y_predicted: pd.Series,
                            number_bins: int) -> pd.DataFrame:
  """Helper function for bin level averages used in creating decile charts.

  Args:
    y_actual: The series with actual LTV for test set customers.
    y_predicted: The series with predicted LTV for test set customers.
    number_bins: Number of bins to split the series into. The default split is
      into deciles.

  Returns:
      Dataframe with bin number, Predicted vs. actual bin average LTV, bin level
      MAPE and MAE.
  """
  if number_bins < 2:
    raise ValueError('Number of bins should be 2 or more')
  bin_number = pd.qcut(
      y_predicted.rank(method='first'), number_bins, labels=False)

  temporary_data = pd.DataFrame(
      list(zip(y_predicted, y_actual, bin_number)),
      columns=['ltv_predicted', 'ltv_actual', 'bin'])
  bin_stats_data = temporary_data.groupby('bin').agg('mean').round(
      3).reset_index()
  bin_stats_data['normalized_MAPE'] = np.round(
      np.abs(bin_stats_data['ltv_predicted'] - bin_stats_data['ltv_actual']) /
      bin_stats_data['ltv_actual'], 4)
  bin_stats_data['normalized_MAE'] = temporary_data.groupby('bin').apply(
      _calculate_normalized_mae)

  return bin_stats_data


def _compute_gini(series1: pd.Series, series2: pd.Series) -> np.float64:
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
  gini_model = _compute_gini(y_actual, y_predicted)
  gini_label = _compute_gini(y_actual, y_actual)
  gini_normalized = gini_model / gini_label
  return round(gini_normalized, 2)


def _plot_summary_stats(bin_data: pd.DataFrame, spearman_correlation: np.float,
                        gini_normalized: np.float) -> None:
  """Creates plots with key evaluation metrics for LTV model.

  This function creates visualization with bar charts of bin (default decile)
  level average predicted and actual LTV as well as provides overall
  Spearman Rank Correlation, Normalized Gini coefficient for the model.

  Args:
    bin_data: Dataframe with bin number, Predicted vs. actual bin average LTV.
    spearman_correlation: Spearman Correlation between actual and predicted LTV
      for test set customers.
    gini_normalized: Normalized Gini coefficient between actual and predicted
      LTV for test set customers

  Returns:
      Plot with Average predicted and actual LTV by decile
      along with Spearman and Normalized Gini Coefficient.
  """
  plot_data = bin_data[['bin', 'ltv_predicted', 'ltv_actual'
                       ]].melt(id_vars='bin').rename(columns=str.title)
  fig, ax1 = plt.subplots(figsize=(10, 7))
  p1 = sns.barplot(x='Bin', y='Value', hue='Variable', data=plot_data)
  ax1.set_title(
      'Model Evaluation  - Bin level average and predicted LTV', fontsize=15)
  p1.set_xlabel('Prediction Bin', fontsize=9)
  p1.set_ylabel('Average LTV', fontsize=9)
  p1.legend(loc='upper left')
  plt.figtext(
      0.5,
      0.001, f'Spearman Correlation is {spearman_correlation}.\n'
      f' Normalized Gini Coefficient is {gini_normalized}.\n',
      horizontalalignment='center')
  sns.despine(fig)


def _load_table_to_bigquery(data: pd.DataFrame,
                            bigquery_client: bigquery.Client, dataset_name: str,
                            table_name: str, location: str) -> None:
  """Loads a Pandas Dataframe to Bigquery."""
  table_id = f'{bigquery_client.project}.{dataset_name}.{table_name}'
  job_config = bigquery.job.LoadJobConfig(
      write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
  bigquery_client.load_table_from_dataframe(
      dataframe=data,
      destination=table_id,
      job_config=job_config,
      location=location).result()


def _create_summary_stats_data(bin_data: pd.DataFrame, model_display_name: str,
                               spearman_correlation: np.float,
                               gini_normalized: np.float) -> pd.DataFrame:
  """Creates dataframe with key evaluation metrics for LTV model.

  This function creates a dataframe with bin level average predicted and actual
  LTV as well as provides overall
  Spearman Rank Correlation, Normalized Gini coefficient and decile level MAE
  and MAPE for the model.

  Args:
    bin_data: Dataframe with bin number, Predicted vs. actual bin average LTV.
    model_display_name: Display Name for the AutoML model.
    spearman_correlation: Spearman Correlation between actual and predicted LTV
      for test set customers.
    gini_normalized: Normalized Gini coefficient between actual and predicted
      LTV for test set customers

  Returns:
      Dataframe with details of model and average predicted and actual LTV by
      decile along with Spearman and Normalized Gini Coefficient.
  """

  all_deciles = pd.DataFrame(np.arange(0, 10))
  all_deciles.columns = ['bin']
  all_deciles_reshaped = pd.merge(
      all_deciles, bin_data, on='bin', how='left').melt(id_vars='bin').rename(
          columns=str.title).sort_values(by=['Bin', 'Variable'])
  all_deciles_reshaped.insert(
      0, 'bin_series', 'bin_' + all_deciles_reshaped['Bin'].astype(str) + '_' +
      all_deciles_reshaped['Variable'])
  all_deciles_reshaped.drop(['Bin', 'Variable'], inplace=True, axis=1)
  all_deciles_model = all_deciles_reshaped.set_index('bin_series').T
  all_deciles_model['model_display_name'] = model_display_name

  metrics_model = pd.DataFrame.from_records([{
      'time_run': pd.to_datetime('now'),
      'model_display_name': model_display_name,
      'spearman_correlation': spearman_correlation,
      'gini_normalized': gini_normalized
  }])

  summary_stats_model = pd.merge(
      metrics_model, all_deciles_model, how='outer', on='model_display_name')
  return summary_stats_model


def evaluate_model_predictions(bigquery_client: bigquery.Client,
                               dataset_name: str,
                               predictions_table: str,
                               model_display_name: str,
                               table_evaluation_stats: str,
                               location: str = 'europe-west4',
                               number_bins: int = 10) -> pd.DataFrame:
  """Creates a plot and Big Query table with evaluation metrics for LTV model.

  This function creates plots and a table with date of running, model name and
  bin level average predicted and actual LTV,Spearman Rank Correlation,
  Normalized Gini coefficient for the model.
  To ensure consistency and comparision across model runs, these outputs are
  sent to a Big Query table that can capture changes in model performance over
  all iterations.

  Args:
    bigquery_client: Name of Big Query Client.
    dataset_name: Input Big Query Dataset with predictions from CrystalValue.
    predictions_table: Input Big Query Table with predictions from CrystalValue.
    model_display_name: Display Name for the AutoML model.
    table_evaluation_stats: Destination Big Query Table to store model results.
    location: Bigquery data location.
    number_bins: Number of bins to split the LTV predictions into for
      evaluation. The default split is into deciles.

  Returns:
      Dataframe with details of model and average predicted and actual LTV,
      Normalized MAE and MAPE by
      decile along with Spearman and Normalized Gini Coefficient.
  """
  test_data = _fetch_test_set_predictions_from_bigquery(
      bigquery_client=bigquery_client,
      dataset_name=dataset_name,
      predictions_table=predictions_table,
      location=location)
  y_actual = test_data['actual_future_value']
  y_predicted = test_data['predicted_future_value']
  spearman_correlation = round(stats.spearmanr(y_actual, y_predicted)[0], 2)
  gini_normalized = _compute_gini_normalized(y_actual, y_predicted)
  bin_data = _calculate_bin_averages(y_actual, y_predicted, number_bins)
  _plot_summary_stats(
      bin_data=bin_data,
      spearman_correlation=spearman_correlation,
      gini_normalized=gini_normalized)
  summary_stats_model = _create_summary_stats_data(
      bin_data=bin_data,
      model_display_name=model_display_name,
      spearman_correlation=spearman_correlation,
      gini_normalized=gini_normalized)
  _load_table_to_bigquery(
      data=summary_stats_model,
      bigquery_client=bigquery_client,
      dataset_name=dataset_name,
      table_name=table_evaluation_stats,
      location=location)
  return bin_data

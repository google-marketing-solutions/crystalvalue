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

-- Build predict dataset for a machine learning model to predict LTV.

-- @param days_lookback INT The number of days to look back to create features.
-- @param customer_id_column STRING The column containing the customer ID.
-- @param date_column STRING The column containing the transaction date.
-- @param value_column STRING The column containing the value column.
-- @param features_sql STRING The SQL for the features and transformations.

WITH
  WindowDate AS (
    SELECT DATE(MAX({date_column})) as date
    FROM {project_id}.{dataset_id}.{table_name}
  ),
  CustomerWindows AS (
    SELECT
      CAST(TX_DATA.{customer_id_column} AS STRING) AS customer_id,
      WindowDate.date AS window_date,
      DATE_SUB((WindowDate.date), INTERVAL {days_lookback} day) AS lookback_start,
      DATE_ADD((WindowDate.date), INTERVAL 1 day) AS lookahead_start,
      DATE_ADD((WindowDate.date), INTERVAL {days_lookahead} day) AS lookahead_stop
    FROM {project_id}.{dataset_id}.{table_name} AS TX_DATA
    {date_window_join_sql}
    GROUP BY 1, 2, 3, 4
  )
SELECT
  CustomerWindows.*,
  IFNULL(
    DATE_DIFF(CustomerWindows.window_date, MAX(DATE(TX_DATA.{date_column})), DAY),
    {days_lookback}) AS days_since_last_transaction,
  IFNULL(
    DATE_DIFF(CustomerWindows.window_date, MIN(DATE(TX_DATA.{date_column})), DAY),
    {days_lookback}) AS days_since_first_transaction,
  COUNT(*) AS count_transactions,
  {features_sql}
FROM
  CustomerWindows
JOIN
  {project_id}.{dataset_id}.{table_name} AS TX_DATA
  ON (
    CAST(TX_DATA.{customer_id_column} AS STRING) = CustomerWindows.customer_id
    AND DATE(TX_DATA.{date_column})
      BETWEEN CustomerWindows.lookback_start
      AND DATE(CustomerWindows.window_date))
GROUP BY
  1, 2, 3, 4, 5;

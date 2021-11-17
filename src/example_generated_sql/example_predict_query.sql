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
    SELECT DATE(MAX(date)) as date
    FROM my_project.my_dataset.my_table
  ),
  CustomerWindows AS (
    SELECT
      CAST(TX_DATA.customer_id AS STRING) AS customer_id,
      WindowDate.date AS window_date,
      DATE_SUB((WindowDate.date), INTERVAL 365 day) AS lookback_start,
      DATE_ADD((WindowDate.date), INTERVAL 1 day) AS lookahead_start,
      DATE_ADD((WindowDate.date), INTERVAL 365 day) AS lookahead_stop
    FROM my_project.my_dataset.my_table AS TX_DATA
    CROSS JOIN WindowDate
    GROUP BY 1, 2, 3, 4
  )
SELECT
  CustomerWindows.*,
  IFNULL(
    DATE_DIFF(CustomerWindows.window_date, MAX(DATE(TX_DATA.date)), DAY),
    365) AS days_since_last_transaction,
  IFNULL(
    DATE_DIFF(CustomerWindows.window_date, MIN(DATE(TX_DATA.date)), DAY),
    365) AS days_since_first_transaction,
  COUNT(*) AS count_transactions,
  MIN(numeric_column) AS min_numeric_column,
  MAX(numeric_column) AS max_numeric_column,
  SUM(numeric_column) AS sum_numeric_column,
  AVG(numeric_column) AS avg_numeric_column,
  MIN(value) AS min_value,
  MAX(value) AS max_value,
  SUM(value) AS sum_value,
  AVG(value) AS avg_value,
  MIN(CAST(bool_column AS INT)) AS min_bool_column,
  MAX(CAST(bool_column AS INT)) AS max_bool_column,
  SUM(CAST(bool_column AS INT)) AS sum_bool_column,
  AVG(CAST(bool_column AS INT)) AS avg_bool_column,
  TRIM(STRING_AGG(DISTINCT categorical_column, " " ORDER BY categorical_column)) AS categorical_column,
  TRIM(STRING_AGG(DISTINCT text_column, " " ORDER BY text_column)) AS text_column
FROM
  CustomerWindows
JOIN
  my_project.my_dataset.my_table AS TX_DATA
  ON (
    CAST(TX_DATA.customer_id AS STRING) = CustomerWindows.customer_id
    AND DATE(TX_DATA.date)
      BETWEEN CustomerWindows.lookback_start
      AND DATE(CustomerWindows.window_date))
GROUP BY
  1, 2, 3, 4, 5;

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

-- Build training dataset for a machine learning model to predict LTV.

-- The query creates a column `predefined_split_column` which takes approximately the
-- values `TEST` for 15% of customers, `VALIDATE` for 15% of customers and
-- `TRAIN` for 70% of customers. Note that there can be multiple rows for each customer (i.e.
-- customers are observed during different time windows). Time windows are defined by customers and
-- months, so a customer can be observed each month (with different features and future values).
-- There is at most 10 rows per customer in the final dataset (randomly sampled months) which
-- prevents frequent purchasers from having too much weight.

-- @param days_lookback INT The number of days to look back to create features.
-- @param days_lookahead INT The number of days to look ahead to predict LTV.
-- @param customer_id_column STRING The column containing the customer ID.
-- @param date_column STRING The column containing the transaction date.
-- @param value_column STRING The column containing the value column.
-- @param features_sql STRING The SQL for the features and transformations.

WITH
  DateWindowsTable AS (
    SELECT window_date
    FROM
      UNNEST(
        GENERATE_DATE_ARRAY(
          DATE_ADD(
            DATE(
              (SELECT MIN(date) FROM my_project.my_dataset.my_table)),
            INTERVAL 365 DAY),
          DATE_SUB(
            DATE(
              (SELECT MAX(date) FROM my_project.my_dataset.my_table)),
            INTERVAL 365 DAY),
          INTERVAL 1 MONTH)) AS window_date
  ),
  CustomerWindows AS (
    SELECT DISTINCT
      TX_DATA.customer_id AS customer_id,
      DateWindowsTable.window_date AS window_date,
      DATE_SUB(DateWindowsTable.window_date, INTERVAL 365 day) AS lookback_start,
      DATE_ADD(DateWindowsTable.window_date, INTERVAL 1 day) AS lookahead_start,
      DATE_ADD(DateWindowsTable.window_date, INTERVAL 365 day) AS lookahead_stop,
      ROW_NUMBER()
        OVER (PARTITION BY TX_DATA.customer_id ORDER BY RAND()) AS customer_window_number
    FROM my_project.my_dataset.my_table AS TX_DATA
    CROSS JOIN DateWindowsTable
  ),
  Target AS (
    SELECT
      CustomerWindows.*,
      SUM(IFNULL(TX_DATA.value, 0)) AS future_value,
    FROM
      CustomerWindows
    LEFT JOIN
      my_project.my_dataset.my_table AS TX_DATA
      ON (
        TX_DATA.customer_id = CustomerWindows.customer_id
        AND DATE(TX_DATA.date)
          BETWEEN CustomerWindows.lookahead_start
          AND CustomerWindows.lookahead_stop)
    GROUP BY
      1, 2, 3, 4, 5, 6
  )
SELECT
  Target.* EXCEPT (customer_window_number),
  CASE
    WHEN
      ABS(
        MOD(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(Target.customer_id))), 100))
      BETWEEN 0
      AND 15
      THEN 'TEST'
    WHEN
      ABS(
        MOD(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(Target.customer_id))), 100))
      BETWEEN 15
      AND 30
      THEN 'VALIDATE'
    WHEN
      ABS(
        MOD(FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(Target.customer_id))), 100))
      BETWEEN 30
      AND 100
      THEN 'TRAIN'
    END AS predefined_split_column,
  IFNULL(
    DATE_DIFF(Target.window_date, MAX(DATE(TX_DATA.date)), DAY),
    365) AS days_since_last_transaction,
  IFNULL(
    DATE_DIFF(Target.window_date, MIN(DATE(TX_DATA.date)), DAY),
    365) AS days_since_first_transaction,
  COUNT(*) AS count_transactions,
  SUM(numeric_column) as sum_numeric_column,
  MAX(numeric_column) as max_numeric_column,
  MIN(numeric_column) as min_numeric_column,
  AVG(numeric_column) as avg_numeric_column,
  SUM(value) as sum_value,
  MAX(value) as max_value,
  MIN(value) as min_value,
  AVG(value) as avg_value,
  SUM(CAST(bool_column AS INT)) as sum_bool_column,
  MAX(CAST(bool_column AS INT)) as max_bool_column,
  MIN(CAST(bool_column AS INT)) as min_bool_column,
  AVG(CAST(bool_column AS INT)) as avg_bool_column,
  TRIM(STRING_AGG(DISTINCT categorical_column, " " ORDER BY categorical_column)) AS categorical_column,
  TRIM(STRING_AGG(DISTINCT text_column, " " ORDER BY text_column)) AS text_column
FROM
  Target
JOIN
  my_project.my_dataset.my_table AS TX_DATA
  ON (
    TX_DATA.customer_id = Target.customer_id
    AND DATE(TX_DATA.date) BETWEEN Target.lookback_start AND DATE(Target.window_date))
WHERE Target.customer_window_number <= 10
GROUP BY
  1, 2, 3, 4, 5, 6, 7;


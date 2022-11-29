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
              (SELECT MIN({date_column}) FROM {project_id}.{dataset_id}.{table_name})),
            INTERVAL {days_lookback} DAY),
          DATE_SUB(
            DATE(
              (SELECT MAX({date_column}) FROM {project_id}.{dataset_id}.{table_name})),
            INTERVAL {days_lookahead} DAY),
          INTERVAL 1 MONTH)) AS window_date
  ),
  CustomerWindows AS (
    SELECT DISTINCT
      CAST(TX_DATA.{customer_id_column} AS STRING) AS customer_id,
      DateWindowsTable.window_date AS window_date,
      DATE_SUB(DateWindowsTable.window_date, INTERVAL {days_lookback} day) AS lookback_start,
      DATE_ADD(DateWindowsTable.window_date, INTERVAL 1 day) AS lookahead_start,
      DATE_ADD(DateWindowsTable.window_date, INTERVAL {days_lookahead} day) AS lookahead_stop
    FROM {project_id}.{dataset_id}.{table_name} AS TX_DATA
    CROSS JOIN DateWindowsTable
  ),
  Target AS (
    SELECT
      CustomerWindows.*,
      SUM(IFNULL(TX_DATA.{value_column}, 0)) AS future_value,
    FROM
      CustomerWindows
    LEFT JOIN
      {project_id}.{dataset_id}.{table_name} AS TX_DATA
      ON (
        CAST(TX_DATA.{customer_id_column} AS STRING) = CustomerWindows.customer_id
        AND DATE(TX_DATA.{date_column})
          BETWEEN CustomerWindows.lookahead_start
          AND CustomerWindows.lookahead_stop)
    GROUP BY
      1, 2, 3, 4, 5
  )
SELECT
  Target.*,
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
    DATE_DIFF(Target.window_date, MAX(DATE(TX_DATA.{date_column})), DAY),
    {days_lookback}) AS days_since_last_transaction,
  IFNULL(
    DATE_DIFF(Target.window_date, MIN(DATE(TX_DATA.{date_column})), DAY),
    {days_lookback}) AS days_since_first_transaction,
  COUNT(*) AS count_transactions,
  {features_sql}
FROM
  Target
JOIN
  {project_id}.{dataset_id}.{table_name} AS TX_DATA
  ON (
    CAST(TX_DATA.{customer_id_column} AS STRING) = Target.customer_id
    AND DATE(TX_DATA.{date_column}) BETWEEN Target.lookback_start AND DATE(Target.window_date))
GROUP BY
  1, 2, 3, 4, 5, 6, 7;

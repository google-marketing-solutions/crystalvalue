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

WITH WindowDate AS (
    SELECT DATE(MAX(date)) as date
    FROM my_project.my_dataset.my_table
),

CustomerWindows AS (
  SELECT
    CAST(TX_DATA.customer_id AS STRING) AS customer_id,
    WindowDate.date AS window_date,
    DATE_SUB((WindowDate.date), INTERVAL 90 day) AS lookback_start,
    DATE_ADD((WindowDate.date), INTERVAL 1 day) AS lookahead_start,
    DATE_ADD((WindowDate.date), INTERVAL 365 day) AS lookahead_stop
  FROM my_project.my_dataset.my_table AS TX_DATA
  CROSS JOIN WindowDate
  GROUP BY 1,2,3,4
)

SELECT
  CustomerWindows.*,
  IFNULL(
    DATE_DIFF(CustomerWindows.window_date, MAX(DATE(TX_DATA.date)), DAY),
    90) AS days_since_last_transaction,
  IFNULL(
    DATE_DIFF(CustomerWindows.window_date, MIN(DATE(TX_DATA.date)), DAY),
    90) AS days_since_first_transaction,
  COUNT(*) AS count_transactions,
  SUM(Quantity) AS sum_Quantity,
  MAX(Quantity) AS max_Quantity,
  MIN(Quantity) AS min_Quantity,
  AVG(Quantity) AS avg_Quantity,
  SUM(Price) AS sum_Price,
  MAX(Price) AS max_Price,
  MIN(Price) AS min_Price,
  AVG(Price) AS avg_Price,
  TRIM(STRING_AGG(DISTINCT StockCode, " " ORDER BY StockCode)) AS StockCode,
  TRIM(STRING_AGG(DISTINCT Description, " " ORDER BY Description)) AS Description,
  TRIM(STRING_AGG(DISTINCT Country, " " ORDER BY Country)) AS Country
FROM
  CustomerWindows
JOIN
  my_project.my_dataset.my_table AS TX_DATA
  ON (
    CAST(TX_DATA.customer_id AS STRING) = CustomerWindows.customer_id
    AND DATE(TX_DATA.date) BETWEEN CustomerWindows.lookback_start AND DATE(CustomerWindows.window_date))
GROUP BY
  1, 2, 3, 4, 5;

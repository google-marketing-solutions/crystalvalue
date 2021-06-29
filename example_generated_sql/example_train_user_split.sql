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

-- Example generated query for building training data to predict LTV.

WITH customer_windows AS (
SELECT
  DISTINCT data.customer_id AS customer_id,
  DATE("2020-04-22") AS window_date,
  DATE_SUB(DATE("2020-04-22"), interval 365 day) AS lookback_start,
  DATE_ADD(DATE("2020-04-22"), interval 1 day) AS lookahead_start,
  DATE_ADD(DATE("2020-04-22"), interval 365 day) AS lookahead_stop
FROM my_project.dataset_ltv.synthetic_transaction_data AS data
ORDER BY RAND()
),

target AS (
SELECT
  customer_windows.*,
  SUM(IFNULL(data.value, 0)) as future_value,
FROM
  customer_windows
LEFT JOIN
  my_project.dataset_ltv.synthetic_transaction_data data
ON
  (data.customer_id = customer_windows.customer_id AND
   DATE(data.date) BETWEEN customer_windows.lookahead_start AND customer_windows.lookahead_stop)
GROUP BY
  1,2,3,4,5
)

SELECT
  target.*,
  IFNULL(DATE_DIFF(target.window_date, MAX(DATE(data.date)), DAY), 365) AS days_since_last_purchase,
  AVG(numeric_column) as avg_numeric_column,
  MAX(numeric_column) as max_numeric_column,
  MIN(numeric_column) as min_numeric_column,
  AVG(value) as avg_value,
  MAX(value) as max_value,
  MIN(value) as min_value,
  TRIM(STRING_AGG(DISTINCT categorical_column, " " order by categorical_column)) AS categorical_column,
  TRIM(STRING_AGG(DISTINCT text_column, " " order by text_column)) AS text_column,

FROM
  target
JOIN
  my_project.dataset_ltv.synthetic_transaction_data data
ON
  (data.customer_id = target.customer_id AND
   DATE(data.date) BETWEEN target.lookback_start AND DATE(target.window_date))
GROUP BY
  1,2,3,4,5,6;

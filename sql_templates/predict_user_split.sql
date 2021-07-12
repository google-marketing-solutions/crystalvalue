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

-- The query creates an additional column `predefined_split_column` which takes approximately the
-- values `TEST` for 15% of rows, `VALIDATE` for 15% of rows and `TRAIN` for 70% of rows.

-- @param window_date STRING The date for the window date
-- @param days_look_back INT The number of days to look back to create features.
-- @param days_look_ahead INT The number of days to look ahead to predict LTV.
-- @param customer_id_column STRING The column containing the customer ID.
-- @param date_column STRING The column containing the transaction date.
-- @param value_column STRING The column containing the value column.
-- @param numerical_features STRING The SQL for numerical features and transformations.
-- @param non_numerical_features STRING The SQL non-numerical features and transformations.


WITH customer_windows AS (
SELECT
  DISTINCT data.{customer_id_column} AS customer_id,
  DATE("{window_date}") AS window_date,
  DATE_SUB(DATE("{window_date}"), interval {days_look_back} day) AS lookback_start,
  DATE_ADD(DATE("{window_date}"), interval 1 day) AS lookahead_start,
  DATE_ADD(DATE("{window_date}"), interval {days_look_ahead} day) AS lookahead_stop
FROM {project_id}.{dataset_id}.{table_name} AS data
)

SELECT
  customer_windows.*,
  IFNULL(DATE_DIFF(customer_windows.window_date, MAX(DATE(data.{date_column})), DAY),
         {days_look_back}) AS days_since_last_purchase,
  {numerical_features_sql},
  {non_numerical_features_sql}
FROM
  customer_windows
JOIN
  {project_id}.{dataset_id}.{table_name} data
ON
  (data.{customer_id_column} = customer_windows.customer_id AND
   DATE(data.{date_column}) BETWEEN customer_windows.lookback_start AND DATE(customer_windows.window_date))
GROUP BY
  1,2,3,4,5,6,7;

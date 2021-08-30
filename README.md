# Crystalvalue

## Overview

Crystalvalue is a solution for running Predictive Customer LifeTime Value (pLTV) projects leveraging [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai). It enables advertisers to quickly build effective pLTV machine learning models using 1st party data and to programmatically set up infrastructure for predictions, continuous pLTV insights and model maintenance. pLTV predictions are used by advertisers to boost digital advertising performance. Crystalvalue was created by experienced Google marketing data scientists and customer solution engineers.


## Quick Start

Check out the demo notebook at **crystalvalue_demo_notebook.ipynb** to run the Crystvalue pLTV solution for a transaction dataset from Kaggle, from feature engineering to scheduling predictions. The demo notebook can be repurposed to run on an advertiser's own data simply by changing the input table name. The demo is intended to be run inside of a [Google Cloud Notebook](https://cloud.google.com/vertex-ai/docs/general/notebooks).

## What is Predictive Customer LifeTime Value?

A pLTV model predicts the ‘value’ (i.e. total profit or revenue) for each customer during a future time period (lookahead window) based on aggregated historical information (‘features’) during a past time period (lookback window).

For model training, ‘customer-windows’ are created which are made up of ‘features’ and ‘value’. For serving predictions, only the ‘features’ need to be created and the model predicts the future value.

  ![Crystalval](https://screenshot.googleplex.com/BR7QoBzQnv8CRL5.png)



## Solution description

Crystalvalue is an advertising solution that runs Predictive Customer LifeTime Value (pLTV) projects. It is a Python library that is intended to be run in a [Google Cloud Notebook](https://cloud.google.com/vertex-ai/docs/general/notebooks) within an advertiser's Google Cloud Platform account. Each step of the LTV modelling and prediction process is carried out programatically from the notebook, including scheduling predictions and model retraining. Crystalvalue uses the Vertex SDK for Python to interact with the user's Google Cloud Platform.

The steps in the Crystalvalue LTV process are outlined below and are set out in Figure 1.

**Figure 1**

  ![Crystalval](https://screenshot.googleplex.com/4xGe5uQ4z7jQMEZ.png)

* **Data cleaning**. Crystalvalue takes in a BigQuery table (e.g. data from Google Analytics or a CRM transaction dataset) and performs automated data checks. It will also output a table in your Bigquery dataset (default name: `crystalvalue_data_statistics`) with important statistics to help the user decide whether data cleaning is necessary (i.e. negative prices, extreme outliers):
  * number of customers
  * number of transactions
  * total analysis days and the number of dates with transactions
  * minimum and maximum transaction dates
  * lookahead return rate (i.e. the rate at which customer come back in the lookahead period)
  * lookahead mean returns (i.e. the average number of purchases made in the lookahead period)
  * conditional lookahead mean returns (i.e. the average number of purchases made in the lookahead period for those who did come back)
  * price distribution
  * number of transactions per customer distribution

* **Build ML data.** Crystalvalue takes in a BigQuery table (e.g. data from Google Analytics or a CRM transaction dataset) and outputs an ML ready dataset (default name for model training: `crystalvalue_train_data`, default name for prediction: `crystalvalue_predict_data`). For model training, this consists of a set of features and targets. For prediction, this is just a set of features. See Figure 2. The same customer can be observed multiple times during different time windows for model training using monthly ‘sliding windows’. Crystalvalue automatically detects data types from the input Bigquery table schema and applies the appropriate feature transformations for any number of features. Default feature transformations for numerical and boolean columns are: MAX, MIN, AVG, SUM. Default feature transformations for string or categorical columns are an array aggregation. See Figure 3 for the typical features and transformations that can be used in LTV models. The processing is programmatically executed in the Python Bigquery API. This step also creates a column (`predefined_split_column`) which assigns customers to a training, validation and test set (split with random 15% of users as test, 15% in validation and 70% in training). Parameters that can be configured:
  *  days_lookback: # days to look back to create features (default: 365 days).
  *  days_lookahead: # days to look ahead to predict value (default: 365 days).
  *  features: Input data columns types (default: None, detected automatically).
  *  customer_id_column: The customer id column.
  *  date_column: The transaction date or timestamp column.
  *  value_column: The value column (i.e. transaction revenue/profit).
  *  ignore_columns: List of columns to ignore for feature engineering or prediction.
  *  write_executed_query_file: The file path to write the executed query (default: None).

  **Figure 2**

  ![Crystalval](https://screenshot.googleplex.com/645o76szJkYPVZg.png)


  **Figure 3**

  ![Crystalval](https://screenshot.googleplex.com/64VJyTq9WiU6Fpp.png)

* **Model Training**. Crystalvalue will programmatically train a Vertex AI Tabular AutoML model which will then be visible in the [Vertex AI dashboard](https://console.cloud.google.com/vertex-ai). The default names of the Vertex Dataset and Model are `crystalvalue_dataset` and `crystalvalue_model`. The amount of node hours of training is specified using the `budget_milli_node_hours` argument (default is 1000 milli hours, which is equivalent to 1 hour). AutoML carries out impressive [data prepation](https://cloud.google.com/automl-tables/docs/data-best-practices#tables-does) before creating the model which means it can ingest features that are:
  * Plain Text (e.g. customer searches on your website)
  * Arrays (e.g. product pages visited)
  * Numerical columns (e.g. add on insurance cost)
  * Categorical columns (e.g. country)



* **Model evaluation.** Crystalvalue will run model evaluation on the test data (a set of customers that were not included in the model training) and report the metrics below. These metrics will be appended to the BigQuery table `crystalvalue_evaluation` (which will be created if it does not exist).
  * time_run: The time that the model evaluation was run.
  * model_id: The ID of the Vertex Model.
  * spearman_correlation: The spearman correlation. This is a measure of how well the model ranked the Liftetime value of customers in the test set.
  * gini_normalised: The normalised Gini coefficient. This is a measure of how well the model predicted the Lifetime value of customers in the test set.
  * mae_normalised: The normalised Mean Average Error (MAE). This is a measure of the error of the model's predictions for Lifetime value in the test set.
  * top_x_percent_predicted_customer_value_share: The proportion of value (i.e. total profit or revenue) in the test set that is accounted for by the top x% model-predicted customers.

* **Customer insights.** (WIP) Crystalvalue will connect your trained model to the What IF tool (see an [example here](https://pair-code.github.io/what-if-tool/demos/age.html)) which can help you understand the characteristics of your high LTV customers.

* **Predictions & Scheduling.** The model will make predictions for all the customers in the input table that have any activity during the lookback window. The pLTV predictions will be for the period from the last date in the input table (which is the start of the lookahead window) until the length of the lookahead window after the last date in the input table. Crystalvalue will provide you the option to schedule your predictions using the model on a regular basis using Vertex Pipelines from within the Notebook. See the example in the demo notebook which schedules predictions for 1am everyday. Once the schedule is set up, it will be visible in [Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).


## Requirements

* **User:** Crystalvalue users should be familiar with basic Python.
* **Data:** Crystalvalue input data should either be transaction or browsing data that includes customer purchases stored in Bigquery. There is no strict limit on the amount of historical data required, however, more data is usually better. For an advertiser seeking to predict 365 days in the future using 365 days in the past they will require at least 2 years of data. The following columns are required (a customer ID column, a transaction date column and a transaction value column) but it is recommended to use more information in pLTV models.
* **Google Cloud Platform:** The following APIs will be needed:
  * BigQuery
  * Vertex AI
  * (only for scheduling) Cloud Storage, Container Registry

## The team

*  dfkelly@google.com
*  dulacp@google.com
*  pduque@google.com
*  sumedhamenon@google.com



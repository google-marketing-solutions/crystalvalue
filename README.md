# Crystalvalue

## Overview

Crystalvalue is a best practice comprehensive framework and solution for running end-to-end
LifeTime Value (LTV) projects leveraging [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai). It allows advertisers to quickly build effective LTV machine learning models and programmatically set up infrastructure for predictions, evaluation, model retraining and maintenance. Crystalvalue was created by experienced Google marketing data scientists and customer engineers.


## Quick start

Head over to the demo notebook at **crystalvalue_demo_notebook.ipynb** to run the Crystvalue LTV solution, from feature engineering to scheduling predictions, for a transaction dataset from Kaggle, or repurpose it for your own data.


## Solution description

Crystalvalue carrys out the key steps for building and maintaining LTV models. It is a python library that is designed to be run in a [Google Cloud Notebook](https://cloud.google.com/vertex-ai/docs/general/notebooks) within an advertiser's Google Cloud Platform. 

The steps in the Crystalvalue LTV process are outlined below and are set out in Figure 1.

**Figure 1**

![Crystalval](https://screenshot.googleplex.com/8UnbsrpZwYByFVb.png)

* **Data cleaning**. Crystalvalue takes in a dataset (e.g. data from Google Analytics or a CRM transaction dataset) and performs automated data checks. It will output a table in your Bigquery dataset with key statistics: # customers, # transactions, distribution of prices, distribution of transactions per customer. These statistics can be used to determine whether any data cleaning routines should be implemented (e.g. remove negative prices or outliers).
* **Build ML data.** Crystalvalue takes in a dataset (e.g. data from Google Analytics or a CRM transaction dataset) and outputs an ML ready dataset. For model training, this consists of a set of features and targets. For prediciton, this is just a set of features. See Figure 2 to see what this looks like in practice. The same customer can be observed multiple times during different time windows for model training using monthly ‘sliding windows’. Crystalvalue can automatically detect data types from the input data and apply the appropriate feature transformations. See Figure 3 for the typical features and transformations that can be used in LTV models. Crystalvalue will take all the data present in the input table and create features (unless the columns are included in the `ignore_features` argument. The processing is programmatically executed in the python Bigquery API under-the-hood for efficiency. This step also creates a column which assigns customers to a training, validation and test set (split with random 15% of users as test, 15% in validation and 70% in training). Crystalvalue sets recommended default parameters which can be configured:
  *  days_lookback: # days to look back to create features (default: 365 days).
  *  days_lookahead: # days to look ahead to predict value (default: 365 days).
  *  features: Input data columns types (default: None, detected automatically).
  *  customer_id_column: The customer id column.
  *  date_column: The transaction date or timestamp column.
  *  value_column: The value column (i.e. transaction revenue/profit).
  *  ignore_columns: Columns to ignore in this step.

  **Figure 2**

  ![Crystalval](https://screenshot.googleplex.com/645o76szJkYPVZg.png)


  **Figure 3**

  ![Crystalval](https://screenshot.googleplex.com/64VJyTq9WiU6Fpp.png)

* **Model Training**. Crystalvalue will programmatically train a Vertex AI Tabular AutoML model which will then be visible in the [Vertex AI dashboard](https://console.cloud.google.com/vertex-ai).


* **Model evaluation.** Crystalvalue will run model evaluation on the test data (a set of customers that were not included in the model training) and report the following metrics:
  * Bin level Charts - predicted vs actual LTV by decile
  * Spearman Correlation
  * Normalized Gini coefficient

* **Customer insights** (WIP) Crystalvalue will connect your trained model to the What IF tool (see an [example here](https://pair-code.github.io/what-if-tool/demos/age.html)) which can help you understand who your high LTV customers are.

* **Predictions & Scheduling** Crystalvalue will provide you the option to schedule your predictions using the model on a regular basis. See the example in the demo notebook which schedules predictions for 1am everyday.


## Prerequisites

CrystalValue is designed to be deployed on Google Cloud Platform, so
the following functionalities and access are required:

*   BigQuery
*   Vertex AI API
*   (Optional for scheduling) Cloud Storage, Container Registry.

## The team

The team working on this is:

*  dfkelly@
*  dulacp@
*  pduque@
*  sumedhamenon@



# credit-admission-prediction
# Credit Card Approval Prediction
<p align="left">
    <a alt="EDA">
        <img src="https://img.shields.io/badge/%20-EDA%20-orange" /></a>
    <a alt="Classification">
        <img src="https://img.shields.io/badge/%20-Classification%20-orange" /></a>
</p>

## General info

#### Problem Statement
**Credit score cards** are a common risk control method in the financial industry. It uses personal information and data submitted by credit card applicants to predict the probability of future defaults and credit card borrowings. The bank is able to **decide whether to issue a credit card to the applicant**. Credit scores can objectively quantify the magnitude of risk.

Aim of this project is to fit a binary Classification model that would predict whether given applicant would pay his debt and consequently decide whether credit card should be granted.

Dataset comes from the **kaggle** platform.

## Technologies

<p align="left">
    <a alt="Jupyter Notebook">
        <img src="https://img.shields.io/badge/%20-Jupyter%20Notebook%20-blue" /></a>
    <a alt="python">
        <img src="https://img.shields.io/badge/%20-python%20-blue" /></a>
</p>

Additionally pre-commit add-ins were used:
<p align="left">
    <a alt="flake8">
        <img src="https://img.shields.io/badge/%20-flake8%20-steelblue" /></a>
    <a alt="isort">
        <img src="https://img.shields.io/badge/%20-isort%20-steelblue" /></a>
    <a alt="interrogate">
        <img src="https://img.shields.io/badge/%20-interrogate%20-steelblue" /></a>
</p>

## References

The datasets [Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) were downloaded from the Kaggle platform. Data used:
* application_record.csv
* credit_record.csv



--------
## How to run it
Run the following command in the bash terminal:

```zsh
python ./main.py
```
The following command would:
* load, aggregate and merge raw data
* classify customers (target creation)
* create new features
* preprocess data, e.g. add PCA components, scale, encode features, deal with imbalanced data
* train and save the model
* evaluate model performance

## Developer requirements

1. python
2. bash
3. conda
________________

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── env.yml            <- File to create environment with list of all necessary packages
    │
    ├── main.py            <- Script with all steps required to retain the classification model
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── params.py          <- Keeps all parameters required to point to the data, preprocess them
    │                         and train the model. Parameters are separated from the code, all stored in one place.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    |   │   ├── udfs_data_prep.py
    │   │   └── data_gathering.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── udfs_modelling.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── utils          <- Scripts with utilities
    │   │   └── utils.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── udfs_visualization
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



## Contributors

jw-zima

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

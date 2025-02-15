Tech DS test
==============================

Predicting customer subscription

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models for predict and data preprocess.
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── pyproject.toml   <- The requirements file for reproducing the analysis environment
    │
    ├── poetry.lock   <- The requirements file for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or preprocess data
    │   │   ├── preprocess_.py
    │   │   └── preprocess.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── train_.py
    │   │   └── train.py
    │   │
    │   ├── server.py      <- Build uvicorn server
    │   │
    │   └── request.py     <- Test request to uvicorn server
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------
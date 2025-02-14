import pickle

import click
import pandas as pd
from catboost import CatBoostClassifier

from src.model.train_ import CatboostCVOptuna

CAT_FEATS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
    "campaign",
    "previous",
    "age_bins",
    "duration_bins",
]
METRIC = "F:beta=2"


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("model_filepath", type=click.Path())
def train(
    input_filepath: str,
    model_filepath: str,
):
    # Download train dataset
    df = pd.read_pickle(input_filepath)

    # Create Pool with X and y
    opt_model = CatboostCVOptuna(cat_feats=CAT_FEATS, metric=METRIC)
    params = opt_model.run(df.drop("y", axis=1), df["y"])

    print(params)

    # Fit final model
    cb_model = CatBoostClassifier(**params)
    cb_model.fit(df.drop("y", axis=1), df["y"])

    # Save model
    with open(model_filepath, "wb") as f:
        pickle.dump(cb_model, f)


if __name__ == "__main__":
    train()

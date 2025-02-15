import pickle

import click
import pandas as pd


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("preprocess_model_filepath", type=click.Path())
@click.argument("model_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def predict(
    input_filepath: str,
    preprocess_model_filepath: str,
    model_filepath: str,
    output_filepath: str,
):
    # Download test dataset
    df = pd.read_excel(input_filepath)

    # Download models
    with open(preprocess_model_filepath, "rb") as f:
        preprocess_model = pickle.load(f)

    with open(model_filepath, "rb") as f:
        model = pickle.load(f)

    # Run pipeline
    df = preprocess_model.transform(df, with_target=False)
    df = pd.Series(model.predict_proba(df)[:, 1])

    # Save predictions
    df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    predict()

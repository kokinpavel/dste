import click
import pickle

import pandas as pd

from src.data.preprocess_ import PreprocessModel

COLS_TO_BIN = ['age', 'duration']
BINS = 20
MIN_COUNT = 20
FILL_VALUE = 999


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("model_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def preprocess(
    input_filepath: str,
    model_filepath: str,
    output_filepath: str,
):
    
    # Download train dataset
    df = pd.read_excel(input_filepath)

    # Run pipeline
    preprocess_model = PreprocessModel()
    df = preprocess_model.fit_transform(df)

    # Save model
    with open(model_filepath, 'wb') as f:
        pickle.dump(preprocess_model, f)

    # Save data
    df.to_pickle(output_filepath)

if __name__ == "__main__":
    preprocess()

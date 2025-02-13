import click
import dotenv
from src.utils.database import read_data_from_db, redshift_connect
from src.utils.S3 import S3Manager
from src.utils.time_transform import create_inference_day

# It can be start of inference
INFERENCE_DAY = create_inference_day(8)


@click.command()
@click.argument("input_sql_query_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
@click.argument("mode", type=str, default="local")
def import_raw_data(input_sql_query_filepath: str, output_filepath: str, mode: str):

    with open(input_sql_query_filepath) as f:
        sql_query = f.read()

    query = sql_query.format(
            inference_date=INFERENCE_DAY
        )

    df = read_data_from_db(engine, query)

    # Save
    print(f"raw_data shape is {df.shape}")
    df.to_pickle(output_filepath)

    # Save S3
    if mode == "S3":
        S3Manager().write(output_filepath, "data")


if __name__ == "__main__":

    if dotenv.load_dotenv():
        print("Enviroment is correct")
    engine = redshift_connect()
    import_raw_data()

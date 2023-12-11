import yaml
import logging
import pandas as pd
import os

def summarize_file(df: pd.DataFrame, file_path: str) -> None:
    """Prints a summary of a data file including total rows, columns, and file size in MB.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        file_path (str): File path
    """

    # filesize in mb
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    # get dimensions
    total_rows = len(df)
    total_columns = len(df.columns)

    print(f"Total number of rows: {total_rows}")
    print(f"Total number of columns: {total_columns}")
    print(f"File size: {file_size_mb:.2f} MB")


def read_config_file(filepath: str) -> dict:
    """Reads a YAML file for data ingestion.

    Args:
        filepath (str): YAML file path

    Returns:
        dict: YAML data
    """

    with open(filepath, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)


def num_col_validation(df: pd.DataFrame, table_config: dict) -> bool:
    """Validates if the number of columns in the DataFrame matches the table configuration.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        table_config (dict): Validation data

    Returns:
        bool: If number of columns match the validation data
    """

    if len(df.columns) == len(table_config["columns"]):
        return True
    else:
        return False


def col_header_val(df: pd.DataFrame, table_config: dict) -> bool:
    """Validates if the header names in the DataFrame match the table configuration.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        table_config (dict): Validation data

    Returns:
        bool: If column headers match the validation data
    """

    # sort, strip leading and trailing spaces, and replace space with _
    df_columns = sorted([col.strip().lower().replace(" ", "_") for col in df.columns])
    yaml_columns = sorted(
        [col.strip().lower().replace(" ", "_") for col in table_config["columns"]]
    )

    if df_columns == yaml_columns:
        return True
    else:
        # find the mismatched columns
        mismatched_columns = set(df_columns) ^ set(yaml_columns)
        print(f"Mismatched columns: {list(mismatched_columns)}")
        return False
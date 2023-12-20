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


def num_col_validation(df: pd.DataFrame, expected_columns: list) -> bool:
    """
    Validates if the number of columns in the DataFrame matches the expected number of columns.
    Prints the missing or extra columns if there's a discrepancy.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        expected_columns (list): List of expected column names

    Returns:
        bool: True if the number of columns in the DataFrame matches the expected number, else False
    """

    actual_columns = df.columns.tolist()
    expected_columns_set = set(expected_columns)
    actual_columns_set = set(actual_columns)

    if len(actual_columns) == len(expected_columns):
        print("Number of columns match!")
        return True
    else:
        missing_columns = expected_columns_set - actual_columns_set
        extra_columns = actual_columns_set - expected_columns_set
        if missing_columns:
            print(f"Missing columns: {list(missing_columns)}")
        if extra_columns:
            print(f"Extra columns: {list(extra_columns)}")
        return False


def col_header_validation(df: pd.DataFrame, expected_columns: list) -> bool:
    """
    Validates if the header names in the DataFrame match the expected column names.
    Column names are compared after sorting, stripping leading/trailing spaces, and replacing spaces with underscores.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        expected_columns (list): List of expected column names

    Returns:
        bool: True if column headers match the expected column names, else False
    """

    df_columns = sorted([col.strip().lower().replace(" ", "_") for col in df.columns])
    expected_columns_formatted = sorted(
        [col.strip().lower().replace(" ", "_") for col in expected_columns]
    )

    if df_columns == expected_columns_formatted:
        print("Column headers match!")
        return True
    else:
        # find the mismatched columns
        mismatched_columns = set(df_columns) ^ set(expected_columns_formatted)
        print(f"Mismatched columns: {list(mismatched_columns)}")
        return False

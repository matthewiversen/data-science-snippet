import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro  # normality test
from sklearn.impute import SimpleImputer  # used for mean/median imputing
import re  # regular expressions
import yaml  # data ingestion validation
import logging


def factorize_df(df: pd.DataFrame) -> None:
    """
    Encodes all object features in the DataFrame.
    """
    for colname in df.select_dtypes("object"):
        df[colname], _ = df[colname].factorize()


def summary(df: pd.DataFrame, file_path: str) -> None:
    """
    Prints a summary of a data file including total rows, columns, and file size in MB.
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


def show_nan_all_columns(df: pd.DataFrame) -> None:
    """
    Prints the number of NaNs for each column of the DataFrame.
    """
    nan_counts = df.isnull().sum().sort_values(ascending=False)
    print(f"NaN Counts:\n{nan_counts}")


def show_nan_columns(df: pd.DataFrame) -> None:
    """
    Prints the number of NaNs in columns that have NaNs.
    """
    nan_counts = df.isnull().sum().sort_values(ascending=False)
    nan_counts = nan_counts[nan_counts > 0]
    print(f"NaN Counts:\n{nan_counts}")


def find_nan_columns(df: pd.DataFrame) -> pd.Index:
    """
    Returns the columns that have NaN values.
    """
    nan_features = df.isnull().sum()
    non_zero_nans = nan_features[nan_features > 0]
    return non_zero_nans.index


def check_for_normality(df: pd.DataFrame, features: list[str]) -> None:
    """
    Checks for normality in given features, useful in deciding how to impute.
    """
    for feature in features:
        p_value = shapiro(df[feature]).pvalue

        if p_value > 0.05:
            print(f"{feature} is normally distributed (p-value > 0.05)")
        else:
            print(f"{feature} is not normally distributed (p-value <= 0.05)")


def impute_with_mean(df: pd.DataFrame, features_to_impute: list[str]) -> None:
    """
    Imputes given features with the mean value.
    """
    imputer = SimpleImputer(strategy="mean")
    df[features_to_impute] = imputer.fit_transform(df[features_to_impute])


def impute_with_median(df: pd.DataFrame, features_to_impute: list[str]) -> None:
    """
    Imputes given features with the median value.
    """
    imputer = SimpleImputer(strategy="median")
    df[features_to_impute] = imputer.fit_transform(df[features_to_impute])


def clean_text(text: str) -> str:
    """
    Removes any extra spaces or special characters from text.
    """
    # remove anything other than letter, number, and spaces then make lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower()

    # remove extra spaces and leading/trailing spaces
    text = " ".join(text.split())

    return text


def read_config_file(filepath: str):
    """
    Reads a YAML file for data ingestion.
    """
    with open(filepath, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)


def num_col_validation(df: pd.DataFrame, table_config: dict) -> bool:
    """
    Validates if the number of columns in the DataFrame matches the table configuration.
    """
    if len(df.columns) == len(table_config["columns"]):
        return True
    else:
        return False


def col_header_val(df: pd.DataFrame, table_config: dict) -> bool:
    """
    Validates if the header names in the DataFrame match the table configuration.
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


def set_pd_max_columns(max_columns: int | None) -> None:
    """
    Changes the number of columns seen on pandas DataFrame output.
    """
    pd.set_option("display.max_columns", max_columns)


def set_pd_max_rows(max_rows: int | None) -> None:
    """
    Changes the number of rows seen on pandas DataFrame output.
    """
    pd.set_option("display.max_rows", max_rows)

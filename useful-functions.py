import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro  # normality test
from sklearn.impute import SimpleImputer  # used for mean/median imputing
import re  # regular expressions
import yaml  # data ingestion validation
import logging
import difflib  # used to compare strings


def show_spelling_errors(
    df: pd.DataFrame, similarity_threshold: float, exclude_columns: list[str]
) -> None:
    """This prints all of the observations in a column that are similar above a threshold

    Args:
        df (pd.DataFrame): Pandas DataFrame
        similarity_threshold (float): Decimal of how similar of results we want to see (0.0-1.0)
        exclude_columns (list[str]): List of columns you want to exclude from spelling check
    """

    spelling_errors = {}

    if exclude_columns is None:
        exclude_columns = []

    # find potential spelling errors for object columns
    for column in df.select_dtypes(include="object"):
        if column not in exclude_columns:
            unique_values = df[column].dropna().unique()
            potential_errors = []

            for i, value1 in enumerate(unique_values):
                for value2 in unique_values[i + 1 :]:
                    similarity = difflib.SequenceMatcher(None, value1, value2).ratio()
                    if similarity > similarity_threshold:
                        potential_errors.append((value1, value2))

            if potential_errors:
                spelling_errors[column] = potential_errors

    # print the errors
    for column, errors in spelling_errors.items():
        print(f"Potential spelling errors in column '{column}':")
        for error in errors:
            print(f"- '{error[0]}' might be similar to '{error[1]}'")


def detect_outliers_iqr(data: pd.DataFrame) -> pd.DataFrame:
    """Detects and returns any outliers for a given dataframe.

    Args:
        data (pd.DataFrame): Pandas DataFrame

    Returns:
        pd.DataFrame: Pandas DataFrame with outliers only
    """

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # filter for outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return outliers


def clean_text(text: str) -> str:
    """Removes any extra spaces or special characters from text.

    Args:
        text (str): Text that you want to filer

    Returns:
        str: The cleaned text
    """

    # remove anything other than letter, number, and spaces then make lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower()

    # remove extra spaces and leading/trailing spaces
    text = " ".join(text.split())

    return text


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Prints info about and removes duplicate columns and rows

    Args:
        df (pd.DataFrame): Incoming Pandas DataFrame

    Returns:
        pd.DataFrame: Pandas DataFrame with no duplicate rows/columns
    """

    # count and remove duplicate rows
    duplicate_rows = df[df.duplicated()]
    num_duplicate_rows = len(duplicate_rows)
    df = df.drop_duplicates()

    # count and remove duplicate columns
    duplicate_columns = df.columns[df.columns.duplicated()]
    num_duplicate_columns = len(duplicate_columns)
    df = df.loc[:, ~df.columns.duplicated()]

    print(f"Number of duplicate rows removed: {num_duplicate_rows}")
    print(f"Number of duplicate columns removed: {num_duplicate_columns}")

    return df


def remove_high_cardinality_cols(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Remove columns with a cardinality higher than the given threshold.

    Args:
        df (pd.DataFrame): Pandas DataFrame to update
        threshold (int): Threshold of cardinality
    """

    # Identify columns that exceed the threshold
    high_cardinality_cols = df.nunique()[df.nunique() > threshold].index

    # Print the dropped columns
    print(
        len(high_cardinality_cols),
        "features/columns dropped due to high cardinality:",
        high_cardinality_cols.values,
    )

    # Drop these columns from the dataframe
    return df.drop(high_cardinality_cols, axis=1)


def remove_high_nan_cols(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Remove columns with a NaN count higher than the given threshold.

    Args:
        df (pd.DataFrame): Pandas DataFrame to update
        threshold (int): Threshold of NaN count
    """

    high_nan_cols = pd.isnull(df).sum()[pd.isnull(df).sum() > threshold].index

    print(len(high_nan_cols), "features/columns dropped:", high_nan_cols.values)

    return df.drop(high_nan_cols, axis=1)


def factorize_columns(
    df: pd.DataFrame, columns: list
) -> (pd.DataFrame, list[str], dict):
    """
    Factorize specified categorical columns in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to factorize.

    Returns:
        pd.DataFrame: A new DataFrame with factorized columns.
        list: A list of column names that were successfully factorized.
        dict: A dictionary containing mappings of original values to factorized values for each specified column.
    """

    df_copy = df.copy()
    mappings = {}
    factorized_columns = []

    for col in columns:
        if df_copy[col].dtype == "object":
            factorized_columns.append(col)
            df_copy[col], mapping = pd.factorize(df_copy[col])
            mappings[col] = mapping
            print(f"{col} factorized.")

    return df_copy, factorized_columns, mappings


def reverse_factorize_columns(
    df: pd.DataFrame, columns: list, mappings: dict
) -> pd.DataFrame:
    """
    Reverse the factorization of specified columns in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with factorized columns.
        columns (list): A list of column names to reverse factorize.
        mappings (dict): A dictionary containing mappings of original values to factorized values for each specified column.

    Returns:
        pd.DataFrame: A new DataFrame with reversed factorized columns.
    """

    df_copy = df.copy()

    for col in columns:
        df_copy[col] = mappings[col].take(df_copy[col])
        print(f"{col} unfactorized.")

    return df_copy


def standardize_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Standardizes selected columns of the given dataframe to a mean of 0 and SD of 1.

    Args:
        df (pd.DataFrame): Pandas Dataframe.
        columns (list): List of column names to standardize.

    Returns:
        pd.DataFrame: Pandas DataFrame with selected columns standardized.
    """

    df_copy = df.copy()

    mean = df_copy[columns].mean(axis=0)
    std = df_copy[columns].std(axis=0)

    # Avoid division by zero
    epsilon = 1e-8
    std[std < epsilon] = epsilon

    df_copy[columns] = (df_copy[columns] - mean) / std

    return df_copy


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


def print_all_nan_counts(df: pd.DataFrame) -> None:
    """Prints the number of NaNs for each column of the DataFrame.

    Args:
        df (pd.DataFrame): Pandas DataFrame
    """

    nan_counts = df.isnull().sum().sort_values(ascending=False)
    print(f"NaN Counts:\n{nan_counts}")


def print_nan_cols(df: pd.DataFrame) -> None:
    """Prints the number of NaNs in columns that have NaNs.

    Args:
        df (pd.DataFrame): Pandas DataFrame
    """

    nan_counts = df.isnull().sum().sort_values(ascending=False)
    nan_counts = nan_counts[nan_counts > 0]
    print(f"NaN Counts:\n{nan_counts}")


def find_nan_columns(df: pd.DataFrame) -> pd.Index:
    """Returns the columns that have NaN values.

    Args:
        df (pd.DataFrame): Pandas DataFrame

    Returns:
        pd.Index: Pandas Index of NaN columns
    """

    nan_features = df.isnull().sum()
    non_zero_nans = nan_features[nan_features > 0]

    return non_zero_nans.index


def check_for_normality(df: pd.DataFrame, features: list[str], pvalue: float) -> None:
    """Checks for normality in given features, useful in deciding how to impute.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        features (list[str]): Features in DataFrame to check
        pvalue (float): P-Value threshold for normality check
    """

    for feature in features:
        p_value = shapiro(df[feature]).pvalue

        if p_value > 0.05:
            print(f"{feature} is normally distributed (p-value > 0.05)")
        else:
            print(f"{feature} is not normally distributed (p-value <= 0.05)")


def impute_with_mean(df: pd.DataFrame, features_to_impute: list[str]) -> None:
    """Imputes given features with the mean value.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        features_to_impute (list[str]): Features to impute with the mean of that feature
    """

    imputer = SimpleImputer(strategy="mean")
    df[features_to_impute] = imputer.fit_transform(df[features_to_impute])


def impute_with_median(df: pd.DataFrame, features_to_impute: list[str]) -> None:
    """Imputes given features with the median value.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        features_to_impute (list[str]): Features to impute with the median of that feature
    """

    imputer = SimpleImputer(strategy="median")
    df[features_to_impute] = imputer.fit_transform(df[features_to_impute])


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


def set_pd_max_columns(max_columns: int | None) -> None:
    """Changes the number of columns seen on pandas DataFrame output.

    Args:
        max_columns (int | None): maximum number columns to set
    """

    pd.set_option("display.max_columns", max_columns)


def set_pd_max_rows(max_rows: int | None) -> None:
    """Changes the number of rows seen on pandas DataFrame output.

    Args:
        max_rows (int | None): maximum number of rows to set
    """

    pd.set_option("display.max_rows", max_rows)

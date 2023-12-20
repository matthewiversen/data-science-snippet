import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def min_max_scale_columns(df: pd.DataFrame, columns: list) -> (pd.DataFrame, dict):
    """
    Min-Max scale specified columns in a Pandas DataFrame using sklearn's MinMaxScaler.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to Min-Max scale.

    Returns:
        pd.DataFrame: A new DataFrame with Min-Max scaled columns.
        dict: A dictionary containing MinMaxScaler instances fitted on the specified columns.
    """
    df_copy = df.copy()
    scalers = {}

    for col in columns:
        if col in df_copy.columns:
            scaler = MinMaxScaler()
            df_copy[col] = scaler.fit_transform(df_copy[[col]])
            scalers[col] = scaler
            print(f"{col} Min-Max scaled.")
        else:
            print(f"{col} not found in dataframe")

    return df_copy, scalers


def reverse_min_max_scale_columns(
    df: pd.DataFrame, columns: list, scalers: dict
) -> pd.DataFrame:
    """
    Reverse the Min-Max scaling of specified columns in a Pandas DataFrame using sklearn's MinMaxScaler.

    Args:
        df (pd.DataFrame): The input DataFrame with Min-Max scaled columns.
        columns (list): A list of column names to reverse Min-Max scale.
        scalers (dict): A dictionary containing MinMaxScaler instances fitted on the specified columns.

    Returns:
        pd.DataFrame: A new DataFrame with reversed Min-Max scaled columns.
    """
    df_copy = df.copy()

    for col in columns:
        if col in df_copy.columns and col in scalers:
            scaler = scalers[col]
            df_copy[col] = scaler.inverse_transform(df_copy[[col]])
            print(f"{col} un-Min-Max scaled.")
        else:
            print(f"{col} not found in dataframe or no scaler available")

    return df_copy


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

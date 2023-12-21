import pandas as pd


def get_object_cols(df: pd.DataFrame) -> list:
    """
    Get a list of column names that have 'object' or categorical data type in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        list: A list of column names containing 'object' or categorical data type.
    """

    object_columns = []

    for col in df.columns:
        if df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
            object_columns.append(col)

    return object_columns


def get_numerical_cols(df: pd.DataFrame) -> list:
    """
    Get a list of column names that have numerical data type in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        list: A list of column names containing numerical data type.
    """

    numerical_columns = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_columns.append(col)

    return numerical_columns


def get_object_bool_mask(df: pd.DataFrame) -> list[bool]:
    """
    Determine whether each column in the DataFrame is discrete or continuous.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        list[bool]: A list of Boolean values, where True indicates a discrete feature,
            and False indicates a continuous feature.
    """

    discrete_features = [dtype == "object" for dtype in df.dtypes]

    return discrete_features

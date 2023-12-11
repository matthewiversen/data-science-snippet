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
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
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

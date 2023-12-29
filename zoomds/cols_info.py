import pandas as pd


def get_cat_num_cols(df: pd.DataFrame) -> (list[str], list[str]):
    """
    Get lists of column names for categorical and numerical data types in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        tuple:
            - First list contains names of categorical columns ('object', 'category' data types).
            - Second list contains names of numerical columns ('int64', 'float64', 'uint8' data types).
    """

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    numerical_cols = df.select_dtypes(include=["int64", "float64", "uint8"]).columns

    return categorical_cols, numerical_cols


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

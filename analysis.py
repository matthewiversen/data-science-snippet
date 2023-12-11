import pandas as pd

def object_cols_distribution(
    df: pd.DataFrame, object_cols: list[str], exclude_cols: list[str] = []
) -> None:
    """
    Print the percentage distribution of values in categorical columns.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        object_cols (list): A list of column names with categorical data.
        exclude_cols (list, optional): A list of columns to exclude from analysis.

    Returns:
        None
    """
    
    for col in object_cols:
        if col not in exclude_cols:
            category_percentage = (df[col].value_counts() / len(df)) * 100
            print(category_percentage)
            print("=" * 50)  # Separator for readability
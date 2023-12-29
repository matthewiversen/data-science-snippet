import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def print_mi_scores(
    features: pd.DataFrame,
    target: pd.DataFrame,
    object_cols_mask: list[bool],
    n_scores: int = None,
    ignore_cols: list[str] = None,
):
    """
    Calculate and print Mutual Information (MI) Scores for feature selection, excluding specified columns.

    Args:
        features (pd.DataFrame): DataFrame containing the features.
        target (pd.DataFrame): DataFrame containing the target variable.
        object_cols_mask (list[bool]): List indicating if each feature is discrete (True) or continuous (False).
        n_scores (int, optional): Number of top scores to print. Defaults to printing all scores.
        ignore_cols (list[str], optional): List of column names to ignore in the calculation.

    Returns:
        None
    """

    if ignore_cols is not None:
        features = features.drop(columns=ignore_cols, errors="ignore")
        object_cols_mask = [
            col
            for col, mask in zip(features.columns, object_cols_mask)
            if col not in ignore_cols
        ]

    mi_scores = mutual_info_regression(
        features, target.squeeze(), discrete_features=object_cols_mask, random_state=42
    )
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    if n_scores is not None:
        print(mi_scores.head(n_scores))
    else:
        print(mi_scores)


def get_mi_scores(
    features: pd.DataFrame,
    target: pd.DataFrame,
    object_cols_mask: list[bool],
    n_scores: int = None,
    ignore_cols: list[str] = None,
) -> pd.Series:
    """
    Calculate and return top Mutual Information (MI) Scores for feature selection, excluding specified columns.

    Args:
        features (pd.DataFrame): DataFrame containing the features.
        target (pd.DataFrame): DataFrame containing the target variable.
        object_cols_mask (list[bool]): List indicating if each feature is discrete (True) or continuous (False).
        n_scores (int, optional): Number of top scores to return. If not specified, returns all scores.
        ignore_cols (list[str], optional): List of column names to ignore in the calculation.

    Returns:
        pd.Series: Series containing the top n MI Scores for each feature, sorted in descending order.
    """

    # Exclude ignored columns if provided
    if ignore_cols is not None:
        features = features.drop(columns=ignore_cols, errors="ignore")
        object_cols_mask = [
            mask
            for col, mask in zip(features.columns, object_cols_mask)
            if col not in ignore_cols
        ]

    mi_scores = mutual_info_regression(
        features, target.squeeze(), discrete_features=object_cols_mask, random_state=42
    )
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
    sorted_mi_scores = mi_scores.sort_values(ascending=False)

    if n_scores is not None and n_scores < len(sorted_mi_scores):
        return sorted_mi_scores.head(n_scores)
    else:
        return sorted_mi_scores

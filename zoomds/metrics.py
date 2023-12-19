import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def print_mi_scores(features: pd.DataFrame, target: pd.DataFrame, object_cols: list[bool], n_scores: int = None):
    """Calculate and print Mutual Information (MI) Scores for feature selection.

    This function calculates MI scores between each feature in 'features' and the 'target', 
    then prints the top 'n_scores' MI scores. If 'n_scores' is not specified, all scores are printed.

    Args:
        features (pd.DataFrame): DataFrame containing the features.
        target (pd.DataFrame): DataFrame containing the target variable.
        object_cols (list[bool]): List indicating if each feature is discrete (True) or continuous (False).
        n_scores (int, optional): Number of top scores to print. Defaults to printing all scores.

    Returns:
        None
    """
    
    mi_scores = mutual_info_regression(features, target, discrete_features=object_cols)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    if n_scores is not None:
        print(mi_scores.head(n_scores))
    else:
        print(mi_scores)


def get_mi_scores(features: pd.DataFrame, target: pd.DataFrame, object_cols: list[bool], n_scores: int = None) -> pd.Series:
    """Calculate and return top Mutual Information (MI) Scores for feature selection.

    This function calculates MI scores between each feature in 'features' and the 'target',
    then returns a Pandas Series of the top 'n_scores' MI scores. 
    If 'n_scores' is not specified, or if it exceeds the number of features, all scores are returned.

    Args:
        features (pd.DataFrame): DataFrame containing the features.
        target (pd.DataFrame): DataFrame containing the target variable.
        object_cols (list[bool]): List indicating if each feature is discrete (True) or continuous (False).
        n_scores (int, optional): Number of top scores to return. If not specified, returns all scores.

    Returns:
        pd.Series: Series containing the top n MI Scores for each feature, sorted in descending order.
    """
    
    mi_scores = mutual_info_regression(features, target, discrete_features=object_cols)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
    sorted_mi_scores = mi_scores.sort_values(ascending=False)

    if n_scores is not None and n_scores < len(sorted_mi_scores):
        return sorted_mi_scores.head(n_scores)
    else:
        return sorted_mi_scores
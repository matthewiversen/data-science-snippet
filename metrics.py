import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def get_mi_scores(features: pd.DataFrame, target: pd.DataFrame, discrete_features: list[bool]) -> pd.Series:
    """Calculate Mutual Information (MI) Scores for feature selection.

    Args:
        features (pd.DataFrame): The DataFrame containing the features.
        target (pd.DataFrame): The DataFrame containing the target variable.
        discrete_features (list[bool]): A list of Boolean values indicating whether each feature
            is discrete (True) or continuous (False).

    Returns:
        pd.Series: A Series containing the MI Scores for each feature, sorted in descending order.

    Source: https://www.kaggle.com/code/ryanholbrook/principal-component-analysis
    """
    
    mi_scores = mutual_info_regression(features, target, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores
import pandas as pd
from scipy.stats import shapiro  # normality test
from sklearn.impute import SimpleImputer  # used for mean/median/mode imputing


def detect_outliers_iqr_columns(data: pd.DataFrame, columns: list) -> dict:
    """Detects and returns outliers for specified columns in a DataFrame.

    Args:
        data (pd.DataFrame): Pandas DataFrame.
        columns (list): List of column names to detect outliers.

    Returns:
        dict: A dictionary where each key is a column name and its value is a DataFrame of outliers and their indices for that column.
    """

    outliers_dict = {}

    for col in columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_condition = (data[col] < lower_bound) | (data[col] > upper_bound)
            outliers = data[col][outlier_condition]
            outliers_dict[col] = pd.DataFrame(
                {"Index": outliers.index, "Outlier": outliers.values}
            )

    return outliers_dict


def check_for_normality(
    df: pd.DataFrame, features: list[str], pvalue_threshold: float
) -> None:
    """Checks for normality in given features, useful in deciding how to impute.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        features (list[str]): Features in DataFrame to check
        pvalue (float): P-Value threshold for normality check
    """

    for feature in features:
        p_value = shapiro(df[feature]).pvalue

        if p_value > pvalue_threshold:
            print(f"{feature} is normally distributed (p-value > {pvalue_threshold})")
        else:
            print(
                f"{feature} is not normally distributed (p-value <= {pvalue_threshold})"
            )


def impute_features(
    df: pd.DataFrame, features_to_impute: list[str], strat: str
) -> pd.DataFrame:
    """Imputes given features with the median value and returns a copy of the DataFrame.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        features_to_impute (list[str]): Features to impute with the median of that feature

    Returns:
        pd.DataFrame: A new DataFrame with imputed values, leaving the original DataFrame unchanged.
    """

    df_copy = df.copy()

    imputer = SimpleImputer(strategy=strat)
    df_copy[features_to_impute] = imputer.fit_transform(df_copy[features_to_impute])

    return df_copy

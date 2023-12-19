import pandas as pd
import difflib  # used to compare strings
import re  # regular expressions

# ! --------------------------Column and Row Cleaning--------------------------

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


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Prints info about and removes duplicate rows.

    Args:
        df (pd.DataFrame): Incoming Pandas DataFrame

    Returns:
        pd.DataFrame: DataFrame with no duplicate rows
    """

    duplicate_rows = df[df.duplicated()]
    num_duplicate_rows = len(duplicate_rows)
    df = df.drop_duplicates()

    if num_duplicate_rows > 0:
        print(f"{num_duplicate_rows} rows removed: {duplicate_rows}")
    else:
        print("No duplicate rows found.")
        
    return df


def remove_duplicate_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Prints info about and removes duplicate columns.

    Args:
        df (pd.DataFrame): Incoming Pandas DataFrame

    Returns:
        pd.DataFrame: DataFrame with no duplicate columns
    """

    duplicate_columns = df.columns[df.columns.duplicated()]
    num_duplicate_columns = len(duplicate_columns)
    df = df.loc[:, ~df.columns.duplicated()]

    if num_duplicate_columns > 0:
        print(f"{num_duplicate_columns} duplicate columns removed: {duplicate_columns.tolist()}")
    else:
        print("No duplicate columns found.")

    return df


# ! --------------------------String/Object Cleaning---------------------------

def show_spelling_errors(
    df: pd.DataFrame, similarity_threshold: float, exclude_columns: list[str] = []
) -> None:
    """This prints all of the observations in a column that are similar above a threshold

    Args:
        df (pd.DataFrame): Pandas DataFrame
        similarity_threshold (float): Decimal of how similar of results we want to see (0.0-1.0)
        exclude_columns (list[str]): List of columns you want to exclude from spelling check
    """

    spelling_errors = {}

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


def remove_strings_from_cols(
    df: pd.DataFrame, columns_to_clean: list[str], strings_to_remove: list[str]
) -> pd.DataFrame:
    """
    Removes specified strings from specified columns in a Pandas DataFrame and returns a copy of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_clean (list[str]): A list of column names to clean.
        strings_to_remove (list[str]): A list of strings to remove from the specified columns.

    Returns:
        pd.DataFrame: A new DataFrame with the specified strings removed from the specified columns, leaving the original DataFrame unchanged.
    """

    df_copy = df.copy()

    for col in columns_to_clean:
        for string in strings_to_remove:
            df_copy[col] = df_copy[col].apply(lambda x: re.sub(string, "", x).strip())

    return df_copy


def alpha_numeric_text_clean(text: str) -> str:
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


# ! -------------------------------NaN Cleaning--------------------------------

def remove_high_nan_cols(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Remove columns with a NaN count higher than the given threshold.

    Args:
        df (pd.DataFrame): Pandas DataFrame to update
        threshold (int): Threshold of NaN count
    """

    high_nan_cols = pd.isnull(df).sum()[pd.isnull(df).sum() > threshold].index

    print(len(high_nan_cols), "features/columns dropped:", high_nan_cols.values)

    return df.drop(high_nan_cols, axis=1)


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

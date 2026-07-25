from datetime import datetime
import pandas as pd

def name_model(name: str) -> str:
    """
    Append the current date and time to a given name to produce a unique,
    timestamped identifier (e.g. for naming saved models or log entries).

    Args:
        name (str): The base name to which the timestamp will be appended.

    Returns:
        str: The original name with a timestamp suffix in the format
        "name_YYYYMMDD_HHMMSS". This format is filesystem-safe (no colons
        or slashes) and sorts chronologically as a plain string.

    Example:
        >>> name_model("rmse_model")
        'rmse_model_20260724_153042'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}"

def train_test_split(df: pd.DataFrame):
    """
    Split a DataFrame into training and test sets based on a 'date' column.

    Training data: all rows with dates before 1 Jan 2021.
    Test data: all rows with dates from 1 Jan 2021 to 31 Dec 2025 (inclusive).
    Rows with dates after 31 Dec 2025 are excluded from both sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'date' column with dates in
        'YYYY-MM-DD' format (as strings or already datetime-like).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (train, test) DataFrames.
    """
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    test = df[(df['date'] >= '2021-01-01') & (df['date'] <= '2025-12-31')]

    train = df[df['date'] < '2021-01-01']

    return train, test
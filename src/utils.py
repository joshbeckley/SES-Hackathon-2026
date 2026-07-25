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
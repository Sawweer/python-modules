import pandas as pd


def summary(df, exclude=None):
    """
    Compute comprehensive statistics for all variables in a DataFrame

    Parameters:
    df : pandas DataFrame
    exclude : list or None, column names to exclude from analysis

    Returns:
    pandas DataFrame with statistics for each column
    """
    if exclude is None:
        exclude = []

    stats_list = []

    for col in df.columns:
        if col in exclude:
            continue
        stats = {}
        stats["variable"] = col

        # Count and percentage of nulls
        stats["nbr.null"] = df[col].isna().sum()
        stats["%null"] = (stats["nbr.null"] / len(df)) * 100

        # Check if column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        if is_numeric:
            # Count and percentage of zeros
            stats["nbr.zero"] = (df[col] == 0).sum()
            stats["%zero"] = (stats["nbr.zero"] / len(df)) * 100

            # Basic statistics (excluding NaN)
            stats["mean"] = df[col].mean()
            stats["median"] = df[col].median()
            stats["std"] = df[col].std()
            stats["min"] = df[col].min()
            stats["max"] = df[col].max()

            # Count and percentage less than 0
            stats["nbr.less_than_0"] = (df[col] < 0).sum()
            stats["%less_than_0"] = (stats["nbr.less_than_0"] / len(df)) * 100
        else:
            # For non-numeric columns, fill numeric stats with None
            stats["nbr.zero"] = None
            stats["%zero"] = None
            stats["mean"] = None
            stats["median"] = None
            stats["std"] = None
            stats["min"] = None
            stats["max"] = None
            stats["nbr.less_than_0"] = None
            stats["%less_than_0"] = None

        # Unique count (works for all types)
        stats["unique"] = df[col].nunique()

        # Top value and its count
        if not df[col].isna().all():
            top_value = df[col].value_counts().index[0]
            top_count = df[col].value_counts().iloc[0]
            stats["top"] = top_value
            stats["nbr.top"] = top_count
            stats["%top"] = (top_count / len(df)) * 100
        else:
            stats["top"] = None
            stats["nbr.top"] = None
            stats["%top"] = None

        stats_list.append(stats)

    # Create DataFrame with desired column order
    result_df = pd.DataFrame(stats_list)
    column_order = [
        "variable",
        "nbr.null",
        "%null",
        "nbr.zero",
        "%zero",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "unique",
        "nbr.less_than_0",
        "%less_than_0",
        "top",
        "nbr.top",
        "%top",
    ]

    return result_df[column_order]

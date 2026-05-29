from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class CustomNAFiller(TransformerMixin, BaseEstimator):
    """
    Custom transformer to fill NA values with different strategies for different columns.

    Parameters
    ----------
    fill_zero_cols : list, optional (default=None)
        List of column names to fill NA with 0
    fill_9999_cols : list, optional (default=None)
        List of column names to fill NA with 9999
    fill_median_cols : list, optional (default=None)
        List of column names to fill NA with median
    fill_mode_cols : list, optional (default=None)
        List of column names to fill NA with mode (most frequent value)
    fill_dict : dict, optional (default=None)
        Dictionary mapping column names to specific fill values

    Examples
    --------
    >>> filler = CustomNAFiller(
    ...     fill_zero_cols=['col1', 'col2'],
    ...     fill_9999_cols=['col3'],
    ...     fill_median_cols=['col4', 'col5'],
    ...     fill_mode_cols=['col6', 'col7'],
    ...     fill_dict={'col8': -1, 'col9': 'missing'}
    ... )
    >>> X_filled = filler.fit_transform(X)
    """

    def __init__(
        self,
        fill_zero_cols=None,
        fill_9999_cols=None,
        fill_median_cols=None,
        fill_mode_cols=None,
        fill_1_cols=None,
        fill_dict=None,
    ):
        self.fill_zero_cols = [col.upper() for col in (fill_zero_cols or [])]
        self.fill_9999_cols = [col.upper() for col in (fill_9999_cols or [])]
        self.fill_median_cols = [col.upper()
                                 for col in (fill_median_cols or [])]
        self.fill_mode_cols = [col.upper() for col in (fill_mode_cols or [])]
        self.fill_1_cols = [col.upper() for col in (fill_1_cols or [])]
        self.fill_dict = {k.upper(): v for k, v in (fill_dict or {}).items()}
        self.median_values_ = {}
        self.mode_values_ = {}

    def fit(self, X, y=None):
        """
        Fit the transformer and calculate medians and modes for specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input data
        y : array-like, optional
            Target variable (ignored)

        Returns
        -------
        self
        """
        # Calculate and store median values for specified columns
        available_cols = set(X.columns)
        for col in self.fill_median_cols:
            if col in available_cols:
                self.median_values_[col] = X[col].median()

        # Calculate and store mode values for specified columns
        for col in self.fill_mode_cols:
            if col in available_cols:
                mode_result = X[col].mode()
                # Use the first mode if multiple modes exist
                self.mode_values_[col] = (
                    mode_result[0] if len(mode_result) > 0 else None
                )

        return self

    def transform(self, X):
        """
        Fill NA values according to the specified strategies.

        Parameters
        ----------
        X : pd.DataFrame
            Input data

        Returns
        -------
        X_filled : pd.DataFrame
            Data with NA values filled
        """
        # Create a copy to avoid modifying the original dataframe
        X_filled = X.copy()

        # Get actual columns present in the dataframe
        available_cols = set(X_filled.columns)

        # Fill with 0
        for col in self.fill_zero_cols:
            if col in available_cols:
                X_filled[col] = X_filled[col].fillna(0)

        for col in self.fill_1_cols:
            if col in available_cols:
                X_filled[col] = X_filled[col].fillna(1)

        # Fill with 9999
        for col in self.fill_9999_cols:
            if col in available_cols:
                X_filled[col] = X_filled[col].fillna(9999)

        # Fill with median
        for col in self.fill_median_cols:
            if col in available_cols and col in self.median_values_:
                X_filled[col] = X_filled[col].fillna(self.median_values_[col])

        # Fill with mode
        for col in self.fill_mode_cols:
            if col in available_cols and col in self.mode_values_:
                if self.mode_values_[col] is not None:
                    X_filled[col] = X_filled[col].fillna(
                        self.mode_values_[col])

        # Fill with specific values from dictionary
        for col, value in self.fill_dict.items():
            if col in available_cols:
                X_filled[col] = X_filled[col].fillna(value)

        return X_filled

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like, optional
            Input feature names

        Returns
        -------
        feature_names_out : ndarray
            Output feature names
        """
        if input_features is None:
            return np.array([])
        return np.asarray(input_features, dtype=object)

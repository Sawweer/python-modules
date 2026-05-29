from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import pandas as pd


class SelectByColumn(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        """
        Parameters
        ----------
        columns : list
            List of column names (for DataFrame) or indices (for numpy array)
        """
        self.columns = columns

    def fit(self, X, y=None):
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self._is_df = True
        else:
            X = check_array(X)
            self.feature_names_in_ = list(range(X.shape[1]))
            self._is_df = False

        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")

        if isinstance(X, pd.DataFrame):
            return X[self.columns]

        X = check_array(X)

        # assume columns are indices for numpy
        return X[:, self.columns]

    def get_feature_names_out(self):
        return self.columns

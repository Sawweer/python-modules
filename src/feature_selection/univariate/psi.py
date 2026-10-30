from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import pandas as pd
import numpy as np


def psi(expected, actual, bins=10, rare_threshold=0.01):
    """
    Calculate PSI between expected and actual distributions.

    Numerical:
        - Quantile bins derived from expected.
        - -1 and NaN are treated as missing.

    Categorical/string:
        - Each category is a bin.
        - Categories with expected frequency < rare_threshold
          are grouped into '__RARE__'.
        - Missing values are represented by ''.
    """
    expected = pd.Series(expected)
    actual = pd.Series(actual)

    is_categorical = (
        pd.api.types.is_string_dtype(expected.dtype)
        or pd.api.types.is_object_dtype(expected.dtype)
        or isinstance(expected.dtype, pd.CategoricalDtype)
    )

    epsilon = 1e-6

    if is_categorical:
        # Normalize missing values
        e = expected.fillna('').astype(str)
        a = actual.fillna('').astype(str)

        # Identify rare categories using expected distribution
        expected_ratio = e.value_counts(normalize=True)
        rare_categories = expected_ratio[
            expected_ratio < rare_threshold
        ].index

        # Merge rare categories into one group
        e = e.where(~e.isin(rare_categories), '__RARE__')
        a = a.where(~a.isin(rare_categories), '__RARE__')

        # Calculate proportions using the same category set
        categories = e.unique()
        categories = pd.Index(categories).union(pd.Index(a.unique()))

        p = e.value_counts(normalize=True).reindex(
            categories, fill_value=0
        ).to_numpy()

        q = a.value_counts(normalize=True).reindex(
            categories, fill_value=0
        ).to_numpy()

    else:
        # Convert to numeric
        expected = pd.to_numeric(expected, errors='coerce')
        actual = pd.to_numeric(actual, errors='coerce')

        e = expected[(expected != -1) & expected.notna()]

        if len(e) == 0:
            return 0.0

        edges = np.unique(
            np.quantile(e, np.linspace(0, 1, bins + 1))
        )

        if len(edges) < 3:
            return 0.0

        def dist(x, base):
            x = np.clip(
                x.dropna().to_numpy(),
                edges[0],
                edges[-1]
            )

            h = np.histogram(x, bins=edges)[0]

            missing_count = (
                (base == -1) | base.isna()
            ).sum()

            return np.append(h, missing_count) / len(base)

        p = dist(e, expected)
        q = dist(actual, actual)

    # Smooth zero proportions
    p = p + epsilon
    q = q + epsilon

    return float(np.sum((q - p) * np.log(q / p)))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import pandas as pd
import numpy as np


class SelectByPSI(TransformerMixin, BaseEstimator):
    def __init__(self, psi_threshold=0.25, base_indices=None,
                 bins=10, rare_threshold=0.01):
        self.psi_threshold = psi_threshold
        self.base_indices = base_indices
        self.bins = bins
        self.rare_threshold = rare_threshold

    def _to_dataframe(self, X):
        """Normalize input to DataFrame."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self._is_df = True
            return X.copy()

        X = check_array(
            X,
            dtype=None,
            ensure_all_finite="allow-nan"
        )
        self.feature_names_in_ = [
            f"x{i}" for i in range(X.shape[1])
        ]
        self._is_df = False

        return pd.DataFrame(
            X, columns=self.feature_names_in_
        )

    def _ensure_dataframe(self, X):
        """Convert transform input without changing feature names."""
        if isinstance(X, pd.DataFrame):
            return X

        X = check_array(
            X,
            dtype=None,
            ensure_all_finite="allow-nan"
        )

        if X.shape[1] != len(self.feature_names_in_):
            raise ValueError(
                "X has a different number of features than during fit."
            )

        return pd.DataFrame(
            X, columns=self.feature_names_in_
        )

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)

        self.feats = self.feature_names_in_

        n_rows = len(X_df)

        if self.base_indices is None:
            raise ValueError(
                "base_indices must be provided."
            )

        base_indices = np.asarray(self.base_indices)

        if base_indices.dtype == bool:
            if len(base_indices) != n_rows:
                raise ValueError(
                    "Boolean base_indices must match the number of rows."
                )
            base_mask = base_indices

        else:
            if not np.issubdtype(
                base_indices.dtype, np.integer
            ):
                raise ValueError(
                    "base_indices must contain integer row indices "
                    "or be a boolean mask."
                )

            if (
                np.any(base_indices < 0)
                or np.any(base_indices >= n_rows)
            ):
                raise ValueError(
                    "base_indices contains out-of-range row indices."
                )

            if len(np.unique(base_indices)) != len(base_indices):
                raise ValueError(
                    "base_indices must not contain duplicate indices."
                )

            base_mask = np.zeros(n_rows, dtype=bool)
            base_mask[base_indices] = True

        self.base_indices_ = np.flatnonzero(base_mask)
        self.test_indices_ = np.flatnonzero(~base_mask)

        if len(self.base_indices_) == 0:
            raise ValueError("The base dataset is empty.")

        if len(self.test_indices_) == 0:
            raise ValueError("The test dataset is empty.")

        base_df = X_df.iloc[self.base_indices_]
        test_df = X_df.iloc[self.test_indices_]

        # Calculate PSI for every feature.
        self.psi_scores_ = pd.Series(
            {
                feat: psi(
                    base_df[feat],
                    test_df[feat],
                    bins=self.bins,
                    rare_threshold=self.rare_threshold
                )
                for feat in self.feats
            },
            name="psi"
        )

        self.selected_features_ = self.psi_scores_[
            self.psi_scores_ < self.psi_threshold
        ].index.tolist()

        self.remove_reasons_ = {
            feat: (
                None
                if feat in self.selected_features_
                else f"PSI >= {self.psi_threshold}"
            )
            for feat in self.feats
        }

        self.selected_idx_ = [
            self.feature_names_in_.index(feat)
            for feat in self.selected_features_
        ]

        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")

        X_df = self._ensure_dataframe(X)

        if self._is_df:
            return X_df[self.selected_features_]

        return X_df[self.selected_features_].values

    def get_feature_names_in(self):
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)

    def get_remove_reasons(self):
        check_is_fitted(self, "remove_reasons_")
        return self.remove_reasons_

    def get_selection_summary(self):
        check_is_fitted(self, "psi_scores_")

        summary = self.psi_scores_.to_frame()
        summary["feature"] = summary.index
        summary["selection_status"] = summary.index.map(
            lambda x: (
                "Selected"
                if x in self.selected_features_
                else "Not Selected"
            )
        )
        summary["reason"] = summary.index.map(
            self.remove_reasons_
        )

        return (
            summary[
                [
                    "feature",
                    "psi",
                    "selection_status",
                    "reason"
                ]
            ]
            .sort_values(by="psi", ascending=True)
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    # Example data
    df = pd.DataFrame({
        "income": [100, 120, 110, 130, 115, 125, 105, 140],
        "age": [25, 30, 28, 35, 27, 32, 29, 36],
        "city": [
            "A", "B", "A", "C",
            "A", "B", "D", "C"
        ]
    })

    # First 4 rows = base dataset
    base_indices = [0, 1, 2, 3]

    selector = SelectByPSI(
        psi_threshold=0.1,
        base_indices=base_indices,
        bins=10,
        rare_threshold=0.01
    )

    selector.fit(df)

    # Inspect PSI scores and selection status
    print(selector.get_selection_summary())

    # Retrieve selected features
    print(selector.get_feature_names_out())

    # Transform the dataset
    df_selected = selector.transform(df)
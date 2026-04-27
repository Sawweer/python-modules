from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import pandas as pd
import numpy as np


class SelectByIdenticalRate(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def _to_dataframe(self, X):
        """Normalize input to DataFrame"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self._is_df = True
            return X.copy()
        else:
            X = check_array(X, dtype=None, ensure_all_finite="allow-nan")
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            self._is_df = False
            return pd.DataFrame(X, columns=self.feature_names_in_)

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)

        self.feats = self.feature_names_in_
        self.cat_feats = X_df.select_dtypes(
            include=["str"]).columns.tolist()
        self.num_feats = [f for f in self.feats if f not in self.cat_feats]

        # identical rate function
        def f_idt_rate(a):
            non_null = a.dropna()
            if len(non_null) == 0:
                return 1.0
            return non_null.value_counts().max() / len(non_null)

        self.identical_perc_ = X_df.apply(f_idt_rate)

        self.selected_features_ = self.identical_perc_[
            self.identical_perc_ < self.threshold
        ].index.tolist()

        self.remove_reasons_ = {
            feat: (
                None
                if feat in self.selected_features_
                else f"Identical rate >= {self.threshold}"
            )
            for feat in self.feats
        }

        # store indices for numpy transform
        self.selected_idx_ = [
            self.feature_names_in_.index(f) for f in self.selected_features_
        ]

        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")

        X_df = self._ensure_dataframe(X)

        if self._is_df:
            return X_df[self.selected_features_]

        return X_df[self.selected_features_].values

    def _ensure_dataframe(self, X):
        """Non-mutating version for transform"""
        if isinstance(X, pd.DataFrame):
            return X
        X = check_array(X, dtype=None, ensure_all_finite="allow-nan")
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def get_feature_names_in(self):
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self, "selected_features_")
        return self.selected_features_

    def get_remove_reasons(self):
        check_is_fitted(self, "remove_reasons_")
        return self.remove_reasons_

    def get_selection_summary(self):
        check_is_fitted(self, "identical_perc_")

        summary = self.identical_perc_.to_frame(name="identical_rate")
        summary["feature"] = summary.index
        summary["selection_status"] = summary.index.map(
            lambda x: "Selected" if x in self.selected_features_ else "Not Selected"
        )
        summary["reason"] = summary.index.map(self.remove_reasons_)

        return (
            summary[
                ["feature", "identical_rate", "selection_status", "reason"]
            ]
            .sort_values(by="identical_rate", ascending=True)
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    # Example usage
    data = {
        "A": [1, 1, 1, 1, 1],
        "B": [1, 2, 3, 4, 5],
        "C": ["x", "x", "x", "y", "y"],
        "D": [np.nan, np.nan, np.nan, np.nan, np.nan],
    }
    df = pd.DataFrame(data)

    selector = SelectByIdenticalRate(threshold=0.8)
    selector.fit(df)
    print(selector.get_selection_summary())

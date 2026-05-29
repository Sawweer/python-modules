from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import pandas as pd
import numpy as np


class SelectByMissingRate(TransformerMixin, BaseEstimator):
    def __init__(self, missing_rate_max=0.75):
        self.missing_rate_max = missing_rate_max

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

    def _ensure_dataframe(self, X):
        """Non-mutating version for transform"""
        if isinstance(X, pd.DataFrame):
            return X
        X = check_array(X, dtype=None, ensure_all_finite="allow-nan")
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)

        self.feats = self.feature_names_in_
        self.cat_feats = X_df.select_dtypes(
            include=["object"]).columns.tolist()
        self.num_feats = [f for f in self.feats if f not in self.cat_feats]

        # Missing rate
        self.missing_rate_ = X_df.isna().mean()

        self.selected_features_ = self.missing_rate_[
            self.missing_rate_ < self.missing_rate_max
        ].index.tolist()

        self.remove_reasons_ = {
            feat: (
                None
                if feat in self.selected_features_
                else f"Missing rate >= {self.missing_rate_max}"
            )
            for feat in self.feats
        }

        # For numpy transform
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

    def get_feature_names_in(self):
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self, "selected_features_")
        return self.selected_features_

    def get_remove_reasons(self):
        check_is_fitted(self, "remove_reasons_")
        return self.remove_reasons_

    def get_selection_summary(self):
        check_is_fitted(self, "missing_rate_")

        summary = self.missing_rate_.to_frame(name="missing_rate")
        summary["feature"] = summary.index
        summary["selection_status"] = summary.index.map(
            lambda x: "Selected" if x in self.selected_features_ else "Not Selected"
        )
        summary["reason"] = summary.index.map(self.remove_reasons_)

        return (
            summary[
                ["feature", "missing_rate", "selection_status", "reason"]
            ]
            .sort_values(by="missing_rate", ascending=True)
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    # Example usage
    data = {
        "A": [1, 2, np.nan, 4],
        "B": [np.nan, np.nan, np.nan, np.nan],
        "C": ["cat", "dog", "cat", "mouse"],
        "D": [1, 2, 3, 4]
    }
    df = pd.DataFrame(data)

    selector = SelectByMissingRate(missing_rate_max=0.5)
    selector.fit(df)
    print(selector.get_selection_summary())

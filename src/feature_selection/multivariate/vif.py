from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import warnings


class SelectByVIF(BaseEstimator, TransformerMixin):
    def __init__(self, vif_max=5.0, n_jobs=-1):
        self.vif_max = vif_max
        self.n_jobs = n_jobs

    def _to_dataframe(self, X):
        """Normalize input to DataFrame"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self._is_df = True
            return X.copy()
        else:
            X = check_array(X, dtype=float)
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            self._is_df = False
            return pd.DataFrame(X, columns=self.feature_names_in_)

    def _compute_vif(self, X):
        X_const = np.column_stack([np.ones(X.shape[0]), X])

        def _vif(i):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    val = variance_inflation_factor(X_const, i + 1)

                if np.isnan(val) or np.isinf(val):
                    return np.inf
                return val
            except Exception:
                return np.inf

        return np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(_vif)(i) for i in range(X.shape[1])
            )
        )

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        X_np = X_df.values

        features = self.feature_names_in_.copy()
        X_current = X_np.copy()

        self.removed_features_ = []
        self.removed_vif_ = {}

        # Iterative elimination
        while X_current.shape[1] > 0:
            vifs = self._compute_vif(X_current)
            max_vif = vifs.max()

            if max_vif < self.vif_max:
                break

            idx = vifs.argmax()
            removed = features[idx]

            self.removed_features_.append(removed)
            self.removed_vif_[removed] = max_vif

            features.pop(idx)
            X_current = np.delete(X_current, idx, axis=1)

        self.selected_features_ = features
        self.selected_idx_ = [
            self.feature_names_in_.index(f) for f in self.selected_features_
        ]

        # Final VIF values
        if self.selected_features_:
            final_vifs = self._compute_vif(X_np[:, self.selected_idx_])
            self.vif_ = dict(zip(self.selected_features_, final_vifs))
        else:
            self.vif_ = {}

        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")

        X_df = self._to_dataframe(X)

        if self._is_df:
            return X_df[self.selected_features_]

        # return numpy if input was numpy
        return X_df[self.selected_features_].values

    def get_feature_names_out(self):
        return self.selected_features_

    def get_vif(self):
        return self.vif_

    def get_selection_summary(self):
        check_is_fitted(self, ["vif_", "removed_vif_"])

        rows = []

        for f in self.feature_names_in_:
            if f in self.selected_features_:
                vif = self.vif_.get(f, np.inf)
                rows.append({
                    "feature": f,
                    "vif": vif,
                    "status": "Selected",
                    "reason": f"VIF < {self.vif_max}"
                })
            else:
                vif = self.removed_vif_.get(f, np.inf)
                rows.append({
                    "feature": f,
                    "vif": vif,
                    "status": "Not Selected",
                    "reason": f"VIF >= {self.vif_max}"
                })

        return pd.DataFrame(rows).sort_values(by="vif", ascending=True).reset_index(drop=True)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification

    X, _ = make_classification(
        n_samples=100, n_features=10, n_informative=5, random_state=42)
    selector = SelectByVIF(vif_max=5.0)
    selector.fit(X)
    print(selector.get_selection_summary())

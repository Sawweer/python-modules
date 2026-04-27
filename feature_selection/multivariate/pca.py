from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


class SelectByPCA(TransformerMixin, BaseEstimator):
    """
    Feature selector that uses PCA to select the most representative feature
    for each principal component — the feature with the highest absolute loading.

    Parameters
    ----------
    n_components : int, default=5
        Number of principal components to extract. One feature is selected per component.
    """

    def __init__(self, n_components=5):
        self.n_components = n_components

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

        pca = PCA(n_components=self.n_components)
        pca.fit(X_df.values)

        self.components_ = pca.components_
        self.explained_variance_ratio_ = pca.explained_variance_ratio_

        self.selected_features_ = []
        self.pc_loadings_ = {}
        assigned = set()

        for pc_idx, component in enumerate(self.components_):
            loadings = np.abs(component)
            ranked = np.argsort(loadings)[::-1]

            for feat_idx in ranked:
                feat = self.feature_names_in_[feat_idx]
                if feat not in assigned:
                    self.selected_features_.append(feat)
                    self.pc_loadings_[feat] = (pc_idx + 1, loadings[feat_idx])
                    assigned.add(feat)
                    break

        return self

    def transform(self, X):
        check_is_fitted(self, ["selected_features_"])
        X_df = self._to_dataframe(X)
        return X_df[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        return self.selected_features_

    def get_selection_summary(self):
        check_is_fitted(self, ["selected_features_",
                        "pc_loadings_", "components_"])

        selected_set = set(self.selected_features_)
        summary_list = []

        for feature in self.feature_names_in_:
            if feature in self.pc_loadings_:
                pc_idx, loading = self.pc_loadings_[feature]
            else:
                pc_idx, loading = np.nan, np.nan

            if feature in selected_set:
                status = "Selected"
                reason = f"Highest absolute loading on PC{pc_idx}"
            else:
                status = "Not Selected"
                reason = "Not the top loading feature on any PC"

            summary_list.append({
                "feature": feature,
                "selection_status": status,
                "pc": pc_idx,
                "pc_loading": loading,
                "reason": reason,
            })

        return (
            pd.DataFrame(summary_list)[
                ["feature", "selection_status", "pc", "pc_loading", "reason"]]
            .sort_values(["pc", "pc_loading"], ascending=[True, False])
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data
    feature_names = data.feature_names

    selector = SelectByPCA(n_components=3)
    selector.fit(X)
    selected_features = selector.get_feature_names_out()
    summary = selector.get_selection_summary()

    print("Selected features:", selected_features)
    print("\nSelection summary:\n", summary)

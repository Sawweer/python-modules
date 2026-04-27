from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


class SelectByPCA(TransformerMixin, BaseEstimator):
    """
    Feature selector that uses PCA to select the most representative feature
    for each principal component — the feature with the highest absolute loading.

    The number of components (and thus selected features) is determined either
    by a fixed count or by the minimum number needed to explain a target
    proportion of total variance.

    Parameters
    ----------
    n_components : int or None, default=None
        Fixed number of principal components to extract. Ignored if
        variance_threshold is set. If both are None, defaults to 5.

    variance_threshold : float or None, default=None
        Minimum cumulative explained variance ratio to reach (e.g. 0.95 for
        95%). When set, n_components is ignored and the number of components
        is chosen automatically.
    """

    def __init__(self, n_components=None, variance_threshold=None):
        self.n_components = n_components
        self.variance_threshold = variance_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_dataframe(self, X):
        """Normalize input to DataFrame."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self._is_df = True
            return X.copy()
        else:
            X = check_array(X, dtype=None, ensure_all_finite="allow-nan")
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            self._is_df = False
            return pd.DataFrame(X, columns=self.feature_names_in_)

    def _resolve_n_components(self, X_values):
        max_components = min(X_values.shape)

        if self.variance_threshold is not None:
            if not (0 < self.variance_threshold <= 1):
                raise ValueError(
                    "variance_threshold must be in the range (0, 1]. "
                    f"Got {self.variance_threshold!r}."
                )
            full_pca = PCA(n_components=max_components)
            full_pca.fit(X_values)
            cumulative = np.cumsum(full_pca.explained_variance_ratio_)
            n = int(np.argmax(cumulative >= self.variance_threshold) + 1)
            self.full_explained_variance_ratio_ = full_pca.explained_variance_ratio_
        else:
            n = self.n_components if self.n_components is not None else 5

            # ── SAFEGUARDS ─────────────────────────────────────────────────
            if self.n_components is not None and self.n_components < 1:
                raise ValueError(
                    f"n_components must be a positive integer. Got {self.n_components!r}."
                )
            if n > max_components:
                raise ValueError(
                    f"n_components={n} exceeds the maximum number of components "
                    f"PCA can extract for this data ({max_components} = "
                    f"min(n_samples={X_values.shape[0]}, "
                    f"n_features={X_values.shape[1]})). "
                    f"Reduce n_components to at most {max_components}."
                )
            # ──────────────────────────────────────────────────────────────

            self.full_explained_variance_ratio_ = None

        self.n_components_used_ = n
        return n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        n = self._resolve_n_components(X_df.values)

        pca = PCA(n_components=n)
        pca.fit(X_df.values)

        self.components_ = pca.components_
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.cumulative_variance_ratio_ = float(
            np.sum(pca.explained_variance_ratio_)
        )

        # Select the top-loading feature for each PC (no repeats)
        self.selected_features_ = []
        self.pc_loadings_ = {}
        assigned = set()

        for pc_idx, component in enumerate(self.components_):
            loadings = np.abs(component)
            for feat_idx in np.argsort(loadings)[::-1]:
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
            feat_idx = self.feature_names_in_.index(feature)

            if feature in selected_set:
                pc_idx, loading = self.pc_loadings_[feature]
                status = "Selected"
                reason = f"Highest absolute loading on PC{pc_idx}"
            else:
                # Find which PC this feature has its highest absolute loading on
                loadings_per_pc = np.abs(self.components_[:, feat_idx])
                best_pc = int(np.argmax(loadings_per_pc)) + 1
                loading = loadings_per_pc[best_pc - 1]
                pc_idx = best_pc
                status = "Not Selected"
                reason = f"Not the top loading feature on PC{pc_idx} (already assigned)"

            summary_list.append(
                {
                    "feature": feature,
                    "selection_status": status,
                    "pc": pc_idx,
                    "pc_loading": loading,
                    "reason": reason,
                }
            )

        return (
            pd.DataFrame(summary_list)[
                ["feature", "selection_status", "pc", "pc_loading", "reason"]
            ]
            .sort_values(["pc", "pc_loading"], ascending=[True, False])
            .reset_index(drop=True)
        )


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data
    feature_names = data.feature_names

    print("=== Fixed n_components=3 ===")
    selector = SelectByPCA(n_components=3)
    selector.fit(X)
    print("Components used    :", selector.n_components_used_)
    print("Cumulative variance:", f"{selector.cumulative_variance_ratio_:.2%}")
    print("Selected features  :", selector.get_feature_names_out())
    print(selector.get_selection_summary(), "\n")

    print("=== variance_threshold=0.8 ===")
    selector2 = SelectByPCA(variance_threshold=0.8)
    selector2.fit(X)
    print("Components used    :", selector2.n_components_used_)
    print("Cumulative variance:",
          f"{selector2.cumulative_variance_ratio_:.2%}")
    print("Selected features  :", selector2.get_feature_names_out())
    print(selector2.get_selection_summary())

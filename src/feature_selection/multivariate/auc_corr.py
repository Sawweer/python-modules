from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np
import pandas as pd


class SelectByCorrAUC(TransformerMixin, BaseEstimator):
    """
    Feature selector for binary classification that filters correlated features,
    keeping the best performer (by AUC) from each correlated group.

    For each group of mutually correlated features (above ``corr_threshold``),
    the feature with the highest ROC-AUC against the binary target is retained
    and all others are dropped.

    Parameters
    ----------
    corr_threshold : float, default=0.7
        Absolute correlation threshold for grouping features. Features with
        pairwise correlation >= this value are placed in the same group.
    corr_method : str, default='spearman'
        Correlation method passed to ``pd.DataFrame.corr``.
        One of ``'pearson'``, ``'spearman'``, ``'kendall'``.
    inverted : bool, default=True
        If True, negates raw feature values before computing AUC. Useful when
        a lower feature value indicates the positive class (e.g. a risk score
        where smaller means higher risk). Ignored when ``model`` is provided.
    model : sklearn estimator or None, default=None
        If provided, this model is fitted on each feature individually and its
        predicted probabilities are used to compute AUC. The estimator must
        implement ``predict_proba``. If None, raw feature values are used directly.
    cv : int, cross-validation generator, or None, default=5
        Cross-validation strategy used when ``model`` is provided.

        - ``None`` or ``0``: fit on the full training set (no CV).
        - ``int``: number of folds for ``StratifiedKFold``.
        - CV splitter: any scikit-learn cross-validator.

        Ignored when ``model`` is None.

    Attributes
    ----------
    feature_names_in_ : list of str
        Names of features seen during ``fit``.
    corr_groups_ : list of list of str
        Groups of correlated features identified during ``fit``.
    feature_auc_scores_ : dict of {str: float}
        ROC-AUC score for every input feature computed during ``fit``.
    selected_features_ : list of str
        Features retained after selection.
    fitted_models_ : dict of {str: estimator}
        Final models fitted on the full training set, keyed by feature name.
        Populated only when ``model`` is provided.
    """

    def __init__(
        self,
        corr_threshold=0.7,
        corr_method="spearman",
        inverted=True,
        model=None,
        cv=5,
    ):
        self.corr_threshold = corr_threshold
        self.corr_method = corr_method
        self.inverted = inverted
        self.model = model
        self.cv = cv

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------

    def _to_dataframe(self, X):
        """Normalize input to a DataFrame, recording feature names."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self._is_df = True
            return X.copy()
        else:
            X = check_array(X, dtype=None, ensure_all_finite="allow-nan")
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            self._is_df = False
            return pd.DataFrame(X, columns=self.feature_names_in_)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Binary target vector.

        Returns
        -------
        self : SelectByCorrAUC
        """
        X_df = self._to_dataframe(X)

        corr_matrix = X_df.corr(method=self.corr_method).abs()
        self.corr_groups_ = self._identify_corr_groups(corr_matrix)

        self.feature_auc_scores_ = {}
        self.fitted_models_ = {}

        for col in X_df.columns:
            try:
                if self.model is not None:
                    auc, fitted_model = self._compute_model_auc(X_df[[col]], y)
                    self.fitted_models_[col] = fitted_model
                else:
                    auc = self._compute_raw_auc(X_df[col], y)
                self.feature_auc_scores_[col] = auc
            except Exception:
                self.feature_auc_scores_[col] = 0.0

        self.selected_features_ = []
        for group in self.corr_groups_:
            best = max(group, key=lambda f: self.feature_auc_scores_[f])
            self.selected_features_.append(best)

        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_out : pd.DataFrame or np.ndarray
            Same type as the input passed to ``fit``.
        """
        check_is_fitted(self, ["selected_features_", "feature_auc_scores_"])

        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]

        indices = [self.feature_names_in_.index(
            f) for f in self.selected_features_]
        return X[:, indices]

    def get_feature_names_out(self, input_features=None):
        """Return the names of the selected features."""
        check_is_fitted(self, "selected_features_")
        return np.array(self.selected_features_)

    def get_selection_summary(self):
        """
        Return a summary of the selection process for every input feature.

        Returns
        -------
        summary : pd.DataFrame
            Columns: ``feature``, ``selection_status``, ``auc``, ``reason``.
        """
        check_is_fitted(self, ["selected_features_",
                        "feature_auc_scores_", "corr_groups_"])

        feature_to_group = {
            feat: group
            for group in self.corr_groups_
            for feat in group
        }
        selected_set = set(self.selected_features_)

        rows = []
        for feature in self.feature_names_in_:
            auc = self.feature_auc_scores_.get(feature, np.nan)
            group = feature_to_group.get(feature, [feature])
            group_str = f"Group: [{', '.join(group)}]"

            if feature in selected_set:
                status = "Selected"
                reason = f"Highest AUC within group. {group_str}"
            elif len(group) > 1:
                best = next(f for f in group if f in selected_set)
                status = "Not Selected"
                reason = f"Lower AUC than '{best}'. {group_str}"
            else:
                status = "Selected"
                reason = f"Only member of group. {group_str}"

            rows.append({"feature": feature, "auc": auc,
                        "selection_status": status, "reason": reason})

        return pd.DataFrame(rows, columns=["feature", "selection_status", "auc", "reason"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_raw_auc(self, feature_series, y):
        """Compute ROC-AUC from raw feature values."""
        values = -feature_series if self.inverted else feature_series
        return roc_auc_score(y, values)

    def _compute_model_auc(self, X_feature, y):
        """Fit ``self.model`` on a single feature and return its ROC-AUC and fitted model."""
        model = clone(self.model)

        if self.cv is not None and self.cv > 0:
            y_proba = cross_val_predict(
                model, X_feature, y, cv=self.cv, method="predict_proba"
            )
        else:
            model.fit(X_feature, y)
            y_proba = model.predict_proba(X_feature)

        auc = roc_auc_score(y, y_proba[:, 1])

        # Refit on full data so the stored model is trained on everything
        model.fit(X_feature, y)
        return auc, model

    def _identify_corr_groups(self, corr_matrix):
        """
        Greedily assign features to correlation groups.

        A feature is added to the first group whose representative feature
        exceeds ``corr_threshold``. Unassigned features start a new group.
        """
        features = corr_matrix.columns.tolist()
        groups = []
        assigned = set()

        for feat in features:
            if feat in assigned:
                continue
            correlated = corr_matrix.index[
                corr_matrix[feat] >= self.corr_threshold
            ].tolist()
            group = [f for f in correlated if f not in assigned]
            groups.append(group)
            assigned.update(group)

        return groups


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    data = load_breast_cancer()
    X, y = data.data, data.target

    selector = SelectByCorrAUC(
        corr_threshold=0.8,
        corr_method="spearman",
        inverted=True,
        model=LogisticRegression(solver="liblinear"),
        cv=5,
    )
    selector.fit(X, y)
    print(selector.get_selection_summary())

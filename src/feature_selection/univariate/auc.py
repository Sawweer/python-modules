from tokenize import group

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np
import pandas as pd


class SelectByAUC(TransformerMixin, BaseEstimator):
    """
    Feature selector for binary classification that retains features with ROC-AUC above a specified threshold.

    Parameters
    ----------
    auc_threshold : float, default=0.5
        ROC-AUC threshold for feature selection. Features with AUC >= this value are retained.
    inverted : bool or None, default=True
        If True, negates raw feature values before computing AUC. Useful when
        a lower feature value indicates the positive class (e.g. a risk score
        where smaller means higher risk). Ignored when ``model`` is provided. If None, inversion is inferred.
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
    feature_auc_scores_ : dict of {str: float}
        ROC-AUC score for every input feature computed during ``fit``.
    selected_features_ : list of str
        Features retained after selection.
    """

    def __init__(
        self,
        auc_threshold=0.5,
        model=None,
        inverted=None,
        cv=5,
    ):
        self.auc_threshold = auc_threshold
        self.model = model
        self.cv = cv
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
        self : SelectByAUC
        """
        X_df = self._to_dataframe(X)

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
        for col, auc in self.feature_auc_scores_.items():
            if auc >= self.auc_threshold:
                self.selected_features_.append(col)

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
                        "feature_auc_scores_"])


        rows = []
        for feature, auc in self.feature_auc_scores_.items():
            if feature in self.selected_features_:
                status = "Selected"
                reason = f"AUC >= {self.auc_threshold}"
            else:
                status = "Not Selected"
                reason = f"AUC < {self.auc_threshold}"
            rows.append([feature, status, auc, reason])

        return pd.DataFrame(rows, columns=["feature", "selection_status", "auc", "reason"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_raw_auc(self, feature_series, y):
        """Compute ROC-AUC from raw feature values."""
        if self.inverted is None:
            return max(roc_auc_score(y, feature_series), 1 - roc_auc_score(y, feature_series))
        elif self.inverted is False:
            return roc_auc_score(y, feature_series)
        else:
            return roc_auc_score(y, -feature_series)

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



if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    data = load_breast_cancer()
    X, y = data.data, data.target

    selector = SelectByAUC(
        auc_threshold=0.55,
        inverted=None,
        cv=5,
    )
    selector.fit(X, y)
    print(selector.get_selection_summary())

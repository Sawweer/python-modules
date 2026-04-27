"""Beam-search based multivariate feature selector.

Implements a constrained beam search that grows feature subsets and
evaluates each candidate using logistic regression (statsmodels).
Paths that violate a p-value or coefficient-sign constraint are
discarded. Final selected features, model and beam history are exposed
after :meth:`fit`.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


class ConstrainedBeamSearchSelector(TransformerMixin, BaseEstimator):
    """
    Beam search feature selector optimizing Gini score with p-value constraint.

    Parameters
    ----------
    k : int
        Number of features to select.
    beam_width : int, default=50
        Number of paths to retain per iteration.
    p_value_threshold : float, default=0.05
        Maximum p-value allowed for all features.
    top_n : int, default=10
        Number of top paths to retain in final beam summary.
    n_jobs : int, default=1
        Number of parallel jobs.
    verbose : bool, default=False
        Print progress per depth.
    """

    def __init__(
        self,
        k: int,
        beam_width: int = 50,
        p_value_threshold: float = 0.05,
        require_negative_coef: bool = True,
        top_n: int = 10,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Create the selector.

        Parameters
        ----------
        k : int
            Number of features to select (search depth).
        beam_width : int, default=50
            Number of candidate paths to keep at each depth.
        p_value_threshold : float, default=0.05
            Maximum allowed p-value for coefficients (features with higher
            p-values are rejected).
        require_negative_coef : bool, default=True
            If True require selected feature coefficients to be negative.
        top_n : int, default=10
            Number of top paths included in the final summary.
        n_jobs : int, default=1
            Number of parallel jobs for candidate evaluation (joblib).
        verbose : bool, default=False
            Print progress info when True.
        """
        self.k = k
        self.beam_width = beam_width
        self.p_value_threshold = p_value_threshold
        self.require_negative_coef = require_negative_coef
        self.top_n = top_n
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _to_dataframe(self, X):
        """Normalize input to a pandas DataFrame.

        This method accepts either a DataFrame or array-like `X`. It sets
        `self.feature_names_in_` and `_is_df` and returns a DataFrame copy.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            DataFrame view of `X` with appropriate column names.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self._is_df = True
            return X.copy()
        else:
            X = check_array(X, dtype=None, ensure_all_finite="allow-nan")
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            self._is_df = False
            return pd.DataFrame(X, columns=self.feature_names_in_)

    def _evaluate_path(self, features: tuple, X: pd.DataFrame, y: pd.Series):
        """Evaluate a candidate feature set.

        Fits a logistic regression using `statsmodels.Logit` on the
        requested `features` and returns a dict with `features`, `gini`
        and the fitted `model` when the candidate satisfies the
        configured constraints. Returns ``None`` if the candidate is
        rejected (e.g., due to p-values or coefficient signs) or if the
        model fails to fit.

        Parameters
        ----------
        features : tuple
            Tuple of feature names to evaluate.
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series or array-like
            Binary target vector.

        Returns
        -------
        dict or None
            Dictionary with keys `features`, `gini`, `model` when valid,
            otherwise ``None``.
        """

        X_curr = sm.add_constant(X[list(features)])
        try:
            model = sm.Logit(y, X_curr).fit(disp=0)
        except Exception:
            return None

        pvals = model.pvalues.drop("const", errors="ignore")
        if (pvals >= self.p_value_threshold).any():
            return None

        if self.require_negative_coef:
            coefs = model.params.drop("const", errors="ignore")
            if (coefs >= 0).any():
                return None

        gini = 2 * roc_auc_score(y, model.predict(X_curr)) - 1
        return {"features": list(features), "gini": gini, "model": model}

    def fit(self, X, y):
        from joblib import Parallel, delayed
        """Run beam search to select `k` features.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix.
        y : array-like or pd.Series
            Binary target vector.

        Returns
        -------
        self
        """

        X_df = self._to_dataframe(X)
        self.n_features_in_ = X_df.shape[1]

        current_beam = [{"features": tuple(), "gini": 0.0}]
        self.beam_history_ = []

        for depth in range(1, self.k + 1):
            candidate_paths = set()
            for path in current_beam:
                remaining = set(self.feature_names_in_) - set(path["features"])
                for feat in remaining:
                    candidate_paths.add(
                        tuple(sorted(set(path["features"]) | {feat})))

            if self.n_jobs == 1:
                results = [self._evaluate_path(p, X_df, y)
                           for p in candidate_paths]
            else:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._evaluate_path)(p, X_df, y) for p in candidate_paths
                )
            results = [r for r in results if r is not None]

            if not results:
                if self.verbose:
                    print(
                        f"Depth {depth}: no valid candidates, stopping early.")
                break

            results.sort(key=lambda x: x["gini"], reverse=True)
            current_beam = results[: self.beam_width]
            self.beam_history_.append(current_beam)

            if self.verbose:
                print(
                    f"Depth {depth}: {len(results)} valid paths, best Gini={current_beam[0]['gini']:.4f}")

        best = current_beam[0]
        self.selected_features_ = best["features"]
        self.model_ = best["model"]
        self.gini_score_ = best["gini"]

        self.final_beam_ = pd.DataFrame([
            {
                "rank": i + 1,
                "gini": p["gini"],
                "n_features": len(p["features"]),
                "features": ", ".join(p["features"]),
            }
            for i, p in enumerate(current_beam[: self.top_n])
        ])

        return self

    def transform(self, X):
        """Reduce `X` to the selected features after fitting.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix to transform.

        Returns
        -------
        pd.DataFrame
            Input `X` restricted to the selected features.
        """

        check_is_fitted(self, ["selected_features_"])
        return self._to_dataframe(X)[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        """Return the selected feature names as a numpy array.

        The `input_features` parameter is accepted for sklearn compatibility
        but ignored.
        """

        check_is_fitted(self, ["selected_features_"])
        return np.array(self.selected_features_)

    def get_beam_history(self) -> pd.DataFrame:
        """Return the full beam search history as a DataFrame.

        The returned DataFrame contains one row per candidate path and
        includes `depth`, `rank`, `gini`, `n_features` and `features`.
        """

        check_is_fitted(self, ["beam_history_"])
        rows = []
        for depth, beam in enumerate(self.beam_history_, start=1):
            for rank, path in enumerate(beam, start=1):
                rows.append({
                    "depth": depth,
                    "rank": rank,
                    "gini": path["gini"],
                    "n_features": len(path["features"]),
                    "features": ", ".join(path["features"]),
                })
        return pd.DataFrame(rows)[["depth", "rank", "gini", "n_features", "features"]]


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    selector = ConstrainedBeamSearchSelector(
        k=3, beam_width=10, p_value_threshold=0.05, require_negative_coef=True, top_n=5, n_jobs=-1, verbose=True
    )
    selector.fit(X, y)
    print("Selected features:", selector.get_feature_names_out())
    print("\nBeam History:")
    print(selector.get_beam_history())

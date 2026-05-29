from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from optbinning import BinningProcess
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

_VALID_METRICS = ("gini", "iv", "js")


class DynamicBinningProcess(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        metric="gini",
        metric_min=None,
        metric_max=None,
        n_jobs=-1,
        binning_process_params=None,
        verbose=True,
        monotonic_trends=None,
        user_splits=None,
    ):
        """
        Sklearn-compatible transformer that bins features via optbinning's
        BinningProcess and selects features based on a metric threshold.

        Fits a BinningProcess, computes Gini, IV and JS for every variable,
        and retains only those within [metric_min, metric_max] for the chosen
        metric. The transform step applies WoE encoding and returns only
        selected features.

        When metric=None, no metric computation or feature selection is
        performed — all features are binned and returned as WoE-encoded
        values. metric_min and metric_max are ignored in this mode.

        Args:
            metric (str or None): Metric used for feature selection. One of
                "gini", "iv", "js", or None. When None, all features are
                retained and metric thresholds are ignored. Default is "gini".
            metric_min (float, optional): Minimum threshold (inclusive).
                Features whose metric value is below this are excluded.
                Ignored when metric=None. Default is None.
            metric_max (float, optional): Maximum threshold (inclusive).
                Features whose metric value is above this are excluded.
                Ignored when metric=None. Default is None.
            n_jobs (int): Number of parallel jobs for BinningProcess.
                Default is -1.
            binning_process_params (dict, optional): Base parameters forwarded
                to BinningProcess for every variable. Per-variable overrides
                are applied from monotonic_trends and user_splits.
                Default is None.
            verbose (bool): Whether to print warnings during fit.
                Default is True.
            monotonic_trends (dict, optional): Mapping of feature -> monotonic
                trend string e.g. {"age": "ascending", "income": "descending"}.
                Valid values: "ascending", "descending", "auto",
                "auto_asc_desc", "auto_heuristic", "peak", "valley".
                Default is None.
            user_splits (dict, optional): Mapping of feature -> list of
                explicit split points passed to BinningProcess.
                Default is None.
        """
        if metric is not None and metric not in _VALID_METRICS:
            raise ValueError(
                f"metric must be one of {_VALID_METRICS} or None, got '{metric}'."
            )
        self.metric = metric
        self.metric_min = metric_min
        self.metric_max = metric_max
        self.n_jobs = n_jobs
        self.binning_process_params = binning_process_params or {}
        self.verbose = verbose
        self.monotonic_trends = monotonic_trends or {}
        self.user_splits = user_splits or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_dataframe(self, X):
        """Normalise input to a DataFrame, recording feature names."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self._is_df = True
            return X.copy()
        else:
            X = check_array(X, dtype=float)
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            self._is_df = False
            return pd.DataFrame(X, columns=self.feature_names_in_)

    def _extract_metrics(self, variable):
        """
        Extract Gini, IV and JS for a single variable from the fitted binner.

        Returns a dict with keys "gini", "iv", "js". Any metric that cannot
        be retrieved is stored as NaN.
        """
        result = {"gini": np.nan, "iv": np.nan, "js": np.nan}
        try:
            binning_obj = self.binner.get_binned_variable(variable)
            bt = binning_obj.binning_table

            result["gini"] = float(bt.gini)
            result["iv"] = float(bt.iv)
            result["js"] = float(bt.js)

        except Exception as e:
            if self.verbose:
                print(f"Could not extract metrics for '{variable}': {e}")

        return result

    def _passes_threshold(self, value):
        """Return True if value sits within [metric_min, metric_max]."""
        if np.isnan(value):
            return False
        if self.metric_min is not None and value < self.metric_min:
            return False
        if self.metric_max is not None and value > self.metric_max:
            return False
        return True

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        Fit the BinningProcess on X and y, then compute Gini, IV and JS for
        every feature and determine which pass the metric threshold.

        When metric=None, metric computation is skipped entirely and all
        features are marked as selected.

        Populates:
            feature_names_in_ : all input feature names.
            metric_values_    : dict mapping feature -> {"gini", "iv", "js"}.
                                Empty when metric=None.
            selected_features_: features passing the threshold (all features
                                when metric=None).

        Args:
            X (DataFrame or array-like): Input features.
            y (array-like): Binary target variable.

        Returns:
            self
        """
        X = self._to_dataframe(X)

        self.categorical_features_ = (
            X.select_dtypes(include="object").columns.tolist()
        )

        # Build per-variable binning params
        self.binning_fit_params_full_ = {}
        for col in self.feature_names_in_:
            params = self.binning_process_params.copy()
            params["monotonic_trend"] = self.monotonic_trends.get(
                col, params.get("monotonic_trend")
            )
            if col in self.user_splits:
                params["user_splits"] = self.user_splits[col]
            self.binning_fit_params_full_[col] = params

        self.binner = BinningProcess(
            n_jobs=self.n_jobs,
            variable_names=self.feature_names_in_,
            binning_fit_params=self.binning_fit_params_full_,
            categorical_variables=self.categorical_features_,
        )
        self.binner.fit(X, y)

        # Always compute metrics for every feature
        self.metric_values_ = {}
        self.selected_features_ = []

        for variable in self.feature_names_in_:
            m = self._extract_metrics(variable)
            self.metric_values_[variable] = m
            # When metric=None retain all features; otherwise apply threshold
            if self.metric is None or self._passes_threshold(m[self.metric]):
                self.selected_features_.append(variable)

        return self

    def transform(self, X):
        """
        Apply WoE encoding via the fitted BinningProcess and return only
        the selected features. When metric=None all features are returned.

        Args:
            X (DataFrame or array-like): Input features.

        Returns:
            DataFrame: WoE-encoded DataFrame.
        """
        check_is_fitted(self, "selected_features_")

        if not self.selected_features_:
            if self.verbose:
                print(
                    f"No features meet the {self.metric} thresholds "
                    f"(min={self.metric_min}, max={self.metric_max}). "
                    "Returning empty DataFrame."
                )
            return pd.DataFrame(index=pd.DataFrame(X).index)

        transformed = self.binner.transform(
            X, metric_missing="empirical", metric_special="empirical"
        )
        cols = [c for c in self.selected_features_ if c in transformed.columns]
        return transformed[cols]

    def fit_transform(self, X, y=None):
        """Fit to X, y then transform X."""
        self.fit(X, y)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Summary / selection helpers
    # ------------------------------------------------------------------

    def get_feature_names_in(self):
        """Return all input feature names."""
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_

    def get_feature_names_out(self, input_features=None):
        """Return selected feature names (post-threshold)."""
        check_is_fitted(self, "selected_features_")
        return self.selected_features_

    def get_binning_process(self):
        """Return the fitted BinningProcess object."""
        check_is_fitted(self, "binner")
        return self.binner

    def get_metric_values(self):
        """
        Return raw metric dicts for all features.

        Returns:
            dict: {feature: {"gini": float, "iv": float, "js": float}}
                  Empty dict when metric=None.
        """
        check_is_fitted(self, "metric_values_")
        return self.metric_values_

    def get_selection_summary(self):
        """
        Return a tidy DataFrame showing each feature's Gini, IV, JS,
        selection status and the reason for inclusion or exclusion.

        When metric=None, metrics are not computed; all features are shown
        as Selected with reason "No metric filtering (metric=None)".

        Returns:
            DataFrame: Sorted descending by the active metric (or by feature
                name when metric=None), with columns:
                feature, gini, iv, js, selection_status, reason.
        """
        check_is_fitted(self, "metric_values_")

        rows = []
        for feature in self.feature_names_in_:
            m = self.metric_values_.get(
                feature, {"gini": np.nan, "iv": np.nan, "js": np.nan}
            )

            if self.metric is None:
                selected = True
                reason = "No metric filtering (metric=None)"
            else:
                value = m[self.metric]
                in_min = self.metric_min is None or (
                    not np.isnan(value) and value >= self.metric_min
                )
                in_max = self.metric_max is None or (
                    not np.isnan(value) and value <= self.metric_max
                )
                selected = in_min and in_max and not np.isnan(value)

                if selected:
                    parts = []
                    if self.metric_min is not None:
                        parts.append(f"{self.metric} >= {self.metric_min}")
                    if self.metric_max is not None:
                        parts.append(f"{self.metric} <= {self.metric_max}")
                    reason = " and ".join(
                        parts) if parts else "No threshold set"
                elif np.isnan(value):
                    reason = "Metric extraction failed"
                elif not in_min:
                    reason = f"{self.metric} < {self.metric_min}"
                else:
                    reason = f"{self.metric} > {self.metric_max}"

            rows.append({
                "feature": feature,
                "gini": m["gini"],
                "iv": m["iv"],
                "js": m["js"],
                "selection_status": "Selected" if selected else "Not Selected",
                "reason": reason,
            })

        df = pd.DataFrame(rows)
        sort_col = self.metric if self.metric is not None else "iv"
        return df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Binning table & plot helpers
    # ------------------------------------------------------------------

    def generate_binning_tables(self, features=None):
        """
        Generate a concatenated binning table for the requested features.

        Args:
            features (list, optional): Feature names to include. Defaults to
                selected_features_.

        Returns:
            DataFrame: Concatenated binning tables with zero-count rows and
                the Totals row dropped.
        """
        check_is_fitted(self, "binner")
        if features is None:
            features = self.selected_features_

        binning_tables = []
        for feature in features:
            bt = self.binner.get_binned_variable(feature).binning_table.build()
            bt["Feature"] = feature
            bt = bt.drop(index=bt.index[-1])  # drop Totals row
            binning_tables.append(bt)

        table = pd.concat(binning_tables, ignore_index=True)
        cols = ["Feature"] + [c for c in table.columns if c != "Feature"]
        table = table[cols]
        table = table.query("Count > 0").reset_index(drop=True)
        return table

    def generate_plot(self, features=None, metric="WoE", save_dir=None):
        """
        Plot stacked event/non-event counts with a secondary WoE or Event rate
        line for each feature, followed by a printed binning table.

        Args:
            features (list, optional): Features to plot. Defaults to
                selected_features_.
            metric (str): Secondary axis metric — "WoE" or "Event rate".
                Default is "WoE".
            save_dir (str, optional): Directory to save figures as PNGs.
                If None, figures are shown interactively.
        """
        check_is_fitted(self, "binner")
        if features is None:
            features = self.selected_features_

        table = self.generate_binning_tables(features)

        for feature in features:
            ft = table.query(f"Feature == '{feature}'").copy()
            plot_ft = ft[
                ~ft["Bin"].astype(str).str.lower().str.startswith("total")
            ]

            bins = plot_ft["Bin"].astype(str).tolist()
            non_events = plot_ft["Non-event"].tolist()
            events = plot_ft["Event"].tolist()
            line_vals = (
                plot_ft["WoE"].tolist()
                if metric == "WoE"
                else plot_ft["Event rate"].tolist()
            )
            line_label = "WoE" if metric == "WoE" else "Event rate"

            fig, ax1 = plt.subplots(figsize=(max(10, len(bins) * 1.4), 6))
            x = np.arange(len(bins))
            width = 0.6

            ax1.bar(x, non_events, width, label="Non-event",
                    color="#378ADD", alpha=0.85)
            ax1.bar(x, events, width, bottom=non_events,
                    label="Event", color="#FF8103", alpha=0.85)
            ax1.set_xlabel("Bin", fontsize=11)
            ax1.set_ylabel("Count", fontsize=11)
            ax1.set_xticks(x)
            ax1.set_xticklabels(bins, rotation=45, ha="right", fontsize=9)
            ax1.legend(loc="upper left", fontsize=9)
            ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

            ax2 = ax1.twinx()
            ax2.plot(
                x, line_vals, color="#FD0000", marker="o",
                linewidth=2, markersize=5, label=line_label, zorder=5,
            )
            ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax2.set_ylabel(line_label, fontsize=11, color="#FF0000")
            ax2.tick_params(axis="y", labelcolor="#FF0000")
            ax2.legend(loc="upper right", fontsize=9)

            plt.title(
                f"{feature}  —  Count distribution & {line_label}",
                fontsize=12, pad=12,
            )
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/{feature}.png", bbox_inches="tight")
                plt.close()
            else:
                plt.show()

            display_cols = [
                "Feature", "Bin", "Count", "Count (%)", "Non-event",
                "Event", "Event rate", "WoE", "IV", "JS",
            ]
            print_cols = [c for c in display_cols if c in ft.columns]
            fmt = {
                "Count (%)":  "{:.2%}".format,
                "Event rate": "{:.2%}".format,
                "WoE":        "{:.4f}".format,
                "IV":         "{:.4f}".format,
                "JS":         "{:.4f}".format,
            }
            styled = ft[print_cols].style.format(
                {k: v for k, v in fmt.items() if k in print_cols}
            ).set_caption(f"Binning table — {feature}")

            try:
                from IPython.display import display
                display(styled)
            except ImportError:
                print(ft[print_cols].to_string(index=False))
            print()


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1000, n_features=3, n_informative=3,
        n_redundant=0, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, 4)])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42
    )

    # Example 1: metric=None — bin everything, no filtering
    binner_no_metric = DynamicBinningProcess(
        metric=None,
        binning_process_params={"max_n_bins": 5, "monotonic_trend": "auto"},
        n_jobs=-1,
    )
    X_train_woe = binner_no_metric.fit_transform(X_train, y_train)
    print("All features (metric=None):",
          binner_no_metric.get_feature_names_out())
    print(binner_no_metric.get_selection_summary())
    print()

    # Example 2: metric="iv" with thresholds
    binner = DynamicBinningProcess(
        metric="iv",
        metric_min=0.1,
        metric_max=None,
        binning_process_params={"max_n_bins": 5, "monotonic_trend": "auto"},
        monotonic_trends={"Feature_1": "ascending", "Feature_2": "descending"},
        user_splits={"Feature_3": [0.5]},
        n_jobs=-1,
    )

    X_train_woe = binner.fit_transform(X_train, y_train)
    X_test_woe = binner.transform(X_test)

    print("Selected features:", binner.get_feature_names_out())
    print()
    print(binner.get_selection_summary())
    print()
    print(binner.generate_binning_tables().head())
    binner.generate_plot(metric="Event rate")

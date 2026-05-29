"""
optbinning_expansion
--------------------

Convenience wrapper around `optbinning.BinningProcess` providing a
scikit-learn compatible transformer, table generation and plotting helpers.

Configuration:
    - `binning_process_params` (dict): base parameters forwarded to
        `BinningProcess` for numeric variables. Per-variable overrides are applied
        from `monotonic_trends` and `user_splits`.
    - `monotonic_trends` (dict): mapping feature -> monotonic trend string
        (e.g. "ascending", "descending", "auto").
    - `user_splits` (dict): mapping feature -> list of split points.

Usage:
    Build and fit the transformer, then call `generate_binning_tables()` and
    `generate_plot()` to inspect results. See the example in the module
    `if __name__ == '__main__'` block.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from optbinning import BinningProcess
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class DynamicBinningProcess(TransformerMixin, BaseEstimator):
    """
    Sklearn compatible optbinning

    Parameters:
    -----------------
    binning_process_params: dict
        dictionary of binning params that will be applied across all columns, except for object ones
    monotonic_trends: dict, optional
        dictionary mapping variable names to monotonic trend constraints
        e.g., {"OS": "ascending", "income": "descending"}
        Valid values: "ascending", "descending", "auto", "auto_asc_desc", "auto_heuristic", "peak", "valley"
    """
    """
    Sklearn-compatible wrapper for `optbinning.BinningProcess`.

    This class builds per-variable `binning_fit_params` from a base
    `binning_process_params` dict and optional per-variable overrides.

    Parameters
    ----------
    binning_process_params : dict or None
        Base parameters passed to `BinningProcess` for each variable.
    monotonic_trends : dict, optional
        Per-variable monotonic trend constraints (e.g. {"age": "ascending"}).
    user_splits : dict, optional
        Per-variable explicit split points (lists) to pass as
        `user_splits` to `BinningProcess`.
    n_jobs : int, default=1
        Number of parallel jobs for `BinningProcess`.
    """

    def __init__(
        self,
        binning_process_params=None,
        monotonic_trends=None,
        user_splits=None,
        n_jobs=1,
    ):
        self.binner = None
        self.n_jobs = n_jobs
        self.binning_process_params = binning_process_params
        self.monotonic_trends = monotonic_trends or {}
        self.user_splits = user_splits or {}

    def fit(self, X, y):
        # Update binning parameters based on selected columns
        """
        Fit the internal `BinningProcess` on features `X` and target `y`.

        Builds per-variable `binning_fit_params` by copying
        `binning_process_params` and applying any per-variable
        `monotonic_trends` and `user_splits` overrides before fitting.

        Returns
        -------
        self
        """
        self.feature_names_in_ = X.columns.to_list()

        self.categorical_features_ = X.select_dtypes(
            include="object").columns.to_list()

        # Build binning params for each variable
        self.binning_fit_params_full_ = {}
        for col in self.feature_names_in_:
            params = self.binning_process_params.copy()

            if 'monotonic_trend' not in params or params["monotonic_trend"] != self.monotonic_trends.get(col):
                params["monotonic_trend"] = self.monotonic_trends.get(col)

            if col in self.user_splits:
                params["user_splits"] = self.user_splits.get(col, None)

            self.binning_fit_params_full_[col] = params

        self.binner = BinningProcess(
            n_jobs=self.n_jobs,
            variable_names=self.feature_names_in_,
            binning_fit_params=self.binning_fit_params_full_,
            categorical_variables=self.categorical_features_,
        )
        self.binner.fit(X, y)
        return self

    def transform(self, X):
        """
        Transform `X` using the fitted `BinningProcess`.

        Returns binned DataFrame-like object produced by `BinningProcess.transform`.
        """
        return self.binner.transform(
            X, metric_missing="empirical", metric_special="empirical"
        )

    def fit_transform(self, X, y):
        """
        Fit to `X, y` then transform `X`.

        Convenience wrapper around `fit` + `transform`.
        """
        # Combine fit and transform for convenience
        self.fit(X, y)
        return self.transform(X)

    def get_binning_summary(self):
        return self.binner.summary()

    def get_feature_names_in(self):
        if not hasattr(self, "feature_names_in_"):
            raise AttributeError(
                "This DynamicBinningProcess instance is not fitted yet."
            )
        return self.feature_names_in_

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.get_feature_names_in()
        return input_features

    def generate_binning_tables(self, features=None):
        """
        Generate a concatenated binning table for `features`.

        Parameters
        ----------
        features : iterable or None
            List of feature names to include. If None, uses features from
            `get_feature_names_in()`.

        Returns
        -------
        pandas.DataFrame
            Concatenated binning tables for requested features (drops zero
            count rows and the Totals row for each variable).
        """
        if features is None:
            features = self.get_feature_names_in()
        binning_tables = []
        for feature in features:
            binning_table = self.binner.get_binned_variable(
                feature).binning_table.build()
            binning_table["Feature"] = feature
            binning_table = binning_table.drop(
                index=binning_table.index[-1])  # drop Totals row
            binning_tables.append(binning_table)
        table = pd.concat(binning_tables, ignore_index=True)
        cols = list(table.columns)
        cols = [cols[-1]] + cols[:-1]
        table = table[cols]
        table = table.query("Count > 0").reset_index(drop=True)
        return table

    def generate_plot(self, features=None, metric="WoE", save_dir=None):
        """
        Generate plots for the requested `features` showing stacked counts and
        a secondary line for `metric` (either "WoE" or "Event rate").

        Parameters
        ----------
        features : iterable or None
            Features to plot. If None, plots all fitted features.
        metric : {"WoE", "Event rate"}, default "WoE"
            Metric to plot on the secondary y-axis.
        save_dir : str or None
            Directory to save figures; if None figures are shown interactively.
        """
        if features is None:
            features = self.get_feature_names_in()

        table = self.generate_binning_tables(features)

        for feature in features:
            ft = table.query(f"Feature == '{feature}'").copy()

            # ── drop the Totals row for plotting but keep it for the table ──
            plot_ft = ft[~ft["Bin"].astype(
                str).str.lower().str.startswith("total")]

            bins = plot_ft["Bin"].astype(str).tolist()
            non_events = plot_ft["Non-event"].tolist()
            events = plot_ft["Event"].tolist()
            line_vals = plot_ft["WoE"].tolist(
            ) if metric == "WoE" else plot_ft["Event rate"].tolist()
            line_label = "WoE" if metric == "WoE" else "Event rate"

            fig, ax1 = plt.subplots(figsize=(max(10, len(bins) * 1.4), 6))

            x = np.arange(len(bins))
            width = 0.6

            # Stacked bars
            bars_ne = ax1.bar(x, non_events, width,
                              label="Non-event", color="#378ADD", alpha=0.85)
            bars_ev = ax1.bar(x, events, width, bottom=non_events,
                              label="Event", color="#FF8103", alpha=0.85)

            ax1.set_xlabel("Bin", fontsize=11)
            ax1.set_ylabel("Count", fontsize=11)
            ax1.set_xticks(x)
            ax1.set_xticklabels(bins, rotation=45, ha="right", fontsize=9)
            ax1.legend(loc="upper left", fontsize=9)
            ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

            # Line on secondary axis
            ax2 = ax1.twinx()
            ax2.plot(x, line_vals, color="#FD0000", marker="o", linewidth=2,
                     markersize=5, label=line_label, zorder=5)
            ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax2.set_ylabel(line_label, fontsize=11, color="#FF0000")
            ax2.tick_params(axis="y", labelcolor="#FF0000")
            ax2.legend(loc="upper right", fontsize=9)

            plt.title(
                f"{feature}  —  Count distribution & {line_label}", fontsize=12, pad=12)
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/{feature}.png", bbox_inches='tight')
                plt.close()
            else:
                plt.show()

            # ── Binning table ──────────────────────────────────────────────
            display_cols = ["Feature", "Bin", "Count", "Count (%)", "Non-event", "Event",
                            "Event rate", "WoE", "IV", "JS"]
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


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=3,
                               n_informative=3, n_redundant=0,
                               random_state=42)
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, 4)])
    df["Target"] = y

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns="Target"), df["Target"], test_size=0.2, random_state=42
    )

    # Initialize and fit DynamicBinningProcess
    binning_params = {"max_n_bins": 5, "monotonic_trend": 'auto'}
    monotonic_trends = {"Feature_1": "ascending", "Feature_2": "descending"}
    user_splits = {"Feature_3": [0.5]}  # Example of user-defined split

    binning_process = DynamicBinningProcess(
        binning_process_params=binning_params,
        monotonic_trends=monotonic_trends,
        user_splits=user_splits,
        n_jobs=-1,
    )
    binning_process.fit(X_train, y_train)

    # Generate binning tables and plots for all features
    table = binning_process.generate_binning_tables()
    print(table.head())

    # plot
    binning_process.generate_plot(metric="Event rate")

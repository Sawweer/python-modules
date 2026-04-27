import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


class UnivariateGiniByDate:
    def __init__(self, df, date_column, target_column, predictors):
        self.df = df
        self.date_column = date_column
        self.target_column = target_column
        self.predictors = predictors
        self._dates = df[date_column].sort_values().unique()

    @staticmethod
    def _gini(y_true, y_score):
        return 2 * roc_auc_score(y_true, y_score) - 1

    def compute(self):
        results = []

        for date in self._dates:
            subset = self.df[self.df[self.date_column] == date]
            y_true = subset[self.target_column]

            for predictor in self.predictors:
                # negative sign keeps your original convention
                y_pred = -subset[predictor]

                try:
                    gini = self._gini(y_true, y_pred)
                except ValueError:
                    # handle edge case: only one class in this date
                    gini = None

                results.append({
                    "PROCESS_DATE": date,
                    "VARIABLE": predictor,
                    "GINI": gini
                })

        return pd.DataFrame(results)


class MetricsByDate:
    def __init__(self, df, date_column, target_column, score_column=None, pred_column=None, save_dir=None):
        self.df = df
        self.date_column = date_column
        self.target_column = target_column
        self.score_column = score_column
        self.pred_column = pred_column
        self._sorted_dates = df[date_column].sort_values().unique()
        self.save_dir = Path(save_dir) if save_dir else None

    def _iter_subsets(self):
        for date in self._sorted_dates:
            yield date, self.df[self.df[self.date_column] == date]

    def _compute_by_date(self, metric_fn, result_key, column):
        return pd.DataFrame([
            {"PROCESS_DATE": date, result_key: metric_fn(
                subset[self.target_column], subset[column])}
            for date, subset in self._iter_subsets()
        ])

    def _save(self, fig, filename):
        """Save figure to save_dir if set."""
        if self.save_dir:
            path = self.save_dir / filename
            fig.savefig(path, bbox_inches="tight", dpi=150)

    def _plot(self, df, metric_col, title=None, ax=None, figsize=(12, 4), color="steelblue"):
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.plot(df["PROCESS_DATE"], df[metric_col], marker="o",
                color=color, linewidth=2, markersize=4)
        ax.set_title(title or metric_col, fontsize=13, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel(metric_col)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        if standalone:
            fig.tight_layout()
            self._save(fig, f"{metric_col.lower()}_by_date.png")
            plt.show()

        return fig, ax

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def gini(self):
        def gini_coefficient(y_true, y_score):
            return 2 * roc_auc_score(y_true, - y_score) - 1
        return self._compute_by_date(gini_coefficient, "GINI", self.score_column)

    def f1(self):
        return self._compute_by_date(f1_score, "F1", self.pred_column)

    def precision(self):
        return self._compute_by_date(precision_score, "PRECISION", self.pred_column)

    def recall(self):
        return self._compute_by_date(recall_score, "RECALL", self.pred_column)

    def all_metrics(self):
        results = {"PROCESS_DATE": list(self._sorted_dates)}
        if self.score_column:
            results["GINI"] = self.gini()["GINI"].values
        if self.pred_column:
            results["F1"] = self.f1()["F1"].values
            results["PRECISION"] = self.precision()["PRECISION"].values
            results["RECALL"] = self.recall()["RECALL"].values
        return pd.DataFrame(results)

    # ------------------------------------------------------------------ #
    # Individual plots                                                     #
    # ------------------------------------------------------------------ #

    def plot_gini(self, ax=None, figsize=(12, 4), color="steelblue"):
        return self._plot(self.gini(), "GINI", title="Gini by Date", ax=ax, figsize=figsize, color=color)

    def plot_f1(self, ax=None, figsize=(12, 4), color="seagreen"):
        return self._plot(self.f1(), "F1", title="F1 Score by Date", ax=ax, figsize=figsize, color=color)

    def plot_precision(self, ax=None, figsize=(12, 4), color="darkorange"):
        return self._plot(self.precision(), "PRECISION", title="Precision by Date", ax=ax, figsize=figsize, color=color)

    def plot_recall(self, ax=None, figsize=(12, 4), color="mediumpurple"):
        return self._plot(self.recall(), "RECALL", title="Recall by Date", ax=ax, figsize=figsize, color=color)

    # ------------------------------------------------------------------ #
    # Dashboard                                                            #
    # ------------------------------------------------------------------ #

    def plot_all(self, figsize=(16, 10)):
        metrics = []
        if self.score_column:
            metrics.append((self.gini(),      "GINI",
                           "Gini",      "steelblue"))
        if self.pred_column:
            metrics.append((self.f1(),        "F1",
                           "F1",        "seagreen"))
            metrics.append((self.precision(), "PRECISION",
                           "Precision", "darkorange"))
            metrics.append((self.recall(),    "RECALL",
                           "Recall",    "mediumpurple"))

        if not metrics:
            raise ValueError(
                "No metrics to plot — provide score_column and/or pred_column.")

        n = len(metrics)
        ncols = 2
        nrows = (n + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.array(axes).flatten()

        for ax, (df, col, title, color) in zip(axes, metrics):
            self._plot(df, col, title=f"{title} by Date", ax=ax, color=color)

        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle("Model Metrics Over Time",
                     fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        self._save(fig, "all_metrics_by_date.png")
        plt.show()
        return fig


if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        "PROCESS_DATE": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "TARGET": [0, 1, 0, 1],
        "SCORE": [0.2, 0.8, 0.3, 0.7],
        "PRED": [0, 1, 0, 1]
    })

    metrics = MetricsByDate(df, "PROCESS_DATE", "TARGET",
                            score_column="SCORE", pred_column="PRED")
    gini_results = metrics.all_metrics()
    print(gini_results)
    metrics.plot_all()

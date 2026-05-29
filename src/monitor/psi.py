import numpy as np
import pandas as pd
import warnings


class FeaturePSIByDate:
    def __init__(
        self,
        df_ref,
        df_cur,
        features,
        date_column="PROCESS_DATE",
        portfolio_column=None,
        categorical_threshold=20,
        bins=10,
    ):
        self.df_ref = df_ref
        self.df_cur = df_cur
        self.features = features
        self.date_column = date_column
        self.portfolio_column = portfolio_column
        self.categorical_threshold = categorical_threshold
        self.bins = bins

    # ------------------------------------------------------------------ #
    # Core PSI logic                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _psi(expected_probs, actual_probs):
        epsilon = 1e-10
        expected_probs = np.maximum(expected_probs, epsilon)
        actual_probs = np.maximum(actual_probs, epsilon)
        return np.sum((expected_probs - actual_probs) * np.log(expected_probs / actual_probs))

    def _categorical_psi(self, expected, actual):
        categories = np.union1d(expected, actual)

        if len(categories) > 20:
            warnings.warn("High cardinality categorical feature.")

        expected_probs = np.array([(expected == c).mean() for c in categories])
        actual_probs = np.array([(actual == c).mean() for c in categories])

        return self._psi(expected_probs, actual_probs)

    def _numerical_psi(self, expected, actual):
        bins = min(self.bins, len(expected))

        if bins < self.bins:
            warnings.warn("Bins reduced due to small sample size.")

        edges = np.percentile(expected, np.linspace(0, 100, bins + 1))

        expected_probs, _ = np.histogram(expected, bins=edges)
        actual_probs, _ = np.histogram(actual, bins=edges)

        expected_probs = expected_probs / len(expected)
        actual_probs = actual_probs / len(actual)

        return self._psi(expected_probs, actual_probs)

    def _feature_psi(self, s_ref, s_cur):
        s_ref = s_ref.dropna()
        s_cur = s_cur.dropna()

        if len(s_ref) == 0 or len(s_cur) == 0:
            return np.nan

        is_categorical = s_ref.nunique() <= self.categorical_threshold

        if is_categorical:
            return self._categorical_psi(s_ref.values, s_cur.values)
        else:
            return self._numerical_psi(s_ref.values, s_cur.values)

    # ------------------------------------------------------------------ #
    # Main API                                                            #
    # ------------------------------------------------------------------ #

    def compute(self):
        results = []
        dates = self.df_cur[self.date_column].sort_values().unique()

        for date in dates:
            df_cur_d = self.df_cur[self.df_cur[self.date_column] == date]

            if self.portfolio_column:
                portfolios = df_cur_d[self.portfolio_column].unique()

                for p in portfolios:
                    df_cur_p = df_cur_d[df_cur_d[self.portfolio_column] == p]
                    df_ref_p = self.df_ref[self.df_ref[self.portfolio_column] == p]

                    for f in self.features:
                        psi = self._feature_psi(df_ref_p[f], df_cur_p[f])
                        results.append({
                            "PROCESS_DATE": date,
                            "PORTFOLIO": p,
                            "FEATURE": f,
                            "PSI": psi
                        })
            else:
                for f in self.features:
                    psi = self._feature_psi(self.df_ref[f], df_cur_d[f])
                    results.append({
                        "PROCESS_DATE": date,
                        "FEATURE": f,
                        "PSI": psi
                    })

        return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    df_ref = pd.DataFrame({
        "PROCESS_DATE": ["2024-01-01"] * 100,
        "FEATURE1": np.random.normal(0, 1, 100),
        "FEATURE2": np.random.choice(["A", "B", "C"], 100)
    })

    df_cur = pd.DataFrame({
        "PROCESS_DATE": ["2024-01-02"] * 100,
        "FEATURE1": np.random.normal(0.5, 1, 100),
        "FEATURE2": np.random.choice(["A", "B", "C"], 100)
    })

    psi_calculator = FeaturePSIByDate(
        df_ref, df_cur, features=["FEATURE1", "FEATURE2"])
    psi_results = psi_calculator.compute()
    print(psi_results)

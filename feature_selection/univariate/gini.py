from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from optbinning import BinningProcess
import pandas as pd


class SelectByGini(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        gini_min=0.2,
        gini_max=None,
        n_jobs=-1,
        binning_process_params=None,
        verbose=True,
        monotonic_trends=None,
        user_splits=None,
    ):
        """
        Initializes the transformer with Gini threshold for variable selection.

        Args:
            gini_min (float): Minimum Gini threshold for variable selection. Default is 0.2.
            gini_max (float, optional): Maximum Gini threshold for variable selection. Default is None.
            binning_process_params (dict, optional): Parameters to pass to BinningProcess. Default is None.
        """
        self.gini_min = gini_min
        self.gini_max = gini_max
        self.n_jobs = n_jobs
        self.binning_process_params = binning_process_params or {}
        self.selected_features = None
        self.binner = None
        self.gini_values = None
        self.verbose = verbose
        self.monotonic_trends = monotonic_trends or {}
        self.user_splits = user_splits or {}

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

    def fit(self, X, y=None):
        """
        Fits the transformer to the dataset by calculating Gini values and determining which columns to select.

        Args:
            X (DataFrame): Input DataFrame with features.
            y (array-like): Target variable for Gini calculation.

        Returns:
            self
        """
        X = self._to_dataframe(X)

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

        # Get Gini values from binning process
        self.gini_values = {}
        self.selected_features = []

        for variable in self.feature_names_in_:
            try:
                # Get the binning table for each variable
                binning_obj = self.binner.get_binned_variable(variable)

                # Access Gini from binning table
                binning_table = binning_obj.binning_table
                gini_value = binning_table.gini

                self.gini_values[variable] = gini_value

                # Select features based on Gini threshold
                if self.gini_min <= gini_value:
                    if self.gini_max is None or gini_value <= self.gini_max:
                        self.selected_features.append(variable)

            except Exception as e:
                print(
                    f"Could not calculate Gini for variable {variable}: {str(e)}")
                # Use -1 to indicate error in Gini calculation
                self.gini_values[variable] = -1.0

        return self

    def transform(self, X):
        """
        Transforms the dataset by retaining only the selected columns based on Gini values.

        Args:
            X (DataFrame): Input DataFrame.

        Returns:
            DataFrame: Transformed DataFrame with only selected columns.
        """
        check_is_fitted(self, "selected_features")

        if not self.selected_features:
            print(
                f"No columns meet the Gini thresholds of {self.gini_min} <= Gini <= {self.gini_max}. Returning empty DataFrame."
            )
            return pd.DataFrame(index=X.index)

        return X[self.selected_features]

    def get_feature_names_in(self):
        """
        Returns the list of all input features.

        Returns:
            list: List of feature names in the input DataFrame.
        """
        return self.feature_names_in_

    def get_feature_names_out(self):
        """
        Returns the list of selected features based on Gini values.

        Returns:
            list: List of feature names that were retained after Gini-based selection.
        """
        check_is_fitted(self, "selected_features")
        return self.selected_features

    def get_gini_values(self):
        """
        Returns the calculated Gini values for all features.

        Returns:
            dict: Dictionary mapping feature names to their Gini values.
        """
        check_is_fitted(self, "gini_values")
        return self.gini_values

    def get_binning_process(self):
        """
        Returns the fitted BinningProcess object.

        Returns:
            BinningProcess: The fitted binning process object.
        """
        check_is_fitted(self, "binner")
        return self.binner

    def get_selection_summary(self):
        """
        Returns a summary table of why each feature was selected or not based on its Gini value.

        Returns:
            pd.DataFrame: DataFrame containing feature names, Gini values, and selection reasons.
        """
        check_is_fitted(self, "gini_values")

        summary_data = []

        for feature in self.feature_names_in_:
            gini_value = self.gini_values.get(feature, 0.0)

            in_min = gini_value >= self.gini_min
            in_max = self.gini_max is None or gini_value <= self.gini_max
            selected = "Selected" if in_min and in_max else "Not Selected"

            if in_min and in_max:
                reason = f"{self.gini_min} <= Gini <= {self.gini_max}" if self.gini_max is not None else f"Gini >= {self.gini_min}"
            elif not in_min:
                reason = f"Gini < {self.gini_min}"
            else:
                reason = f"Gini > {self.gini_max}"

            summary_data.append(
                {
                    "feature": feature,
                    "gini": gini_value,
                    "selection_status": selected,
                    "reason": reason,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        return summary_df.sort_values(by="gini", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, random_state=42)
    selector = SelectByGini(gini_min=0.1, gini_max=0.5)
    selector.fit(X, y)
    print(selector.get_selection_summary())

from sklearn.model_selection import cross_val_predict
from sklearn.base import TransformerMixin, BaseEstimator, clone
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils.validation import check_is_fitted, check_array
from optbinning import BinningProcess
from sklearn.utils.validation import check_is_fitted
from sklearn.base import TransformerMixin, BaseEstimator
import statsmodels.api as sm
import warnings
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection._base import SelectorMixin
from joblib import Parallel, delayed
from typing import Optional, List
import logging
import random
import time
import math

# Configure the logging module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingRateSelector(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        # Logging feature information
        self.feats = X.columns.tolist()
        self.cat_feats = X.select_dtypes(include=["object"]).columns.tolist()
        self.num_feats = [
            item for item in self.feats if item not in self.cat_feats]

        # Calculate missing rate for each feature
        def f_missing_rate(a): return a.isna().sum() / a.size
        self.missing_rate = X.apply(f_missing_rate)

        # Select features with missing rate below threshold
        self.selected_features = self.missing_rate[
            self.missing_rate < self.threshold
        ].index.to_list()

        # Store the remove reasons
        self.remove_reasons = {
            feat: (
                None
                if feat in self.selected_features
                else f"Missing rate above (>=) threshold {self.threshold}"
            )
            for feat in self.feats
        }
        return self

    def transform(self, X):
        # Select the features with missing rate below the threshold
        return X[self.selected_features]

    def get_feature_names_in(self):
        return self.feats

    def get_feature_names_out(self):
        return self.selected_features

    def get_remove_reasons(self):
        return self.remove_reasons

    def get_selection_summary(self):
        # Add a column for feature names and missing rates
        summary = self.missing_rate.to_frame(name="missing rate")
        summary["Feature Name"] = summary.index
        summary["Selection Status"] = summary.index.map(
            lambda x: "Selected" if x in self.selected_features else "Not Selected"
        )
        summary["Reason"] = summary.index.map(self.remove_reasons)
        return summary[
            ["Feature Name", "missing rate", "Selection Status", "Reason"]
        ].sort_values(by="missing rate", ascending=False)


class IdenticalRateSelector(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        # Logging feature information
        self.feats = X.columns.tolist()
        self.cat_feats = X.select_dtypes(include=["object"]).columns.tolist()
        self.num_feats = [
            item for item in self.feats if item not in self.cat_feats]

        # Compute identical rate excluding null values
        def f_idt_rate(a):
            non_null = a.dropna()
            if len(non_null) == 0:
                return 1  # No non-null values, return 1
            return non_null.value_counts().max() / len(non_null)

        self.identical_perc = X.apply(f_idt_rate)
        self.selected_features = self.identical_perc[
            self.identical_perc < self.threshold
        ].index.to_list()

        # Store the remove reasons
        self.remove_reasons = {
            feat: (
                None
                if feat in self.selected_features
                else f"Identical rate above (>=) threshold {self.threshold}"
            )
            for feat in self.feats
        }
        return self

    def transform(self, X):
        # Select the features with identical rate above the threshold
        return X[self.selected_features]

    def get_feature_names_in(self):
        return self.feats

    def get_feature_names_out(self):
        return self.selected_features

    def get_remove_reasons(self):
        return self.remove_reasons

    def get_selection_summary(self):
        # Add a column for feature names and identical rates
        summary = self.identical_perc.to_frame(name="identical rate")
        summary["Feature Name"] = summary.index
        summary["Selection Status"] = summary.index.map(
            lambda x: "Selected" if x in self.selected_features else "Not Selected"
        )
        summary["Reason"] = summary.index.map(self.remove_reasons)
        return summary[
            ["Feature Name", "identical rate", "Selection Status", "Reason"]
        ].sort_values(by="identical rate", ascending=False)


class SelectByIV(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        iv_threshold=0.1,
        binning_process_params=None,
        verbose=True,
        monotonic_trends=None,
        user_splits=None,
    ):
        """
        Initializes the transformer with IV threshold for variable selection.

        Args:
            iv_threshold (float): Minimum Information Value threshold for variable selection. Default is 0.1.
            binning_process_params (dict, optional): Parameters to pass to BinningProcess. Default is None.
        """
        self.iv_threshold = iv_threshold
        self.binning_process_params = binning_process_params or {}
        self.selected_features = None  # Initialize the selected_features attribute
        self.binner = None
        self.iv_values = None
        self.verbose = verbose
        self.monotonic_trends = monotonic_trends or {}
        self.user_splits = user_splits or {}

    def fit(self, X, y=None):
        """
        Fits the transformer to the dataset by calculating IV values and determining which columns to select.

        Args:
            X (DataFrame): Input DataFrame with features.
            y (array-like): Target variable for IV calculation.

        Returns:
            self
        """
        if y is None:
            raise ValueError(
                "Target variable y is required for IV calculation.")

        self.feats = X.columns.tolist()
        self.categorical_feats = X.select_dtypes(
            include="object").columns.to_list()
        # Initialize BinningProcess with provided parameters
        # Build binning params for each variable
        self.binning_fit_params_full = {}
        for col in self.feats:
            if col not in self.categorical_feats:
                # Start with base params
                params = self.binning_process_params.copy()

                # Override monotonic_trend if specified for this variable
                if col in self.monotonic_trends:
                    params["monotonic_trend"] = self.monotonic_trends.get(
                        col, "auto")
                if col in self.user_splits:
                    params["user_splits"] = self.user_splits.get(col)

                self.binning_fit_params_full[col] = params

        self.binner = BinningProcess(
            n_jobs=-1,
            variable_names=[col for col in self.feats],
            binning_fit_params=self.binning_fit_params_full,
            categorical_variables=self.categorical_feats,
        )
        self.binner.fit(X, y)

        # Get IV values from binning process
        self.iv_values = {}
        self.selected_features = []

        for variable in self.feats:
            try:
                # Get the binning table for each variable
                binning_obj = self.binner.get_binned_variable(variable)

                # Access IV from binning table
                binning_table = binning_obj.binning_table
                if hasattr(binning_table, "iv"):
                    iv_value = binning_table.iv
                elif "IV" in binning_table.build().columns:
                    iv_value = binning_table.build().loc["Totals", "IV"]

                self.iv_values[variable] = iv_value

                # Select features based on IV threshold
                if iv_value >= self.iv_threshold:
                    self.selected_features.append(variable)

            except Exception as e:
                print(
                    f"Could not calculate IV for variable {variable}: {str(e)}")
                self.iv_values[variable] = 0.0

        if self.verbose:
            print("=" * 80)
            print(f"Select by IV:")
            print(f"IV values calculated: {self.iv_values}")
            print(
                f"Number of selected features: {len(self.selected_features)}")
            print(
                f"Selected columns (IV >= {self.iv_threshold}): {self.selected_features}"
            )

        return self

    def transform(self, X):
        """
        Transforms the dataset by retaining only the selected columns based on IV values.

        Args:
            X (DataFrame): Input DataFrame.

        Returns:
            DataFrame: Transformed DataFrame with only selected columns.
        """
        check_is_fitted(
            self, "selected_features")  # Check if fit() has been called

        if not self.selected_features:
            print(
                f"No columns meet the IV threshold of {self.iv_threshold}. Returning empty DataFrame."
            )
            # Return empty DataFrame with same index
            return pd.DataFrame(index=X.index)

        return X[self.selected_features]

    def get_feature_names_in(self):
        """
        Returns the list of all input features.

        Returns:
            list: List of feature names in the input DataFrame.
        """
        return self.feats

    def get_feature_names_out(self):
        """
        Returns the list of selected features based on IV values.

        Returns:
            list: List of feature names that were retained after IV-based selection.
        """
        check_is_fitted(
            self, "selected_features")  # Check if fit() has been called
        return self.selected_features

    def get_iv_values(self):
        """
        Returns the calculated IV values for all features.

        Returns:
            dict: Dictionary mapping feature names to their IV values.
        """
        check_is_fitted(self, "iv_values")  # Check if fit() has been called
        return self.iv_values

    def get_binning_process(self):
        """
        Returns the fitted BinningProcess object.

        Returns:
            BinningProcess: The fitted binning process object.
        """
        check_is_fitted(self, "binner")  # Check if fit() has been called
        return self.binner

    def get_selection_summary(self):
        """
        Returns a summary table of why each feature was selected or not based on its IV value.

        Returns:
            pd.DataFrame: DataFrame containing feature names, IV values, and selection reasons.
        """
        check_is_fitted(self, "iv_values")  # Check if fit() has been called

        summary_data = []

        for feature in self.feats:
            iv_value = self.iv_values.get(feature, 0.0)
            selected = "Selected" if iv_value >= self.iv_threshold else "Not Selected"
            reason = (
                f"IV >= {self.iv_threshold}"
                if iv_value >= self.iv_threshold
                else f"IV < {self.iv_threshold} (IV = {iv_value:.4f})"
            )

            summary_data.append(
                {
                    "feature": feature,
                    "iv": iv_value,
                    "selection_status": selected,
                    "reason": reason,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        return summary_df


class SelectByVIF(TransformerMixin, BaseEstimator):
    def __init__(self, vif_threshold=5.0, verbose=True):
        """
        Initializes the transformer with VIF threshold for variable selection.

        Args:
            vif_threshold (float): Maximum VIF threshold for variable selection. Default is 5.0.
            verbose (bool): Whether to print progress information. Default is True.
        """
        self.vif_threshold = vif_threshold
        self.selected_features = None  # Initialize the selected_features attribute
        self.vif_values = None
        self.removed_features = None
        self.removed_vif_values = None  # Store VIF values for removed features
        self.verbose = verbose
        self.feature_names_in_ = None  # Store input feature names
        self.is_dataframe_input = None  # Track input type
        self.selected_feature_indices_ = (
            None  # Store indices for array-based transforms
        )

    def _calculate_vif(self, X, n_jobs=-1):
        """Calculate VIF for all features in X using parallel processing."""

        def _vif_for_feature(i):
            try:
                vif_value = variance_inflation_factor(X, i)
                if np.isinf(vif_value) or np.isnan(vif_value):
                    return np.inf
                return vif_value
            except Exception:
                return np.inf

        vif_data = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_vif_for_feature)(i) for i in range(X.shape[1])
        )

        return np.array(vif_data)

    def _validate_and_convert_input(self, X):
        """
        Validate input and convert to appropriate format.

        Args:
            X: Input data (DataFrame or array-like)

        Returns:
            tuple: (X_array, feature_names, is_dataframe)
        """
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values.astype(np.float64)
            is_dataframe = True
        else:
            # Handle numpy arrays and other array-like inputs
            X_array = check_array(X, dtype=np.float64, ensure_2d=True)
            n_features = X_array.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]
            is_dataframe = False

        return X_array, feature_names, is_dataframe

    def fit(self, X, y=None):
        """
        Fits the transformer to the dataset by calculating VIF values and determining which columns to select.

        Args:
            X (DataFrame or array-like): Input data with features.
            y (array-like, optional): Target variable (not used for VIF calculation).

        Returns:
            self
        """
        # Validate and convert input
        X_array, feature_names, is_dataframe = self._validate_and_convert_input(
            X)

        # Store input information
        self.feature_names_in_ = feature_names
        self.is_dataframe_input = is_dataframe
        self.feats = feature_names  # Keep for backward compatibility

        # Check if we have enough samples for VIF calculation
        if X_array.shape[0] <= X_array.shape[1]:
            warnings.warn(
                f"Number of samples ({X_array.shape[0]}) is less than or equal to "
                f"number of features ({X_array.shape[1]}). VIF calculation may be unreliable.",
                UserWarning,
            )

        # Initialize tracking variables
        self.removed_features = []
        self.removed_vif_values = {}  # Store VIF values for removed features
        remaining_features = feature_names.copy()

        # Add constant term for VIF calculation (required by statsmodels)
        X_with_const = np.column_stack([np.ones(X_array.shape[0]), X_array])
        current_feature_indices = list(
            range(1, X_with_const.shape[1])
        )  # Skip constant term

        if self.verbose:
            print("=" * 80)
            print("Select by VIF:")

        iteration = 0
        while (
            len(current_feature_indices) > 0
        ):  # Continue while there are features left to consider
            iteration += 1
            # Calculate VIF for current features
            X_current = X_with_const[
                :, [0] + current_feature_indices
            ]  # Include constant
            vif_values = self._calculate_vif(X_current)

            # Skip constant term (index 0) when finding max VIF
            feature_vifs = vif_values[1:]  # VIF values for actual features

            if len(feature_vifs) == 0:
                break

            max_vif_idx = np.argmax(feature_vifs)
            max_vif_value = feature_vifs[max_vif_idx]

            if max_vif_value <= self.vif_threshold:
                # All remaining features have acceptable VIF
                break

            # Remove feature with highest VIF
            feature_to_remove_idx = current_feature_indices[max_vif_idx]
            original_feature_idx = (
                feature_to_remove_idx - 1
            )  # Convert back to original indexing
            feature_name = feature_names[original_feature_idx]

            self.removed_features.append(feature_name)
            self.removed_vif_values[feature_name] = (
                max_vif_value  # Save the VIF of the removed feature
            )
            remaining_features.remove(feature_name)
            current_feature_indices.remove(feature_to_remove_idx)

            if self.verbose:
                print(
                    f"Iteration {iteration}: Removed feature '{feature_name}' (VIF = {max_vif_value:.3f})"
                )

        # Set selected features
        self.selected_features = remaining_features

        # Store selected feature indices for array-based transforms
        self.selected_feature_indices_ = [
            feature_names.index(feat) for feat in self.selected_features
        ]

        # Calculate final VIF values for selected features
        self.vif_values = {}
        if len(self.selected_features) > 0:
            selected_indices = [
                feature_names.index(feat) for feat in self.selected_features
            ]
            X_selected = X_array[:, selected_indices]
            X_final = np.column_stack(
                [np.ones(X_selected.shape[0]), X_selected])
            final_vifs = self._calculate_vif(X_final)

            # Map VIF values to feature names (excluding constant term)
            for i, feature_name in enumerate(self.selected_features):
                # Skip constant term
                self.vif_values[feature_name] = final_vifs[i + 1]

        if self.verbose:
            print(f"VIF values calculated: {self.vif_values}")
            print(
                f"Number of selected features: {len(self.selected_features)}")
            print(
                f"Selected columns (VIF <= {self.vif_threshold}): {self.selected_features}"
            )
            if self.removed_features:
                print(f"Removed features: {self.removed_features}")

        return self

    def transform(self, X):
        """
        Transforms the dataset by retaining only the selected columns based on VIF values.
        Works with both DataFrame and numpy array inputs regardless of fit input type.

        Args:
            X (DataFrame or array-like): Input data.

        Returns:
            DataFrame or ndarray: Transformed data with only selected columns.
                                 Returns DataFrame if input is DataFrame, otherwise numpy array.
        """
        check_is_fitted(
            self, "selected_features")  # Check if fit() has been called

        # Handle empty selection case first
        if not self.selected_features:
            if self.verbose:
                print(
                    f"No columns meet the VIF threshold of {self.vif_threshold}. Returning empty data."
                )
            if isinstance(X, pd.DataFrame):
                return pd.DataFrame(index=X.index)
            else:
                X_array = check_array(X, dtype=np.float64, ensure_2d=True)
                return np.empty((X_array.shape[0], 0))

        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            # Check if DataFrame has the required columns
            available_features = X.columns.tolist()
            missing_features = [
                feat
                for feat in self.selected_features
                if feat not in available_features
            ]

            if missing_features:
                # If fitted on DataFrame but transform DataFrame is missing columns, fall back to position-based selection
                if len(X.columns) == len(self.feature_names_in_):
                    warnings.warn(
                        f"Some selected features {missing_features} not found in DataFrame columns. "
                        f"Using position-based selection instead.",
                        UserWarning,
                    )
                    # Use position-based selection
                    X_transformed = X.iloc[:, self.selected_feature_indices_]
                    # Create new column names based on selected features from fit
                    X_transformed.columns = self.selected_features
                    return X_transformed
                else:
                    raise ValueError(
                        f"Selected features {missing_features} not found in DataFrame columns "
                        f"and column count mismatch (got {len(X.columns)}, expected {len(self.feature_names_in_)}). "
                        f"Cannot perform transformation."
                    )
            else:
                # All selected features are available by name
                return X[self.selected_features]

        # Handle numpy array and other array-like inputs
        else:
            X_array = check_array(X, dtype=np.float64, ensure_2d=True)

            # Check if array has the correct number of features
            if X_array.shape[1] != len(self.feature_names_in_):
                raise ValueError(
                    f"Input array has {X_array.shape[1]} features, but transformer was fitted on "
                    f"{len(self.feature_names_in_)} features."
                )

            # Use stored indices for position-based selection
            X_transformed = X_array[:, self.selected_feature_indices_]
            return X_transformed

    def get_feature_names_in(self):
        """
        Returns the list of all input features.

        Returns:
            list: List of feature names in the input data.
        """
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_

    def get_feature_names_out(self, input_features=None):
        """
        Returns the list of selected features based on VIF values.

        Args:
            input_features: Not used, kept for sklearn compatibility.

        Returns:
            list: List of feature names that were retained after VIF-based selection.
        """
        check_is_fitted(
            self, "selected_features")  # Check if fit() has been called
        return self.selected_features

    def get_vif_values(self):
        """
        Returns the calculated VIF values for selected features.

        Returns:
            dict: Dictionary mapping feature names to their VIF values.
        """
        check_is_fitted(self, "vif_values")  # Check if fit() has been called
        return self.vif_values

    def get_removed_features(self):
        """
        Returns the list of features that were removed due to high VIF.

        Returns:
            list: List of feature names that were removed.
        """
        check_is_fitted(
            self, "removed_features")  # Check if fit() has been called
        return self.removed_features

    def get_removed_vif_values(self):
        """
        Returns the VIF values for removed features.

        Returns:
            dict: Dictionary mapping removed feature names to their VIF values.
        """
        check_is_fitted(
            self, "removed_vif_values")  # Check if fit() has been called
        return self.removed_vif_values

    def get_selection_summary(self):
        """
        Returns a summary table of why each feature was selected or not based on its VIF value.

        Returns:
            pd.DataFrame: DataFrame containing feature names, VIF values, and selection reasons.
        """
        check_is_fitted(self, "vif_values")  # Check if fit() has been called

        summary_data = []

        # Include all features (selected and removed)
        for feature in self.feature_names_in_:
            if feature in self.selected_features:
                vif_value = self.vif_values.get(feature, np.inf)
                selected = "Selected"
                reason = f"VIF <= {self.vif_threshold}"
            else:
                vif_value = self.removed_vif_values.get(feature, np.inf)
                selected = "Not Selected"
                reason = f"VIF > {self.vif_threshold} (VIF = {vif_value:.4f})"

            summary_data.append(
                {
                    "feature": feature,
                    "vif": vif_value,
                    "selection_status": selected,
                    "reason": reason,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        return summary_df


class SelectByCorrAUC(TransformerMixin, BaseEstimator):
    """
    Feature selector that filters variables by correlation threshold,
    then selects the best feature from each correlated group based on AUC score.

    Parameters
    ----------
    corr_threshold : float, default=0.7
        Correlation threshold for grouping features.
    corr_method : str, default='spearman'
        Method for computing correlation ('pearson', 'spearman', 'kendall').
    auc_method : str, default='ovr'
        Multi-class AUC method ('ovr' or 'ovo').
    inverted : bool, default=True
        Whether to invert feature values when computing AUC.
    model : sklearn estimator or None, default=None
        If provided, fits this model on each individual feature and uses
        predicted probabilities to compute AUC. If None, uses raw feature values.
        Model should have a predict_proba method.
    cv : int, cross-validation generator or None, default=5
        Cross-validation strategy when using a model. Can be:
        - None or 0: No cross-validation (fit on entire dataset)
        - int: Number of folds for StratifiedKFold
        - cross-validation generator: Custom CV splitter
        Ignored when model is None.
    """

    def __init__(
        self,
        corr_threshold=0.7,
        corr_method="spearman",
        auc_method="ovr",
        inverted=True,
        model=None,
        cv=5,
    ):
        self.corr_threshold = corr_threshold
        self.corr_method = corr_method
        self.auc_method = auc_method
        self.inverted = inverted
        self.model = model
        self.cv = cv

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            self.feature_names_in_ = X.columns.tolist()
        else:
            X_df = pd.DataFrame(X)
            self.feature_names_in_ = [
                f"feature_{i}" for i in range(X.shape[1])]
            X_df.columns = self.feature_names_in_

        corr_matrix = X_df.corr(method=self.corr_method).abs()
        self.corr_groups_ = self._identify_corr_groups(corr_matrix)

        self.feature_auc_scores_ = {}
        self.fitted_models_ = {}

        for col in X_df.columns:
            try:
                if self.model is not None:
                    # Use model-based AUC computation
                    auc, fitted_model = self._compute_model_auc(
                        X_df[[col]], y, len(np.unique(y))
                    )
                    self.fitted_models_[col] = fitted_model
                else:
                    # Use raw feature values for AUC computation
                    auc = self._compute_raw_auc(
                        X_df[col], y, len(np.unique(y)))

                self.feature_auc_scores_[col] = auc
            except (ValueError, Exception) as e:
                self.feature_auc_scores_[col] = 0.0

        self.selected_features_ = []
        for group in self.corr_groups_:
            group_aucs = {feat: self.feature_auc_scores_[
                feat] for feat in group}
            best_feature = max(group_aucs, key=group_aucs.get)
            self.selected_features_.append(best_feature)

        return self

    def _compute_raw_auc(self, feature_series, y, n_classes):
        """Compute AUC using raw feature values."""
        if n_classes == 2:
            if self.inverted:
                return roc_auc_score(y, -feature_series)
            else:
                return roc_auc_score(y, feature_series)
        else:
            if self.inverted:
                return roc_auc_score(y, -feature_series, multi_class=self.auc_method)
            else:
                return roc_auc_score(y, feature_series, multi_class=self.auc_method)

    def _compute_model_auc(self, X_feature, y, n_classes):
        """Compute AUC by fitting a model and using predicted probabilities with CV."""
        # Clone the model to avoid modifying the original
        model = clone(self.model)

        # Use cross-validation if cv is specified and > 0
        if self.cv is not None and self.cv > 0:
            # Use cross_val_predict to get out-of-fold predictions
            y_pred_proba = cross_val_predict(
                model, X_feature, y, cv=self.cv, method="predict_proba"
            )
        else:
            # No CV: fit on entire dataset
            model.fit(X_feature, y)
            y_pred_proba = model.predict_proba(X_feature)

        # Compute AUC based on number of classes
        if n_classes == 2:
            # For binary classification, use probability of positive class
            auc = roc_auc_score(y, y_pred_proba[:, 1])
        else:
            # For multi-class, use all probabilities
            auc = roc_auc_score(y, y_pred_proba, multi_class=self.auc_method)

        # Fit final model on entire dataset for potential future use
        model.fit(X_feature, y)

        return auc, model

    def _identify_corr_groups(self, corr_matrix):
        features = corr_matrix.columns.tolist()
        groups = []
        assigned = set()

        for feat in features:
            if feat in assigned:
                continue

            correlated = corr_matrix.index[
                (corr_matrix[feat] >= self.corr_threshold)
            ].tolist()

            group = [f for f in correlated if f not in assigned]
            groups.append(group)
            assigned.update(group)

        return groups

    def transform(self, X):
        check_is_fitted(self, ["selected_features_", "feature_auc_scores_"])

        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            indices = [
                self.feature_names_in_.index(feat) for feat in self.selected_features_
            ]
            return X[:, indices]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        return self.selected_features_

    def get_selection_summary(self):
        """
        Generates a summary DataFrame of the selection process for all input features.

        Returns
        -------
        summary_df : pd.DataFrame
            DataFrame containing feature name, AUC score, selection status, and reason.
        """
        check_is_fitted(
            self, ["selected_features_", "feature_auc_scores_", "corr_groups_"]
        )

        summary_list = []
        selected_set = set(self.selected_features_)

        # Create a reverse mapping for easy group lookup
        feature_to_group_index = {}
        for i, group in enumerate(self.corr_groups_):
            for feature in group:
                feature_to_group_index[feature] = i

        for feature in self.feature_names_in_:
            auc_score = self.feature_auc_scores_.get(feature, np.nan)

            # Determine the group for this feature
            group_index = feature_to_group_index.get(feature)
            group = (
                self.corr_groups_[
                    group_index] if group_index is not None else [feature]
            )
            group_members_str = f"Group: [{', '.join(group)}]"

            if feature in selected_set:
                status = "Selected"
                reason = f"Highest AUC within group. {group_members_str}"
            else:
                status = "Not Selected"

                if len(group) > 1:
                    best_feature = next(f for f in group if f in selected_set)
                    reason = f"Lower AUC than '{best_feature}'. {group_members_str}"
                else:
                    reason = f"Selected by default (Group size 1). {group_members_str}"

            summary_list.append(
                {
                    "Variable": feature,
                    "AUC_Score": auc_score,
                    "Selection_Status": status,
                    "Reason": reason,
                }
            )

        summary_df = pd.DataFrame(summary_list)
        return summary_df[["Variable", "Selection_Status", "AUC_Score", "Reason"]]


# Assuming BinningProcess is imported from a relevant library like 'optbinning'
# from optbinning import BinningProcess
# You need to ensure 'BinningProcess' is imported from your environment


class SelectByGini(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        gini_threshold=0.2,
        binning_process_params=None,
        verbose=True,
        monotonic_trends=None,
        user_splits=None,
    ):
        """
        Initializes the transformer with Gini threshold for variable selection.

        Args:
            gini_threshold (float): Minimum Gini threshold for variable selection. Default is 0.2.
                                    A common threshold for Gini is often 0.2 or higher.
            binning_process_params (dict, optional): Parameters to pass to BinningProcess. Default is None.
        """
        self.gini_threshold = gini_threshold
        self.binning_process_params = binning_process_params or {}
        self.selected_features = None  # Initialize the selected_features attribute
        self.binner = None
        self.gini_values = None
        self.verbose = verbose
        self.monotonic_trends = monotonic_trends or {}
        self.user_splits = user_splits or {}

    def fit(self, X, y=None):
        """
        Fits the transformer to the dataset by calculating Gini values and determining which columns to select.

        Args:
            X (DataFrame): Input DataFrame with features.
            y (array-like): Target variable for Gini calculation.

        Returns:
            self
        """
        if y is None:
            raise ValueError(
                "Target variable y is required for Gini calculation.")

        self.feats = X.columns.tolist()
        self.categorical_feats = X.select_dtypes(
            include="object").columns.to_list()

        # Build binning params for each variable
        self.binning_fit_params_full = {}
        for col in self.feats:
            if col not in self.categorical_feats:
                # Start with base params
                params = self.binning_process_params.copy()

                # Override monotonic_trend if specified for this variable
                if col in self.monotonic_trends:
                    params["monotonic_trend"] = self.monotonic_trends.get(
                        col, "auto")

                if col in self.user_splits:
                    params["user_splits"] = self.user_splits.get(col)

                self.binning_fit_params_full[col] = params

        self.binner = BinningProcess(
            n_jobs=-1,
            variable_names=[col for col in self.feats],
            binning_fit_params=self.binning_fit_params_full,
            categorical_variables=self.categorical_feats,
        )
        self.binner.fit(X, y)

        # Get Gini values from binning process
        self.gini_values = {}
        self.selected_features = []

        for variable in self.feats:
            try:
                # Get the binning table for each variable
                binning_obj = self.binner.get_binned_variable(variable)

                # Access Gini from binning table
                # NOTE: This assumes the binning table has a 'gini' attribute or 'Gini' column.
                binning_table = binning_obj.binning_table

                gini_value = -1  # Default value if not found

                # Check for direct 'gini' attribute
                if hasattr(binning_table, "gini"):
                    gini_value = binning_table.gini

                self.gini_values[variable] = gini_value

                # Select features based on Gini threshold
                if gini_value >= self.gini_threshold:
                    self.selected_features.append(variable)

            except Exception as e:
                if self.verbose:
                    print(
                        f"Could not calculate Gini for variable {variable}: {str(e)}")
                self.gini_values[variable] = -1

        if self.verbose:
            print("=" * 80)
            print(f"Select by Gini:")
            print(f"Gini values calculated: {self.gini_values}")
            print(
                f"Number of selected features: {len(self.selected_features)}")
            print(
                f"Selected columns (Gini >= {self.gini_threshold}): {self.selected_features}"
            )

        return self

    def transform(self, X):
        """
        Transforms the dataset by retaining only the selected columns based on Gini values.

        Args:
            X (DataFrame): Input DataFrame.

        Returns:
            DataFrame: Transformed DataFrame with only selected columns.
        """
        check_is_fitted(
            self, "selected_features")  # Check if fit() has been called

        if not self.selected_features:
            print(
                f"No columns meet the Gini threshold of {self.gini_threshold}. Returning empty DataFrame."
            )
            # Return empty DataFrame with same index
            return pd.DataFrame(index=X.index)

        return X[self.selected_features]

    def get_feature_names_in(self):
        """
        Returns the list of all input features.

        Returns:
            list: List of feature names in the input DataFrame.
        """
        return self.feats

    def get_feature_names_out(self):
        """
        Returns the list of selected features based on Gini values.

        Returns:
            list: List of feature names that were retained after Gini-based selection.
        """
        check_is_fitted(
            self, "selected_features")  # Check if fit() has been called
        return self.selected_features

    def get_gini_values(self):
        """
        Returns the calculated Gini values for all features.

        Returns:
            dict: Dictionary mapping feature names to their Gini values.
        """
        check_is_fitted(self, "gini_values")  # Check if fit() has been called
        return self.gini_values

    def get_binning_process(self):
        """
        Returns the fitted BinningProcess object.

        Returns:
            BinningProcess: The fitted binning process object.
        """
        check_is_fitted(self, "binner")  # Check if fit() has been called
        return self.binner

    def get_selection_summary(self):
        """
        Returns a summary table of why each feature was selected or not based on its Gini value.

        Returns:
            pd.DataFrame: DataFrame containing feature names, Gini values, and selection reasons.
        """
        check_is_fitted(self, "gini_values")  # Check if fit() has been called

        summary_data = []

        for feature in self.feats:
            gini_value = self.gini_values.get(feature, 0.0)
            selected = (
                "Selected" if gini_value >= self.gini_threshold else "Not Selected"
            )
            reason = (
                f"Gini >= {self.gini_threshold}"
                if gini_value >= self.gini_threshold
                else f"Gini < {self.gini_threshold} (Gini = {gini_value:.4f})"
            )

            summary_data.append(
                {
                    "feature": feature,
                    "gini": gini_value,
                    "selection_status": selected,
                    "reason": reason,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        return summary_df


class ConstrainedBeamSearchSelector(BaseEstimator, TransformerMixin):
    """
    Constrained Beam Search Feature Selection với GINI optimization tuân thủ scikit-learn API.

    Class này thực hiện beam search - một balanced approach giữa greedy forward selection
    và exhaustive combination search. Beam search maintain nhiều promising paths song song,
    cho phép explore nhiều hơn forward selection nhưng nhanh hơn nhiều so với exhaustive search.

    Beam Search hoạt động như sau:
    - Bắt đầu với KHÔNG có feature nào
    - Mỗi iteration: expand mỗi path trong beam bằng cách thêm 1 feature
    - Evaluate tất cả candidate paths (beam_width × n_remaining features)
    - Giữ lại top beam_width paths tốt nhất
    - Lặp lại cho đến đạt k features
    - Chọn best path từ final beam

    So với Forward Selection:
    - Forward: 1 path, deterministic, fast nhưng dễ stuck
    - Beam: beam_width paths, explore nhiều directions, tốt hơn nhưng chậm hơn beam_width lần

    So với Exhaustive Search:
    - Exhaustive: test TẤT CẢ C(n,k) combinations (~10 tỷ với n=50, k=10)
    - Beam: test ~(n × k × beam_width) combinations (~50K với n=50, k=10, beam=100)

    Parameters:
        k: Số features trong mỗi combination (required)
        beam_width: Số paths giữ lại mỗi iteration (mặc định: 50)
        vif_threshold: Ngưỡng VIF tối đa cho phép (mặc định: 10.0)
        weight_min: Trọng số tối thiểu của feature (mặc định: 0.05)
        weight_max: Trọng số tối đa của feature (mặc định: 0.30)
        p_value_threshold: Ngưỡng p-value tối đa (mặc định: 0.05)
        require_negative_coef: Yêu cầu coefficients phải âm cho PD models (mặc định: True)
        top_n: Số top paths giữ lại trong final beam để compare (mặc định: 10)
        n_jobs: Số CPU cores cho parallel processing (mặc định: 1)
        verbose: Nếu True, in thông tin chi tiết trong quá trình search

    Attributes:
        selected_features_: List các features của best path
        model_: Statsmodels Logit model được fit với best path
        gini_score_: GINI score của best path
        beam_history_: List chứa beam evolution qua các iterations
        final_beam_: DataFrame chứa top paths trong final beam
        total_tested_: Tổng số paths đã evaluate
        n_features_in_: Số features ban đầu
        feature_names_in_: Tên các features ban đầu
        vif_df_: DataFrame chứa VIF values của best path
        weight_df_: DataFrame chứa weight values của best path

    Example:
        >>> # Beam Search với 10 features, beam width = 50
        >>> selector = ConstrainedBeamSearchSelector(
        ...     k=10,
        ...     beam_width=50,
        ...     vif_threshold=10.0,
        ...     weight_min=0.05,
        ...     weight_max=0.30,
        ...     p_value_threshold=0.05,
        ...     require_negative_coef=True,
        ...     n_jobs=-1,
        ...     verbose=True
        ... )
        >>> selector.fit(X_train, y_train)
        >>>
        >>> # Xem final beam paths
        >>> final_paths = selector.get_final_beam()
        >>> print(f"Best GINI: {selector.gini_score_:.4f}")
        >>> print(f"Total tested: {selector.total_tested_}")
    """

    def __init__(
        self,
        k: int,
        beam_width: int = 5,
        vif_threshold: float = 10.0,
        weight_min: float = 0.05,
        weight_max: float = 0.30,
        p_value_threshold: float = 0.05,
        require_negative_coef: bool = True,
        top_n: int = 10,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """
        Initialize ConstrainedBeamSearchSelector.

        Args:
            k: Số features trong final combination (REQUIRED)
            beam_width: Số paths giữ lại mỗi iteration
            vif_threshold: Ngưỡng VIF tối đa
            weight_min: Trọng số tối thiểu
            weight_max: Trọng số tối đa
            p_value_threshold: Ngưỡng p-value tối đa
            require_negative_coef: Yêu cầu coefficients âm
            top_n: Số top paths trong final beam để track
            n_jobs: Số CPU cores cho parallel processing
            verbose: Enable logging
        """
        if k < 1:
            raise ValueError(f"k phải >= 1, nhận được {k}")
        if beam_width < 1:
            raise ValueError(f"beam_width phải >= 1, nhận được {beam_width}")

        self.k = k
        self.beam_width = beam_width
        self.vif_threshold = vif_threshold
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.p_value_threshold = p_value_threshold
        self.require_negative_coef = require_negative_coef
        self.top_n = top_n
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Attributes được set sau khi fit
        self.selected_features_: Optional[List[str]] = None
        self.model_ = None
        self.gini_score_: Optional[float] = None
        self.beam_history_: List[List[dict]] = []
        self.final_beam_: Optional[pd.DataFrame] = None
        self.total_tested_: int = 0
        self.unique_tested_: int = 0
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.vif_df_: Optional[pd.DataFrame] = None
        self.weight_df_: Optional[pd.DataFrame] = None

    def _calc_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """Tính Variance Inflation Factor cho các features."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_df = pd.DataFrame()
        vif_df["Variable"] = X.columns
        vif_df["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]
        return vif_df

    def _calc_weight(self, model, X: pd.DataFrame) -> tuple:
        """Tính trọng số (weight) từ standardized coefficients."""
        import math

        LOGIT_SCALE_FACTOR = math.sqrt(3) / math.pi

        coef = model.params.drop("const", errors="ignore")
        std_devs = X.std()
        standardized_coef = std_devs * coef * LOGIT_SCALE_FACTOR

        abs_std_coef = np.abs(standardized_coef)
        weights = abs_std_coef / abs_std_coef.sum()

        weight_ok = np.all((weights >= self.weight_min) &
                           (weights <= self.weight_max))

        weight_df = pd.DataFrame(
            {
                "Variable": X.columns,
                "Coef.": coef.values,
                "std_devs": std_devs.values,
                "standardized_coef": standardized_coef.values,
                "weight": weights.values,
            }
        )

        return weight_ok, weight_df

    def _check_constraints(self, model, X: pd.DataFrame) -> tuple:
        """
        Check tất cả constraints cho path hiện tại.

        Constraints được apply theo số lượng biến:
        - p-value và coefficient signs: Luôn check từ đầu
        - VIF và weight: Chỉ check khi có >= 5 biến

        Returns:
            Tuple (all_ok, reason, vif_df, weight_df)
        """
        n_features = X.shape[1]

        # Check p-values (LUÔN CHECK)
        pvals = model.pvalues.drop("const", errors="ignore")
        if (pvals >= self.p_value_threshold).any():
            worst_p = pvals.max()
            return False, f"p-value={worst_p:.4f}", None, None

        # Check coefficient signs (LUÔN CHECK nếu required)
        if self.require_negative_coef:
            coefs = model.params.drop("const", errors="ignore")
            if (coefs >= 0).any():
                bad_coef = coefs[coefs >= 0].iloc[0]
                return False, f"positive_coef={bad_coef:.4f}", None, None

        # Check VIF (CHỈ CHECK KHI >= 5 BIẾN)
        vif_df = None
        if n_features >= 5:
            vif_df = self._calc_vif(X)
            if (vif_df["VIF"] > self.vif_threshold).any():
                worst_vif = vif_df["VIF"].max()
                return False, f"VIF={worst_vif:.4f}", vif_df, None

        # Check weights (CHỈ CHECK KHI >= 5 BIẾN)
        weight_df = None
        if n_features >= 5:
            weight_ok, weight_df = self._calc_weight(model, X)
            if not weight_ok:
                bad_weights = weight_df[
                    (weight_df["weight"] < self.weight_min)
                    | (weight_df["weight"] > self.weight_max)
                ]
                if len(bad_weights) > 0:
                    worst_weight = bad_weights["weight"].iloc[0]
                    return False, f"weight={worst_weight:.4f}", vif_df, weight_df

        return True, "OK", vif_df, weight_df

    def _evaluate_path(
        self, features: tuple, X: pd.DataFrame, y: pd.Series
    ) -> Optional[dict]:
        """
        Evaluate một path và return kết quả nếu pass constraints.

        Args:
            features: Tuple các feature names trong path
            X: Feature matrix
            y: Target variable

        Returns:
            Dict chứa result info nếu pass constraints, None nếu fail
        """
        # Prepare data
        X_curr = X[list(features)]
        X_sm = sm.add_constant(X_curr)

        # Fit model
        try:
            model = sm.Logit(y, X_sm).fit(disp=0)
        except Exception:
            return None

        # Check constraints
        all_ok, reason, vif_df, weight_df = self._check_constraints(
            model, X_curr)

        if not all_ok:
            return None

        # Calculate GINI
        y_pred = model.predict(X_sm)
        gini = 2 * roc_auc_score(y, y_pred) - 1

        # Return result
        return {
            "features": list(features),
            "gini": gini,
            "model": model,
            "vif_df": vif_df,
            "weight_df": weight_df,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit constrained beam search selector trên training data.

        Beam Search Algorithm:
        1. Initialize beam với empty path
        2. Cho mỗi depth từ 1 đến k:
           a. Expand mỗi path trong current beam:
              - Thử thêm mỗi feature chưa có trong path
              - Fit model và check constraints
              - Tính GINI nếu pass constraints
           b. Collect tất cả valid candidate paths
           c. Sort theo GINI descending
           d. Giữ top beam_width paths làm new beam
        3. Return best path từ final beam

        Args:
            X: Feature matrix của training data
            y: Target variable của training data

        Returns:
            self: Fitted estimator

        Raises:
            ValueError: Nếu X hoặc y không hợp lệ, hoặc k > n
        """
        from joblib import Parallel, delayed

        # Validate input
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array")

        # Store feature names
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist()

        # Validate k
        if self.k > self.n_features_in_:
            raise ValueError(
                f"k={self.k} không thể lớn hơn số features={self.n_features_in_}"
            )

        if self.verbose:
            logger.info("=" * 100)
            logger.info(f"🔍 CONSTRAINED BEAM SEARCH")
            logger.info(
                f"Total features: {self.n_features_in_}, Target k: {self.k}")
            logger.info(f"Beam width: {self.beam_width}")
            logger.info(
                f"Constraints: VIF<={self.vif_threshold}, p-value<={self.p_value_threshold}, "
                f"weight range=[{self.weight_min}, {self.weight_max}], "
                f"negative coef={self.require_negative_coef}"
            )
            logger.info(
                f"Parallel jobs: {self.n_jobs if self.n_jobs > 0 else 'all CPUs'}"
            )
            logger.info("=" * 100)

        # Initialize
        self.total_tested_ = 0
        self.unique_tested_ = 0
        self.beam_history_ = []
        start_time = time.time()

        # Start with empty beam - sẽ expand trong iteration đầu
        current_beam = [{"features": tuple(), "gini": 0.0}]

        # Beam search iterations
        for depth in range(1, self.k + 1):
            if self.verbose:
                logger.info(f"\n{'='*80}")
                logger.info(f"📍 Depth {depth}/{self.k}: Expanding beam...")

            candidates = []

            # Step 1: Collect all candidate paths from all parent paths
            all_candidate_paths = set()
            total_generated = 0

            for path_info in current_beam:
                current_features = set(path_info["features"])
                remaining_features = set(
                    self.feature_names_in_) - current_features

                # Generate candidate paths by adding each remaining feature
                for new_feature in remaining_features:
                    new_path = tuple(sorted(current_features | {new_feature}))
                    all_candidate_paths.add(new_path)
                    total_generated += 1

            # Step 2: Deduplicated candidate paths (unique combinations only)
            unique_candidate_paths = list(all_candidate_paths)
            n_duplicates = total_generated - len(unique_candidate_paths)

            # Step 3: Evaluate unique candidates in parallel
            if self.n_jobs == 1:
                # Sequential
                results = []
                for candidate_path in unique_candidate_paths:
                    result = self._evaluate_path(candidate_path, X, y)
                    if result is not None:
                        results.append(result)
            else:
                # Parallel
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._evaluate_path)(candidate_path, X, y)
                    for candidate_path in unique_candidate_paths
                )
                results = [r for r in results if r is not None]

            candidates.extend(results)

            # Step 4: Update tracking counters
            self.total_tested_ += (
                total_generated  # Total attempts (includes duplicates)
            )
            # Unique evaluations
            self.unique_tested_ += len(unique_candidate_paths)

            # Check if we have valid candidates
            if not candidates:
                if self.verbose:
                    logger.warning(
                        f"⚠️  No valid candidates at depth {depth}. "
                        f"Stopping with {depth-1} features."
                    )
                    if current_beam and current_beam[0]["features"]:
                        logger.info(
                            f"📌 Will use best path from depth {depth-1} with "
                            f"{len(current_beam[0]['features'])} features and "
                            f"GINI={current_beam[0]['gini']:.4f}"
                        )
                    logger.warning(
                        f"💡 Suggestion: Constraints có thể quá strict. "
                        f"Hãy thử giảm VIF threshold, tăng p-value threshold, "
                        f"hoặc nới lỏng weight constraints."
                    )
                break

            # Sort by GINI and keep top beam_width
            candidates.sort(key=lambda x: x["gini"], reverse=True)
            current_beam = candidates[: self.beam_width]

            # Save beam state
            self.beam_history_.append(
                [
                    {
                        "depth": depth,
                        "features": path["features"],
                        "gini": path["gini"],
                        "n_features": len(path["features"]),
                    }
                    for path in current_beam
                ]
            )

            # Log progress
            if self.verbose:
                best_path = current_beam[0]
                elapsed = time.time() - start_time
                duplicate_pct = (
                    (n_duplicates / total_generated *
                     100) if total_generated > 0 else 0
                )
                logger.info(
                    f"✓ Evaluated {len(candidates)} valid candidates, kept top {len(current_beam)} | "
                    f"Unique: {len(unique_candidate_paths)}, Total: {total_generated}, "
                    f"Duplicates: {n_duplicates} ({duplicate_pct:.1f}%)"
                )
                logger.info(
                    f"🏆 Best at depth {depth}: GINI={best_path['gini']:.4f}, "
                    f"Features: {', '.join(best_path['features'][:5])}"
                    f"{' ...' if len(best_path['features']) > 5 else ''}"
                )
                overall_duplicate_pct = (
                    (self.total_tested_ - self.unique_tested_)
                    / self.total_tested_
                    * 100
                    if self.total_tested_ > 0
                    else 0
                )
                logger.info(
                    f"⏱️  Time: {elapsed:.1f}s, Unique tested: {self.unique_tested_:,}, "
                    f"Total: {self.total_tested_:,} ({overall_duplicate_pct:.1f}% duplicates)"
                )

        # Select best path from final beam
        if not current_beam or not current_beam[0]["features"]:
            logger.warning(
                "⚠️  No valid paths found! Returning empty selection.")
            logger.warning(
                "💡 Suggestion: All features fail constraints. "
                "Please relax constraints or check data quality."
            )
            self.selected_features_ = tuple()
            self.model_ = None
            self.gini_score_ = 0.0
            self.vif_df_ = None
            self.weight_df_ = None
            self.final_beam_ = pd.DataFrame()
            return self

        best_path = current_beam[0]
        self.selected_features_ = best_path["features"]

        # Refit model for best path to ensure we have all info
        # (in case best_path doesn't have model/vif/weight from initialization)
        if not self.selected_features_:
            logger.warning(
                "⚠️  Best path has no features! Returning all features.")
            self.selected_features_ = tuple(self.feature_names_in_)
            self.model_ = None
            self.gini_score_ = 0.0
            self.vif_df_ = None
            self.weight_df_ = None
            self.final_beam_ = pd.DataFrame()
            return self

        X_best = X[list(self.selected_features_)]
        X_sm = sm.add_constant(X_best)

        try:
            self.model_ = sm.Logit(y, X_sm).fit(disp=0)
            y_pred = self.model_.predict(X_sm)
            self.gini_score_ = 2 * roc_auc_score(y, y_pred) - 1

            # Recalculate VIF and weights for final model
            if X_best.shape[1] >= 2:
                self.vif_df_ = self._calc_vif(X_best)
            else:
                self.vif_df_ = None

            _, self.weight_df_ = self._calc_weight(self.model_, X_best)

        except Exception as e:
            if self.verbose:
                logger.error(f"Failed to refit best model: {str(e)}")
            # Fallback to stored values if available
            self.model_ = best_path.get("model")
            self.gini_score_ = best_path.get("gini", 0.0)
            self.vif_df_ = best_path.get("vif_df")
            self.weight_df_ = best_path.get("weight_df")

        # Create final beam summary
        self.final_beam_ = pd.DataFrame(
            [
                {
                    "rank": i + 1,
                    "gini": path["gini"],
                    "n_features": len(path["features"]),
                    "features": ", ".join(path["features"]),
                }
                for i, path in enumerate(current_beam[: self.top_n])
            ]
        )

        # Final logging
        if self.verbose:
            total_time = time.time() - start_time
            duplicate_count = self.total_tested_ - self.unique_tested_
            duplicate_pct = (
                (duplicate_count / self.total_tested_ * 100)
                if self.total_tested_ > 0
                else 0
            )
            logger.info("\n" + "=" * 100)
            logger.info(f"✅ BEAM SEARCH COMPLETED")
            logger.info(
                f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
            logger.info(
                f"📈 Paths evaluated: Unique={self.unique_tested_:,}, Total={self.total_tested_:,}, "
                f"Duplicates={duplicate_count:,} ({duplicate_pct:.1f}%)"
            )
            logger.info(
                f"⚡ Speed: {self.unique_tested_/total_time:.0f} unique paths/second"
            )
            logger.info(f"✓  Final beam size: {len(current_beam)}")
            logger.info("")
            logger.info(f"🏆 BEST Path (Rank #1):")
            logger.info(f"   GINI Score: {self.gini_score_:.4f}")
            logger.info(f"   Features ({len(self.selected_features_)}):")
            for i, feat in enumerate(self.selected_features_, 1):
                logger.info(f"     {i}. {feat}")
            logger.info("")
            logger.info("🥇 Top 5 Paths in Final Beam:")
            for _, row in self.final_beam_.head(5).iterrows():
                logger.info(f"   #{int(row['rank'])}: GINI={row['gini']:.4f}")
            logger.info("=" * 100)
            logger.info("\n" + str(self.model_.summary()))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X bằng cách chỉ giữ lại best path features."""
        if self.selected_features_ is None:
            raise ValueError(
                "Model chưa được fit. Gọi fit() trước khi transform().")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        return X[self.selected_features_]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability của positive class."""
        if self.model_ is None:
            raise ValueError(
                "Model chưa được fit. Gọi fit() trước khi predict().")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        X_selected = self.transform(X)
        return self.model_.predict(sm.add_constant(X_selected))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities cho X."""
        proba_positive = self.predict(X)
        proba_negative = 1 - proba_positive
        return np.column_stack([proba_negative, proba_positive])

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> np.ndarray:
        """Get output feature names cho transform."""
        if self.selected_features_ is None:
            raise ValueError("Model chưa được fit.")

        return np.array(self.selected_features_)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters của estimator."""
        return {
            "k": self.k,
            "beam_width": self.beam_width,
            "vif_threshold": self.vif_threshold,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "p_value_threshold": self.p_value_threshold,
            "require_negative_coef": self.require_negative_coef,
            "top_n": self.top_n,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "ConstrainedBeamSearchSelector":
        """Set parameters của estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_vif_table(self) -> pd.DataFrame:
        """Get VIF table của best path."""
        if self.vif_df_ is None:
            raise ValueError("Model chưa được fit.")
        return self.vif_df_.copy()

    def get_weight_table(self) -> pd.DataFrame:
        """Get weight table của best path."""
        if self.weight_df_ is None:
            raise ValueError("Model chưa được fit.")
        return self.weight_df_.copy()

    def get_final_beam(self) -> pd.DataFrame:
        """
        Get DataFrame chứa top N paths trong final beam.

        Returns:
            DataFrame với columns: rank, gini, n_features, features

        Raises:
            ValueError: Nếu model chưa được fit
        """
        if self.final_beam_ is None:
            raise ValueError("Model chưa được fit.")
        return self.final_beam_.copy()

    def get_beam_history(self) -> List[pd.DataFrame]:
        """
        Get history của beam evolution qua các depths.

        Returns:
            List of DataFrames, mỗi DataFrame chứa beam state tại một depth

        Raises:
            ValueError: Nếu model chưa được fit
        """
        if not self.beam_history_:
            raise ValueError("Model chưa được fit.")

        return [pd.DataFrame(beam_state) for beam_state in self.beam_history_]

    def get_search_summary(self) -> dict:
        """
        Get summary thống kê của quá trình beam search.

        Returns:
            Dictionary chứa thông tin summary bao gồm:
            - unique_paths_evaluated: Số unique combinations thực sự được evaluate
            - total_path_attempts: Tổng số attempts (bao gồm duplicates)
            - duplicate_paths_avoided: Số duplicates bị loại bỏ
            - duplicate_percentage: % duplicates trong tổng số attempts
        """
        if self.selected_features_ is None:
            raise ValueError("Model chưa được fit.")

        # Calculate duplicate statistics
        duplicate_count = self.total_tested_ - self.unique_tested_
        duplicate_pct = (
            (duplicate_count / self.total_tested_ * 100)
            if self.total_tested_ > 0
            else 0
        )

        # Calculate theoretical max unique combinations per depth
        max_unique_per_depth = sum(
            [
                len(self.beam_history_[i]) * (self.n_features_in_ - i - 1)
                for i in range(len(self.beam_history_))
            ]
        )

        return {
            "n_features_input": self.n_features_in_,
            "k": self.k,
            "beam_width": self.beam_width,
            "depths_completed": len(self.beam_history_),
            "unique_paths_evaluated": self.unique_tested_,
            "total_path_attempts": self.total_tested_,
            "duplicate_paths_avoided": duplicate_count,
            "duplicate_percentage": duplicate_pct,
            "max_possible_unique_evaluations": max_unique_per_depth,
            "final_beam_size": (
                len(self.final_beam_) if self.final_beam_ is not None else 0
            ),
            "best_gini": self.gini_score_,
            "avg_beam_size": (
                np.mean([len(beam) for beam in self.beam_history_])
                if self.beam_history_
                else 0
            ),
        }

    def get_selection_summary(self) -> pd.DataFrame:
        """
        Get summary của beam search selection process.

        Returns:
            DataFrame với feature names, selection status, và reasons.
        """
        if not hasattr(self, "feature_names_in_") or self.feature_names_in_ is None:
            raise ValueError("Model chưa được fit.")

        # Tạo summary list
        summary_list = []
        selected_set = (
            set(self.selected_features_)
            if self.selected_features_ is not None
            else set()
        )

        for feature in self.feature_names_in_:
            if feature in selected_set:
                status = "Selected"
                reason = f"Selected in best path (GINI: {self.gini_score_:.4f})"
            else:
                status = "Not Selected"
                reason = "Not in final beam search result"

            summary_list.append(
                {"Feature": feature, "Selection_Status": status, "Reason": reason}
            )

        summary_df = pd.DataFrame(summary_list)
        return summary_df[["Feature", "Selection_Status", "Reason"]]


class SelectByCorrAUC(TransformerMixin, BaseEstimator):
    """
    Feature selector that filters variables by correlation threshold,
    then selects the best feature from each correlated group based on AUC score.

    Parameters
    ----------
    corr_threshold : float, default=0.7
        Correlation threshold for grouping features.
    corr_method : str, default='spearman'
        Method for computing correlation ('pearson', 'spearman', 'kendall').
    auc_method : str, default='ovr'
        Multi-class AUC method ('ovr' or 'ovo').
    inverted : bool, default=True
        Whether to invert feature values when computing AUC.
    model : sklearn estimator or None, default=None
        If provided, fits this model on each individual feature and uses
        predicted probabilities to compute AUC. If None, uses raw feature values.
        Model should have a predict_proba method.
    cv : int, cross-validation generator or None, default=5
        Cross-validation strategy when using a model. Can be:
        - None or 0: No cross-validation (fit on entire dataset)
        - int: Number of folds for StratifiedKFold
        - cross-validation generator: Custom CV splitter
        Ignored when model is None.
    n_jobs : int, default=1
        Number of parallel jobs for AUC computation. -1 uses all processors.
        Only applies when model is provided.
    """

    def __init__(
        self,
        corr_threshold=0.7,
        corr_method="spearman",
        auc_method="ovr",
        inverted=True,
        model=None,
        cv=5,
        n_jobs=1,
    ):
        self.corr_threshold = corr_threshold
        self.corr_method = corr_method
        self.auc_method = auc_method
        self.inverted = inverted
        self.model = model
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            self.feature_names_in_ = X.columns.tolist()
        else:
            X_df = pd.DataFrame(X)
            self.feature_names_in_ = [
                f"feature_{i}" for i in range(X.shape[1])]
            X_df.columns = self.feature_names_in_

        corr_matrix = X_df.corr(method=self.corr_method).abs()
        self.corr_groups_ = self._identify_corr_groups(corr_matrix)

        n_classes = len(np.unique(y))

        if self.model is not None:
            # Parallel processing for model-based AUC computation
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_single_feature_auc)(
                    X_df[[col]], y, n_classes, col, use_model=True
                )
                for col in X_df.columns
            )

            self.feature_auc_scores_ = {}
            self.fitted_models_ = {}
            for col, auc, fitted_model in results:
                self.feature_auc_scores_[col] = auc
                self.fitted_models_[col] = fitted_model
        else:
            # Parallel processing for raw feature AUC computation
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_single_feature_auc)(
                    X_df[col], y, n_classes, col, use_model=False
                )
                for col in X_df.columns
            )

            self.feature_auc_scores_ = {col: auc for col, auc, _ in results}
            self.fitted_models_ = {}

        self.selected_features_ = []
        for group in self.corr_groups_:
            group_aucs = {feat: self.feature_auc_scores_[
                feat] for feat in group}
            best_feature = max(group_aucs, key=group_aucs.get)
            self.selected_features_.append(best_feature)

        return self

    def _compute_single_feature_auc(self, X_feature, y, n_classes, col_name, use_model):
        """
        Compute AUC for a single feature (used for parallel processing).

        Returns
        -------
        tuple : (column_name, auc_score, fitted_model or None)
        """
        try:
            if use_model:
                auc, fitted_model = self._compute_model_auc(
                    X_feature, y, n_classes)
                return col_name, auc, fitted_model
            else:
                auc = self._compute_raw_auc(X_feature, y, n_classes)
                return col_name, auc, None
        except (ValueError, Exception) as e:
            return col_name, 0.0, None

    def _compute_raw_auc(self, feature_series, y, n_classes):
        """Compute AUC using raw feature values."""
        if n_classes == 2:
            if self.inverted:
                return roc_auc_score(y, -feature_series)
            else:
                return roc_auc_score(y, feature_series)
        else:
            if self.inverted:
                return roc_auc_score(y, -feature_series, multi_class=self.auc_method)
            else:
                return roc_auc_score(y, feature_series, multi_class=self.auc_method)

    def _compute_model_auc(self, X_feature, y, n_classes):
        """Compute AUC by fitting a model and using predicted probabilities with CV."""
        # Clone the model to avoid modifying the original
        model = clone(self.model)

        # Use cross-validation if cv is specified and > 0
        if self.cv is not None and self.cv > 0:
            # Use cross_val_predict to get out-of-fold predictions
            y_pred_proba = cross_val_predict(
                model,
                X_feature,
                y,
                cv=self.cv,
                method="predict_proba",
                n_jobs=self.n_jobs,
            )
        else:
            # No CV: fit on entire dataset
            model.fit(X_feature, y)
            y_pred_proba = model.predict_proba(X_feature)

        # Compute AUC based on number of classes
        if n_classes == 2:
            # For binary classification, use probability of positive class
            auc = roc_auc_score(y, y_pred_proba[:, 1])
        else:
            # For multi-class, use all probabilities
            auc = roc_auc_score(y, y_pred_proba, multi_class=self.auc_method)

        # Fit final model on entire dataset for potential future use
        model.fit(X_feature, y)

        return auc, model

    def _identify_corr_groups(self, corr_matrix):
        features = corr_matrix.columns.tolist()
        groups = []
        assigned = set()

        for feat in features:
            if feat in assigned:
                continue

            correlated = corr_matrix.index[
                (corr_matrix[feat] >= self.corr_threshold)
            ].tolist()

            group = [f for f in correlated if f not in assigned]
            groups.append(group)
            assigned.update(group)

        return groups

    def transform(self, X):
        check_is_fitted(self, ["selected_features_", "feature_auc_scores_"])

        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            indices = [
                self.feature_names_in_.index(feat) for feat in self.selected_features_
            ]
            return X[:, indices]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        return self.selected_features_

    def get_selection_summary(self):
        """
        Generates a summary DataFrame of the selection process for all input features.

        Returns
        -------
        summary_df : pd.DataFrame
            DataFrame containing feature name, AUC score, selection status, and reason.
        """
        check_is_fitted(
            self, ["selected_features_", "feature_auc_scores_", "corr_groups_"]
        )

        summary_list = []
        selected_set = set(self.selected_features_)

        # Create a reverse mapping for easy group lookup
        feature_to_group_index = {}
        for i, group in enumerate(self.corr_groups_):
            for feature in group:
                feature_to_group_index[feature] = i

        for feature in self.feature_names_in_:
            auc_score = self.feature_auc_scores_.get(feature, np.nan)

            # Determine the group for this feature
            group_index = feature_to_group_index.get(feature)
            group = (
                self.corr_groups_[
                    group_index] if group_index is not None else [feature]
            )
            group_members_str = f"Group: [{', '.join(group)}]"

            if feature in selected_set:
                status = "Selected"
                reason = f"Highest AUC within group. {group_members_str}"
            else:
                status = "Not Selected"

                if len(group) > 1:
                    best_feature = next(f for f in group if f in selected_set)
                    reason = f"Lower AUC than '{best_feature}'. {group_members_str}"
                else:
                    reason = f"Selected by default (Group size 1). {group_members_str}"

            summary_list.append(
                {
                    "Variable": feature,
                    "AUC_Score": auc_score,
                    "Selection_Status": status,
                    "Reason": reason,
                }
            )

        summary_df = pd.DataFrame(summary_list)
        return summary_df[["Variable", "Selection_Status", "AUC_Score", "Reason"]]

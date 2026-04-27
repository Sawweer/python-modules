import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from utils.score import rescale_score
from typing import Type
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


class SMLogit(ClassifierMixin, BaseEstimator):
    """A universal sklearn-style wrapper for statsmodels classifier
    Parameters:
    --------------
    model_class : default = sm.Logit
        statsmodels' binary classifier (Logit, Probit, etc.)
    fit_intercept: bool, default = True
        should intercept be fitted
    Attributes:
    --------------
    classes_ : np.array
        Array of unique classes, as required by sklearn's API
    model_ :
        Underlying statsmodels' classifier
    """

    def __init__(self, model_class=sm.Logit, fit_intercept: bool = True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.classes_ = None
        self.X_std_ = None

    def fit(self, X, y):
        """Fit the model."""
        self.classes_ = np.unique(y)
        self.feature_names_in_ = (
            list(X.columns)
            if isinstance(X, pd.DataFrame)
            else [f"x{i + 1}" for i in range(X.shape[1])]
        )

        # Store standard deviations of features for standardized coefficients
        self.X_std_ = np.std(X, axis=0)

        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")
        self.model_ = self.model_class(y, X).fit(disp=False, maxiter=100)
        return self

    def predict(self, X):
        """Predict class labels (0 or 1) for given data."""
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")
        probs = self.model_.predict(X)
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        """Predict probabilities for each class."""
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")
        probs = self.model_.predict(X)
        return np.column_stack([1 - probs, probs])

    def score(self, X, y):
        """Return the metric scoring from X and y"""
        y_pred = self.predict(X)
        return (y_pred == y).mean()

    def score_samples(self, X):
        """Return the score of the data sample, rescaled from predict_proba."""
        y_pred = self.predict_proba(X)[:, 1]
        return pd.Series(rescale_score(y_pred), name="score")

    def summary(self):
        """Get a summary of the fitted statsmodels model."""
        return self.model_.summary()

    def get_feature_names_in(self):
        return self.feature_names_in_

    def get_standardized_coef(self):
        """Calculate and return standardized coefficients.

        Standardized coefficients = unstandardized coef * (std X * (sqrt(3) / pi))

        Returns:
        --------
        pd.Series
            Standardized coefficients indexed by feature names
        """
        coefs = self.model_.params

        # Calculate the scaling factor for logistic regression
        scale_factor = np.sqrt(3) / np.pi

        # Get feature names (excluding intercept if present)
        if self.fit_intercept:
            feature_coefs = coefs[1:]  # Skip intercept
            feature_names = self.feature_names_in_
        else:
            feature_coefs = coefs
            feature_names = self.feature_names_in_

        # Calculate standardized coefficients
        std_coefs = feature_coefs * (self.X_std_ * scale_factor)

        # Create a Series with feature names
        std_coef_series = pd.Series(
            std_coefs, index=feature_names, name="standardized_coef"
        )

        # Add intercept if it exists (intercept is not standardized)
        if self.fit_intercept:
            intercept_series = pd.Series(
                [coefs[0]], index=["const"], name="standardized_coef"
            )
            std_coef_series = pd.concat([intercept_series, std_coef_series])

        return std_coef_series

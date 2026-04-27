import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from keras.layers import BatchNormalization, Dropout
from utils.score import rescale_score


f05 = keras.metrics.FBetaScore(
    average=None, beta=0.5, threshold=0.5, name="f05", dtype=None
)

f15 = keras.metrics.FBetaScore(
    average=None, beta=1.5, threshold=0.5, name="f15", dtype=None
)

f1 = keras.metrics.F1Score(threshold=0.5, name="f1")


def build_fn(
    n_layers,
    n_neurons,
    learning_rate,
    loss,
    l1,
    l2,
    beta_1,
    beta_2,
    activation,
    kernel_initializer,
    batch_norm,
):
    model = Sequential()
    for layer in range(n_layers - 1):
        model.add(
            Dense(
                n_neurons,
                activation=activation,
                kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2),
                kernel_initializer=kernel_initializer,
            )
        )
        if batch_norm:
            model.add(BatchNormalization())
    model.add(Dense(1, activation="sigmoid"))
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
    )
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["auc", f1, f05, f15],
    )
    return model


class MLPBinaryClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_layers,
        n_neurons,
        learning_rate,
        loss="binary_crossentropy",
        batch_norm=True,
        activation="selu",
        kernel_initializer="lecun_normal",
        l1=0,
        l2=0,
        beta_1=0.9,
        beta_2=0.999,
        class_weight=None,
        validation_split=0.2,
        start_from_epoch=1,
        epochs=100,
        batch_size=32,
        verbose=0,
        patience=10,
        min_delta=0,
        monitor="val_f1",
    ):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.class_weight = class_weight
        self.validation_split = validation_split
        self.start_from_epoch = start_from_epoch
        self.patience = patience
        self.min_delta = min_delta
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.loss = loss
        self.classes_ = None  # Required by scikit-learn
        self.l1 = l1
        self.l2 = l2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.batch_norm = batch_norm
        self.monitor = monitor

    def fit(self, X, y):
        # Store unique classes (required by scikit-learn)
        self.classes_ = np.unique(y)

        # Build and compile the model
        self.model = self.build_fn(
            self.n_layers,
            self.n_neurons,
            self.learning_rate,
            self.loss,
            self.l1,
            self.l2,
            self.beta_1,
            self.beta_2,
            self.activation,
            self.kernel_initializer,
            self.batch_norm,
        )

        # Set up early stopping
        early_stopping_cb = keras.callbacks.EarlyStopping(
            patience=self.patience,
            restore_best_weights=True,
            monitor=self.monitor if self.validation_split > 0 else "loss",
            min_delta=self.min_delta,
            start_from_epoch=self.start_from_epoch,
            mode="max" if self.validation_split > 0 else "min",
        )

        # Fit the model
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            class_weight=self.class_weight,
            validation_split=self.validation_split,
            callbacks=[early_stopping_cb],
        )
        return self

    def predict(self, X):
        """Predict class labels (0 or 1) based on threshold 0.5 for binary classification"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        probs = self.model.predict(X, verbose=0)
        return (probs > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """Return class probabilities for both classes (0 and 1)."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        probs_class_1 = self.model.predict(X, verbose=0).flatten()
        probs_class_0 = 1 - probs_class_1

        # Return as numpy array with shape (n_samples, 2)
        return np.column_stack([probs_class_0, probs_class_1])

    def score(self, X, y):
        """Calculate and return ROC AUC score"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        # Get probabilities for class 1
        probs = self.model.predict(X, verbose=0).flatten()

        # Calculate and return ROC AUC score
        try:
            return roc_auc_score(y, probs)
        except ValueError as e:
            # Handle cases where AUC cannot be calculated (e.g., only one class present)
            print(f"Warning: Could not calculate ROC AUC - {e}")
            return np.nan

    def score_samples(self, X):
        y_pred = self.predict_proba(X)[:, 1]
        return pd.Series(rescale_score(y_pred), name="SCORE")

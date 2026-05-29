import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from keras.layers import BatchNormalization, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def build_fn(
    n_classes,
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
    f1_average,
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
    model.add(Dense(n_classes, activation="softmax"))
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
    )

    # Create F1 metric
    f1 = keras.metrics.F1Score(average=f1_average, name="f1", dtype=None)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["accuracy", f1],
    )
    return model


class MLPMulticlassClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_layers,
        n_neurons,
        learning_rate,
        loss="categorical_crossentropy",  # Changed to categorical
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
        f1_average="macro",
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
        self.f1_average = f1_average
        self.label_encoder_ = LabelEncoder()

    def fit(self, X, y):
        # Encode labels to ensure they are 0, 1, 2, ..., n_classes-1
        y_encoded = self.label_encoder_.fit_transform(y)

        # Store unique classes (required by scikit-learn)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)

        # Convert to one-hot encoding for categorical crossentropy
        y_categorical = to_categorical(y_encoded, num_classes=n_classes)

        # Build and compile the model
        self.model = self.build_fn(
            n_classes,
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
            self.f1_average,
        )

        # Determine monitor metric and mode
        monitor_metric = self.monitor if self.validation_split > 0 else "loss"
        mode = (
            "max" if "accuracy" in monitor_metric or "f1" in monitor_metric else "min"
        )

        # Set up early stopping
        early_stopping_cb = keras.callbacks.EarlyStopping(
            patience=self.patience,
            restore_best_weights=True,
            monitor=monitor_metric,
            min_delta=self.min_delta,
            start_from_epoch=self.start_from_epoch,
            mode=mode,
        )

        # Fit the model
        self.model.fit(
            X,
            y_categorical,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            class_weight=self.class_weight,
            validation_split=self.validation_split,
            callbacks=[early_stopping_cb],
        )
        return self

    def predict(self, X):
        """Predict class labels based on highest probability"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        probs = self.model.predict(X, verbose=0)
        y_pred_encoded = np.argmax(probs, axis=1)

        # Decode back to original labels
        return self.label_encoder_.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        """Return class probabilities for all classes."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        probs = self.model.predict(X, verbose=0)
        return probs

    def score(self, X, y):
        """Calculate and return accuracy score"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

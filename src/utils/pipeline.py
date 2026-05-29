import pandas as pd


def transform_x(X, pipe, passthrough_cols=None, until_step=None):
    if passthrough_cols is None:
        passthrough_cols = []

    # Resolve until_step to an index
    if until_step is None:
        steps = pipe.steps[:-1]
    elif isinstance(until_step, int):
        steps = pipe.steps[:until_step + 1]
    elif isinstance(until_step, str):
        step_names = [name for name, _ in pipe.steps]
        if until_step not in step_names:
            raise ValueError(
                f"Step '{until_step}' not found. Available steps: {step_names}")
        idx = step_names.index(until_step)
        steps = pipe.steps[:idx + 1]
    else:
        raise TypeError(
            f"until_step must be an int or str, got {type(until_step).__name__}")

    # Columns to transform
    transform_cols = [c for c in X.columns if c not in passthrough_cols]
    X_transformed = X[transform_cols]
    X_passthrough = X[passthrough_cols]

    # Apply resolved steps
    for name, step in steps:
        X_transformed = step.transform(X_transformed)

    # Convert to DataFrame if output is numpy
    if not isinstance(X_transformed, pd.DataFrame):
        X_transformed = pd.DataFrame(X_transformed, index=X.index)

    # Combine back
    X_final = pd.concat([X_passthrough, X_transformed], axis=1)
    return X_final

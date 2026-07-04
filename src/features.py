import numpy as np
import pandas as pd


def create_sliding_windows(
    X,
    y=None,
    feature_cols=None,
    target_col=None,
    window_size=30,
    window_step=1,
    engine_col="engine",
    cycle_col="cycle"
):
    """
    Transform engine time series into fixed-length sliding windows.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature dataframe containing engine, cycle and feature columns.
    y : pandas.DataFrame, optional
        Target dataframe containing engine, cycle and target columns.
    feature_cols : list of str
        Feature columns to include in each window.
    target_col : str, optional
        Target column assigned at the final cycle of each window.
    window_size : int
        Number of consecutive cycles per window.
    window_step : int
        Step size between consecutive windows.
    engine_col : str
        Engine identifier column.
    cycle_col : str
        Cycle column used for temporal ordering.

    Returns
    -------
    windows : numpy.ndarray
        Array with shape (samples, window_size, n_features).
    targets : numpy.ndarray, optional
        Target value for each window, returned only if y is provided.
    metadata : pandas.DataFrame
        Engine ID, start cycle and end cycle for each generated window.
    """

    if feature_cols is None:
        raise ValueError("feature_cols must be provided.")

    if y is not None and target_col is None:
        raise ValueError("target_col must be provided when y is not None.")

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    if window_step <= 0:
        raise ValueError("window_step must be positive.")

    windows = []
    targets = []
    metadata = []

    for engine_id, engine_X in X.groupby(engine_col):
        engine_X = engine_X.sort_values(cycle_col).reset_index(drop=True)

        if y is not None:
            engine_y = (
                y[y[engine_col] == engine_id]
                .sort_values(cycle_col)
                .reset_index(drop=True)
            )

            if not engine_X[cycle_col].equals(engine_y[cycle_col]):
                raise ValueError(
                    f"Feature and target cycles are not aligned for engine {engine_id}."
                )

        values = engine_X[feature_cols].to_numpy()

        for start_idx in range(0, len(engine_X) - window_size + 1, window_step):
            end_idx = start_idx + window_size

            windows.append(values[start_idx:end_idx])
            metadata.append({
                engine_col: engine_id,
                "start_cycle": engine_X.iloc[start_idx][cycle_col],
                "end_cycle": engine_X.iloc[end_idx - 1][cycle_col]
            })

            if y is not None:
                targets.append(engine_y.iloc[end_idx - 1][target_col])

    windows = np.array(windows)
    metadata = pd.DataFrame(metadata)

    if y is None:
        return windows, metadata

    targets = np.array(targets)

    return windows, targets, metadata


def extract_statistical_window_features(windows, feature_cols):
    """
    Extract tabular statistical features from sequence windows.

    For each original feature, the function computes mean, standard deviation,
    minimum, maximum, last value and linear slope within each window.
    """

    if windows.ndim != 3:
        raise ValueError("windows must have shape (samples, window_size, n_features).")

    if windows.shape[2] != len(feature_cols):
        raise ValueError(
            "The number of window features does not match len(feature_cols)."
        )

    feature_rows = []
    time_index = np.arange(windows.shape[1])

    for window in windows:
        row = {}

        for feature_idx, feature_name in enumerate(feature_cols):
            values = window[:, feature_idx]

            row[f"{feature_name}__mean"] = values.mean()
            row[f"{feature_name}__std"] = values.std()
            row[f"{feature_name}__min"] = values.min()
            row[f"{feature_name}__max"] = values.max()
            row[f"{feature_name}__last"] = values[-1]
            row[f"{feature_name}__slope"] = np.polyfit(
                time_index,
                values,
                deg=1
            )[0]

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)

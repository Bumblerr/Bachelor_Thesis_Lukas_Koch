import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def add_rul_to_training_data(df):
    """
    Add Remaining Useful Life target to training data.

    RUL = max_cycle_per_engine - current_cycle
    """

    df = df.copy()

    max_cycle_per_engine = (
        df.groupby("engine")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "max_cycle"})
    )

    df = df.merge(max_cycle_per_engine, on="engine", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    return df


def add_capped_rul(df, cap=125, source_col="RUL", target_col="RUL_capped"):
    """
    Add capped RUL target.
    """

    df = df.copy()
    df[target_col] = df[source_col].clip(upper=cap)

    return df


def prepare_test_rul(df_test, df_test_RUL):
    """
    Prepare test RUL summary by linking provided RUL values
    to the corresponding test engines.
    """

    df_test_RUL = df_test_RUL.copy()
    df_test_RUL["engine"] = np.arange(1, len(df_test_RUL) + 1)

    test_last_cycles = (
        df_test
        .groupby("engine")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "last_observed_cycle"})
    )

    df_test_rul_summary = test_last_cycles.merge(
        df_test_RUL,
        on="engine",
        how="left"
    )

    df_test_rul_summary["estimated_failure_cycle"] = (
        df_test_rul_summary["last_observed_cycle"]
        + df_test_rul_summary["RUL"]
    )

    return df_test_rul_summary


def identify_constant_features(df, feature_cols):
    """
    Identify feature columns with only one unique value.

    The function is intended to be fitted on training data only, so that
    preprocessing decisions do not use validation or test information.
    """

    return [
        feature
        for feature in feature_cols
        if df[feature].nunique() == 1
    ]


def summarize_constant_features(df, constant_features, setting_names=None):
    """
    Create a summary table for constant features.
    """

    setting_names = set(setting_names or [])

    return pd.DataFrame({
        "feature": constant_features,
        "type": [
            "setting" if feature in setting_names else "sensor"
            for feature in constant_features
        ],
        "constant_value": [
            df[feature].iloc[0]
            for feature in constant_features
        ]
    })


def remove_features(df, features_to_remove):
    """
    Return a dataframe without the selected features.
    """

    return df.drop(columns=features_to_remove)


def define_feature_columns(df, index_cols, target_cols=None, helper_cols=None):
    """
    Define model input columns by excluding index, target and helper columns.
    """

    target_cols = target_cols or []
    helper_cols = helper_cols or []
    excluded_cols = set(index_cols + target_cols + helper_cols)

    return [
        col
        for col in df.columns
        if col not in excluded_cols
    ]


def create_feature_target_frames(
    df_train,
    df_test,
    feature_cols,
    target_col,
    index_cols=None
):
    """
    Create full feature and target frames before train-validation splitting.
    """

    index_cols = index_cols or ["engine", "cycle"]

    X_train_full = df_train[index_cols + feature_cols].copy()
    y_train_full = df_train[index_cols + [target_col]].copy()
    X_test_full = df_test[index_cols + feature_cols].copy()

    return X_train_full, y_train_full, X_test_full


def split_train_validation_by_engine(
    df,
    validation_size=0.2,
    random_state=42,
    engine_col="engine"
):
    """
    Split a dataframe into training and validation sets by engine unit.
    """

    unique_engines = df[engine_col].unique()

    train_engines, val_engines = train_test_split(
        unique_engines,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True
    )

    train_mask = df[engine_col].isin(train_engines)
    val_mask = df[engine_col].isin(val_engines)

    df_train_split = df[train_mask].copy()
    df_val_split = df[val_mask].copy()

    return df_train_split, df_val_split, train_engines, val_engines


def create_split_feature_target_frames(
    df_train_split,
    df_val_split,
    feature_cols,
    target_col,
    index_cols=None
):
    """
    Create feature and target frames from train and validation splits.
    """

    index_cols = index_cols or ["engine", "cycle"]

    X_train = df_train_split[index_cols + feature_cols].copy()
    y_train = df_train_split[index_cols + [target_col]].copy()

    X_val = df_val_split[index_cols + feature_cols].copy()
    y_val = df_val_split[index_cols + [target_col]].copy()

    return X_train, y_train, X_val, y_val


def scale_feature_sets(
    X_train,
    X_val,
    X_test,
    feature_cols,
    scaler=None
):
    """
    Fit a scaler on training features and transform train, validation and test.
    """

    scaler = scaler or StandardScaler()
    scaler.fit(X_train[feature_cols])

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[feature_cols] = scaler.transform(X_train[feature_cols])
    X_val_scaled[feature_cols] = scaler.transform(X_val[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def summarize_scaler(scaler, feature_cols):
    """
    Create a summary table of fitted scaler parameters.
    """

    return pd.DataFrame({
        "feature": feature_cols,
        "mean": scaler.mean_,
        "scale": scaler.scale_
    })

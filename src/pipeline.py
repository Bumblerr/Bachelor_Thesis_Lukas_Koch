import pandas as pd

from src.data_loading import (
    load_cmapss_dataset,
    SETTING_NAMES,
    SENSOR_NAMES,
)
from src.features import (
    create_sliding_windows,
    extract_statistical_window_features,
)
from src.preprocessing import (
    add_capped_rul,
    add_rul_to_training_data,
    create_feature_target_frames,
    create_split_feature_target_frames,
    define_feature_columns,
    identify_constant_features,
    prepare_test_rul,
    remove_features,
    scale_feature_sets,
    split_train_validation_by_engine,
)


def recreate_prediction_pipeline(config):
    """
    Recreate the tabular RUL prediction pipeline from raw C-MAPSS data.

    The pipeline loads one dataset subset, creates capped RUL targets, removes
    constant features, performs an engine-wise train-validation split, scales
    features, creates sliding windows and extracts statistical window features.
    """

    dataset_id = config["dataset"]["dataset_id"]
    preprocessing_config = config["preprocessing"]
    feature_config = config["features"]

    rul_cap = preprocessing_config["rul_cap"]
    validation_size = preprocessing_config["validation_size"]
    random_state = preprocessing_config["random_state"]
    index_cols = preprocessing_config["index_cols"]
    target_cols = preprocessing_config["target_cols"]
    helper_cols = preprocessing_config["helper_cols"]
    target_col = preprocessing_config["target_col"]
    window_size = feature_config["window_size"]
    window_step = feature_config["window_step"]

    df_train_raw, df_test_raw, df_test_rul_raw = load_cmapss_dataset(dataset_id)

    df_train_rul = add_rul_to_training_data(df_train_raw)
    df_train_rul = add_capped_rul(
        df_train_rul,
        cap=rul_cap,
        source_col="RUL",
        target_col=target_col,
    )

    df_test_rul_summary = prepare_test_rul(
        df_test_raw,
        df_test_rul_raw,
    )

    candidate_feature_cols = SETTING_NAMES + SENSOR_NAMES
    constant_features = identify_constant_features(
        df_train_raw,
        candidate_feature_cols,
    )

    df_train_preprocessed = remove_features(
        df_train_rul,
        constant_features,
    )
    df_test_preprocessed = remove_features(
        df_test_raw,
        constant_features,
    )

    feature_cols = define_feature_columns(
        df_train_preprocessed,
        index_cols=index_cols,
        target_cols=target_cols,
        helper_cols=helper_cols,
    )

    X_train_full, y_train_full, X_test_full = create_feature_target_frames(
        df_train_preprocessed,
        df_test_preprocessed,
        feature_cols=feature_cols,
        target_col=target_col,
        index_cols=index_cols,
    )

    df_train_split, df_val_split, train_engines, val_engines = (
        split_train_validation_by_engine(
            df_train_preprocessed,
            validation_size=validation_size,
            random_state=random_state,
            engine_col="engine",
        )
    )

    X_train, y_train, X_val, y_val = create_split_feature_target_frames(
        df_train_split,
        df_val_split,
        feature_cols=feature_cols,
        target_col=target_col,
        index_cols=index_cols,
    )

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_feature_sets(
        X_train,
        X_val,
        X_test_full,
        feature_cols=feature_cols,
    )

    X_train_windows, y_train_windows, train_window_metadata = create_sliding_windows(
        X_train_scaled,
        y_train,
        feature_cols=feature_cols,
        target_col=target_col,
        window_size=window_size,
        window_step=window_step,
    )

    X_val_windows, y_val_windows, val_window_metadata = create_sliding_windows(
        X_val_scaled,
        y_val,
        feature_cols=feature_cols,
        target_col=target_col,
        window_size=window_size,
        window_step=window_step,
    )

    X_test_windows, test_window_metadata = create_sliding_windows(
        X_test_scaled,
        y=None,
        feature_cols=feature_cols,
        window_size=window_size,
        window_step=window_step,
    )

    X_train_tabular = extract_statistical_window_features(
        X_train_windows,
        feature_cols,
    )
    X_val_tabular = extract_statistical_window_features(
        X_val_windows,
        feature_cols,
    )
    X_test_tabular = extract_statistical_window_features(
        X_test_windows,
        feature_cols,
    )

    y_train_tabular = pd.Series(
        y_train_windows,
        name=target_col,
    )
    y_val_tabular = pd.Series(
        y_val_windows,
        name=target_col,
    )

    return {
        "df_train_raw": df_train_raw,
        "df_test_raw": df_test_raw,
        "df_test_rul_raw": df_test_rul_raw,
        "df_train_rul": df_train_rul,
        "df_test_rul_summary": df_test_rul_summary,
        "candidate_feature_cols": candidate_feature_cols,
        "constant_features": constant_features,
        "df_train_preprocessed": df_train_preprocessed,
        "df_test_preprocessed": df_test_preprocessed,
        "feature_cols": feature_cols,
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "X_test_full": X_test_full,
        "df_train_split": df_train_split,
        "df_val_split": df_val_split,
        "train_engines": train_engines,
        "val_engines": val_engines,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "X_train_windows": X_train_windows,
        "y_train_windows": y_train_windows,
        "train_window_metadata": train_window_metadata,
        "X_val_windows": X_val_windows,
        "y_val_windows": y_val_windows,
        "val_window_metadata": val_window_metadata,
        "X_test_windows": X_test_windows,
        "test_window_metadata": test_window_metadata,
        "X_train_tabular": X_train_tabular,
        "X_val_tabular": X_val_tabular,
        "X_test_tabular": X_test_tabular,
        "y_train_tabular": y_train_tabular,
        "y_val_tabular": y_val_tabular,
        "tabular_feature_cols": X_train_tabular.columns.tolist(),
    }

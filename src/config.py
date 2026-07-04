from copy import deepcopy
from pathlib import Path
import json


DEFAULT_CONFIG = {
    "dataset": {
        "dataset_id": "FD001",
    },
    "preprocessing": {
        "rul_cap": 125,
        "validation_size": 0.2,
        "random_state": 42,
        "target_col": "RUL_capped",
        "index_cols": ["engine", "cycle"],
        "target_cols": ["RUL", "RUL_capped"],
        "helper_cols": ["max_cycle"],
    },
    "features": {
        "window_size": 30,
        "window_step": 1,
    },
    "models": {
        "xgboost": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "n_jobs": -1,
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 2,
            "n_jobs": -1,
        },
        "gru_sequence": {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "batch_size": 256,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "max_epochs": 20,
            "patience": 5,
            "huber_delta": 10.0,
            "prediction_batch_size": 1024,
        },
    },
    "conformal": {
        "coverage_level": 0.9,
    },
    "simulation": {
        "fleet_validation_size": 0.3,
        "fleet_simulation_size": 0.67,
        "fixed_maintenance_interval": 150,
        "predictive_rul_threshold": 20,
        "oracle_rul_threshold": 20,
        "maintenance_delay_cycles": 5,
        "planned_maintenance_cost": 1.0,
        "failure_cost": 10.0,
        "early_maintenance_cost_per_unused_cycle": 0.02,
    },
}


def deep_update(base, updates):
    """
    Recursively update a nested dictionary and return a new dictionary.
    """

    result = deepcopy(base)

    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def get_default_config():
    """
    Return a copy of the default project configuration.
    """

    return deepcopy(DEFAULT_CONFIG)


def load_config(config_path=None, overrides=None):
    """
    Load project configuration from defaults, an optional file and overrides.

    JSON files are supported without additional dependencies. YAML files are
    supported if PyYAML is installed in the active environment.
    """

    config = get_default_config()

    if config_path is not None:
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path.suffix.lower() == ".json":
            with config_path.open("r", encoding="utf-8") as file:
                file_config = json.load(file)
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to load YAML config files. "
                    "Install it or use a JSON config file."
                ) from exc

            with config_path.open("r", encoding="utf-8") as file:
                file_config = yaml.safe_load(file) or {}
        else:
            raise ValueError(
                "Unsupported config file type. Use .json, .yaml or .yml."
            )

        config = deep_update(config, file_config)

    if overrides:
        config = deep_update(config, overrides)

    return config


def flatten_config(config):
    """
    Flatten the nested project config into commonly used notebook variables.
    """

    preprocessing = config["preprocessing"]
    features = config["features"]

    return {
        "CURRENT_DATASET": config["dataset"]["dataset_id"],
        "RUL_CAP": preprocessing["rul_cap"],
        "VALIDATION_SIZE": preprocessing["validation_size"],
        "RANDOM_STATE": preprocessing["random_state"],
        "TARGET_COL": preprocessing["target_col"],
        "INDEX_COLS": preprocessing["index_cols"],
        "TARGET_COLS": preprocessing["target_cols"],
        "HELPER_COLS": preprocessing["helper_cols"],
        "WINDOW_SIZE": features["window_size"],
        "WINDOW_STEP": features["window_step"],
    }

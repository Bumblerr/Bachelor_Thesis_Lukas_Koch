from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"

DATASET_IDS = ["FD001", "FD002", "FD003", "FD004"]

# Index columns
INDEX_NAMES = ["engine", "cycle"]

# Operational settings
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]

# Sensor measurements
SENSOR_NAMES = [
    "T2 (Fan inlet temperature)",
    "T24 (LPC outlet temperature)",
    "T30 (HPC outlet temperature)",
    "T50 (LPT outlet temperature)",
    "P2 (Fan inlet pressure)",
    "P15 (bypass duct pressure)",
    "P30 (HPC outlet pressure)",
    "Nf (Physical fan speed)",
    "Nc (Physical core speed)",
    "epr (Engine pressure ratio (P50/P2))",
    "Ps30 (Static pressure at HPC outlet)",
    "phi (Ratio of fuel flow to Ps30)",
    "NRf (Corrected fan speed)",
    "NRc (Corrected core speed)",
    "BPR (Bypass ratio)",
    "farB (Fuel-air ratio)",
    "htBleed (Bleed enthalpy)",
    "Nf_dmd (Demanded fan speed)",
    "PCNfr_dmd (Demanded corrected fan speed)",
    "W31 (HPT Cool air flow)",
    "W32 (LPT Cool air flow)"
]

COLUMN_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES


def load_cmapss_dataset(dataset_id, data_dir=DATA_DIR):
    """
    Load train, test and RUL files for one NASA C-MAPSS dataset subset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier: "FD001", "FD002", "FD003" or "FD004".
    data_dir : pathlib.Path
        Directory containing the C-MAPSS text files.

    Returns
    -------
    df_train : pandas.DataFrame
        Training data containing complete run-to-failure trajectories.
    df_test : pandas.DataFrame
        Test data containing partial trajectories.
    df_test_rul : pandas.DataFrame
        True RUL values for the final cycle of each test engine.
    """

    dataset_id = dataset_id.upper()

    if dataset_id not in DATASET_IDS:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. Choose one of {DATASET_IDS}."
        )

    train_path = data_dir / f"train_{dataset_id}.txt"
    test_path = data_dir / f"test_{dataset_id}.txt"
    rul_path = data_dir / f"RUL_{dataset_id}.txt"

    for path in [train_path, test_path, rul_path]:
        if not path.exists():
            raise FileNotFoundError(f"File '{path}' not found.")

    df_train = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    df_test = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    df_test_rul = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["RUL"]
    )

    return df_train, df_test, df_test_rul


def load_all_cmapss_datasets(data_dir=DATA_DIR):
    """
    Load all NASA C-MAPSS dataset subsets into a dictionary.

    Returns
    -------
    datasets : dict
        Dictionary with dataset IDs as keys and train, test and test_RUL dataframes as values.
    """

    datasets = {}

    for dataset_id in DATASET_IDS:
        df_train, df_test, df_test_rul = load_cmapss_dataset(
            dataset_id,
            data_dir=data_dir
        )

        datasets[dataset_id] = {
            "train": df_train,
            "test": df_test,
            "test_RUL": df_test_rul
        }

    return datasets

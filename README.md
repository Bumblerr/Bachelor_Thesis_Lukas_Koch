# Predictive Maintenance with NASA C-MAPSS

This project builds a predictive maintenance workflow for Remaining Useful Life
(RUL) prediction using the NASA C-MAPSS turbofan engine degradation dataset.

The project combines exploratory analysis, preprocessing, sliding-window feature
engineering, classical machine learning, SHAP explainability, maintenance
scheduling simulation and conformal prediction for uncertainty-aware RUL
intervals.

## Project Goal

Industrial machines often show gradual degradation long before failure. These
patterns are hidden in noisy multivariate sensor streams and are difficult to
detect manually.

This project uses C-MAPSS to:

- analyze run-to-failure engine trajectories
- construct raw and capped RUL targets
- create sliding-window features from sensor time series
- train and compare RUL regression models
- explain model predictions with SHAP
- connect predictions to simplified maintenance decisions
- estimate uncertainty with conformal prediction intervals

## Dataset

The project uses the NASA C-MAPSS turbofan engine degradation dataset.

Expected raw data location:

```text
data/raw/CMAPSSData/
```

The dataset contains four subsets:

| Subset | Operating conditions | Fault modes |
| --- | ---: | ---: |
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

The active subset is selected centrally in:

```text
src/config.py
```

Change the value below to switch the dataset used by the configured notebooks
and pipeline:

```python
"dataset": {
    "dataset_id": "FD001",
}
```

FD001 is the default starting point because it contains one operating condition
and one fault mode. FD002 and FD004 require more careful treatment of operating
conditions because they contain multiple operating regimes.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       └── CMAPSSData/
├── notebooks/
│   ├── 01_data_import_and_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_rul_target_and_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training_and_evaluation.ipynb
│   ├── 06_explainability_shap.ipynb
│   ├── 07_maintenance_scheduling_simulation.ipynb
│   └── 08_conformal_prediction_uncertainty.ipynb
└── src/
    ├── config.py
    ├── data_loading.py
    ├── features.py
    ├── models.py
    ├── pipeline.py
    └── preprocessing.py
```

## Current Architecture

The project follows a hybrid structure:

- notebooks contain the analysis narrative, plots, interpretation and model
  comparison
- `src/` contains reusable project logic
- `src/config.py` contains shared configuration
- `src/pipeline.py` recreates the repeated prediction pipeline for later
  notebooks

This keeps the notebooks readable while avoiding repeated preprocessing and
feature-engineering code across model training, explainability, simulation and
uncertainty estimation.

## Notebook Workflow

The notebooks are intended to be run in numerical order.

### 01 Data Import and Understanding

Loads the raw C-MAPSS files, assigns column names and verifies the basic dataset
structure.

### 02 Exploratory Data Analysis

Explores engine lifetimes, operational settings, sensor variability, constant
features, RUL behavior and degradation-related sensor trajectories.

### 03 RUL Target and Preprocessing

Creates raw and capped RUL targets, prepares test RUL information, removes
constant features, performs an engine-wise train-validation split and prepares
scaled feature sets.

### 04 Feature Engineering

Transforms multivariate sensor trajectories into sliding windows and extracts
statistical window features for classical machine learning models.

### 05 Model Training and Evaluation

Trains and compares an age baseline, Random Forest, XGBoost and a GRU sequence
model. Evaluation uses MAE, RMSE and the asymmetric C-MAPSS score.

### 06 Explainability with SHAP

Uses SHAP to explain the XGBoost model globally and locally. Engineered window
features are interpreted at feature, sensor and statistic level.

### 07 Maintenance Scheduling Simulation

Translates RUL predictions into a rolling fleet-level maintenance simulation.
Compares fixed, point-estimate, soft uncertainty-aware, hard lower-bound and
oracle-style scheduling policies under a transparent cost model.

### 08 Conformal Prediction Uncertainty

Extends point predictions with conformal prediction intervals and compares
point-estimate maintenance decisions with interval-based uncertainty-aware
decisions.

## Source Modules

### `src/config.py`

Central project configuration.

Contains:

- selected dataset subset
- RUL cap
- train-validation split settings
- target, index and helper columns
- sliding-window settings
- model hyperparameters
- conformal prediction coverage level
- maintenance simulation parameters

The notebooks use:

```python
config = load_config()
flat_config = flatten_config(config)
```

### `src/data_loading.py`

Reusable C-MAPSS loading utilities and column definitions.

Contains:

- dataset IDs
- index, setting and sensor column names
- single-subset loading
- loading all C-MAPSS subsets

### `src/preprocessing.py`

Reusable preprocessing functions.

Contains:

- raw RUL construction
- capped RUL construction
- test RUL preparation
- constant feature identification and removal
- feature and target frame creation
- engine-wise train-validation splitting
- feature scaling

### `src/features.py`

Reusable feature engineering functions.

Contains:

- sliding-window generation
- statistical window feature extraction

### `src/models.py`

Reusable model construction helpers.

Contains:

- Random Forest model creation
- XGBoost model creation
- GRU sequence model construction, training and prediction helpers

The model helpers read hyperparameters from `src/config.py`, while the notebooks
still orchestrate fitting and evaluation explicitly.

### `src/pipeline.py`

Centralized prediction pipeline used by notebooks 05 to 08.

The main function is:

```python
recreate_prediction_pipeline(config)
```

It returns a dictionary containing the relevant intermediate artifacts, including:

- raw and preprocessed train/test data
- train and validation splits
- scaled feature sets
- sliding windows
- tabular window features
- window metadata
- target vectors
- selected feature columns
- fitted scaler

In the notebooks, this is currently unpacked with:

```python
pipeline_outputs = recreate_prediction_pipeline(config)
globals().update(pipeline_outputs)
```

This keeps the following notebook cells concise and allows existing variables
such as `X_train_tabular`, `y_train_tabular`, `X_test_tabular`,
`test_window_metadata` and `df_test_rul_summary` to be used directly.

## Methodological Notes

### RUL Capping

The project uses both raw RUL and capped RUL. The capped target reflects the
common assumption that early-life degradation is weak or not directly observable.

The primary modeling target is:

```text
RUL_capped
```

with a default upper cap of 125 cycles.

### Data Leakage Prevention

Train-validation splitting is performed by engine unit, not by individual rows.
This prevents cycles from the same engine trajectory from appearing in both
training and validation data.

Feature scaling is fitted only on the training split and then applied to
validation and test data.

### Sliding-Window Features

Sensor trajectories are transformed into fixed-length windows. For classical
machine learning models, each window is converted into tabular statistical
features such as mean, standard deviation, minimum, maximum, last value and
slope.

### Maintenance Simulation

The maintenance simulation is intentionally simplified. It illustrates how RUL
predictions can be connected to decision-making and cost trade-offs. It should
not be interpreted as a complete industrial maintenance optimization system.

### Conformal Prediction

Conformal prediction is used to construct uncertainty intervals around capped
RUL predictions. The intervals are calibrated using validation residuals and
evaluated against capped test RUL values.

## Installation

Create and activate your preferred Python environment, then install the required
packages from the project root:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `shap`
- `notebook` / `jupyterlab`

## Running the Project

Open the project root as the working directory:

```bash
cd Bachelor_Thesis_Lukas_Koch
or
Open the project root directory
```

Start Jupyter:

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

Then run the notebooks in numerical order from `01` to `08`.

If you change `src/config.py`, restart the notebook kernel and rerun the
notebook from the top so that the updated configuration is loaded.

The committed notebooks include saved outputs that document the results used for
the project. Re-running the notebooks may lead to small numerical or visual
differences because some models are retrained and execution environments can
vary slightly.

## Current Status

The project currently contains a complete notebook-based end-to-end workflow:

```text
raw data
→ EDA
→ preprocessing
→ feature engineering
→ model training
→ explainability
→ maintenance simulation
→ uncertainty estimation
```

The most repeated pipeline logic has been moved into `src/pipeline.py`. The
notebooks remain the primary execution and documentation layer.
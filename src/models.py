import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _load_torch_components():
    """
    Import PyTorch lazily so tree-based models remain usable without it.
    """

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the GRU sequence model. "
            "Install torch in the active environment before running it."
        ) from exc

    return torch, nn, DataLoader, TensorDataset


def build_random_forest_model(model_config, random_state=42):
    """
    Build a Random Forest regressor from the project model config.
    """

    return RandomForestRegressor(
        **model_config["random_forest"],
        random_state=random_state,
    )


def build_xgboost_model(model_config, random_state=42):
    """
    Build an XGBoost regressor from the project model config.
    """

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required to build the XGBoost model. "
            "Install it in the active notebook environment."
        ) from exc

    return XGBRegressor(
        **model_config["xgboost"],
        random_state=random_state,
    )


def get_torch_device():
    """
    Select the best available PyTorch device for sequence model training.
    """

    torch, _, _, _ = _load_torch_components()

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def create_sequence_dataloaders(
    X_train_windows,
    y_train_windows,
    X_val_windows,
    y_val_windows,
    X_test_windows,
    batch_size,
    random_state=None,
):
    """
    Convert sliding-window arrays into PyTorch tensors and data loaders.
    """

    torch, _, DataLoader, TensorDataset = _load_torch_components()

    X_train_sequence_tensor = torch.from_numpy(
        np.ascontiguousarray(X_train_windows, dtype=np.float32)
    )
    y_train_sequence_tensor = torch.from_numpy(
        np.ascontiguousarray(y_train_windows, dtype=np.float32)
    ).view(-1, 1)

    X_val_sequence_tensor = torch.from_numpy(
        np.ascontiguousarray(X_val_windows, dtype=np.float32)
    )
    y_val_sequence_tensor = torch.from_numpy(
        np.ascontiguousarray(y_val_windows, dtype=np.float32)
    ).view(-1, 1)

    X_test_sequence_tensor = torch.from_numpy(
        np.ascontiguousarray(X_test_windows, dtype=np.float32)
    )

    generator = None
    if random_state is not None:
        generator = torch.Generator()
        generator.manual_seed(random_state)

    train_sequence_loader = DataLoader(
        TensorDataset(X_train_sequence_tensor, y_train_sequence_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_sequence_loader = DataLoader(
        TensorDataset(X_val_sequence_tensor, y_val_sequence_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "X_train_sequence_tensor": X_train_sequence_tensor,
        "y_train_sequence_tensor": y_train_sequence_tensor,
        "X_val_sequence_tensor": X_val_sequence_tensor,
        "y_val_sequence_tensor": y_val_sequence_tensor,
        "X_test_sequence_tensor": X_test_sequence_tensor,
        "train_sequence_loader": train_sequence_loader,
        "val_sequence_loader": val_sequence_loader,
    }


def build_gru_sequence_model(
    n_features,
    model_config,
    device=None,
    random_state=None,
):
    """
    Build a GRU regressor for raw sliding-window sensor sequences.
    """

    torch, nn, _, _ = _load_torch_components()

    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    class GRURegressor(nn.Module):
        def __init__(self, n_features, hidden_size, num_layers, dropout):
            super().__init__()
            self.gru = nn.GRU(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            _, hidden = self.gru(x)
            last_hidden = hidden[-1]
            return self.regressor(last_hidden)

    if device is None:
        device = get_torch_device()

    return GRURegressor(
        n_features=n_features,
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(device)


def _run_sequence_epoch(model, data_loader, loss_fn, device, optimizer=None):
    """
    Run one training or validation epoch for a sequence model.
    """

    torch, _, _, _ = _load_torch_components()
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_X.shape[0]
            total_samples += batch_X.shape[0]

    return total_loss / total_samples


def train_gru_sequence_model(
    model,
    train_loader,
    val_loader,
    model_config,
    device,
    random_state=42,
    verbose=True,
):
    """
    Train a GRU sequence model with early stopping on validation loss.
    """

    torch, nn, _, _ = _load_torch_components()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    loss_fn = nn.HuberLoss(delta=model_config["huber_delta"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
    )

    best_val_loss = np.inf
    best_model_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, model_config["max_epochs"] + 1):
        train_loss = _run_sequence_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
        )
        val_loss = _run_sequence_epoch(
            model,
            val_loader,
            loss_fn,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if verbose:
            print(
                f"Epoch {epoch:02d} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= model_config["patience"]:
            if verbose:
                print(f"Early stopping after {epoch} epochs.")
            break

    model.load_state_dict(best_model_state)

    return model, pd.DataFrame(history)


def predict_sequence_model(model, X_tensor, device, batch_size):
    """
    Predict RUL values for a tensor of raw sequence windows.
    """

    torch, _, _, _ = _load_torch_components()
    model.eval()
    predictions = []

    with torch.no_grad():
        for start_idx in range(0, X_tensor.shape[0], batch_size):
            batch_X = X_tensor[start_idx:start_idx + batch_size].to(device)
            batch_pred = model(batch_X).cpu().numpy().reshape(-1)
            predictions.append(batch_pred)

    return np.concatenate(predictions)

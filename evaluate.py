"""
evaluate.py — Model evaluation, result plotting, and gradient flow inspection.

Run:
    python evaluate.py              # full evaluation of the trained GRU+Skip model
    python evaluate.py --ablation   # also overlay ablation loss curves in one plot

What this script does:
    1. Load the best saved model from results/best_model.pth
    2. Compute MSE, MAE, RMSE on the test set
    3. Plot training + validation loss curves
    4. Plot predicted vs actual sequence on a test batch
    5. Print gradient norms after one forward+backward pass
    6. (Optional) plot ablation comparison loss curves
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import Config
from dataset import create_dataloaders
from model import GRUWithSkip


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, cfg: Config, use_skip: bool) -> GRUWithSkip:
    """Load a saved GRUWithSkip from a checkpoint file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUWithSkip(cfg.INPUT_SIZE, cfg.HIDDEN_SIZE, cfg.OUTPUT_SIZE, use_skip=use_skip)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def denormalise(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Reverse z-score normalisation to recover original signal scale."""
    return tensor * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    model: GRUWithSkip,
    test_loader: torch.utils.data.DataLoader,
    cfg: Config,
) -> dict:
    """
    Compute MSE, MAE, and RMSE on the full test set.

    All metrics are computed in normalised space (same space as training loss)
    for a fair comparison with the training curve values.

    Returns:
        dict with keys: mse, mae, rmse
    """
    device = next(model.parameters()).device
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    total_mse = 0.0
    total_mae = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            total_mse += criterion_mse(preds, y_batch).item()
            total_mae += criterion_mae(preds, y_batch).item()
            n_batches += 1

    mse  = total_mse / n_batches
    mae  = total_mae / n_batches
    rmse = mse ** 0.5

    print("\n── Test Set Metrics ─────────────────────────────────────")
    print(f"  MSE:  {mse:.5f}")
    print(f"  MAE:  {mae:.5f}")
    print(f"  RMSE: {rmse:.5f}")
    print("─────────────────────────────────────────────────────────\n")

    return {"mse": mse, "mae": mae, "rmse": rmse}


# ─────────────────────────────────────────────────────────────────────────────
# Gradient norm inspection
# ─────────────────────────────────────────────────────────────────────────────

def check_gradient_norms(model: GRUWithSkip, test_loader, cfg: Config) -> None:
    """
    Run one forward+backward pass and print gradient norms for all parameters.

    Purpose: empirically verify that gradients are healthy (non-vanishing) throughout
    the network — including the earliest GRU cell weights.

    A very small norm (< 1e-6) in early layers would indicate vanishing gradients.
    Healthy norms in all layers confirm the skip connection is doing its job.
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()

    model.train()   # need train mode to allow gradient accumulation
    X_batch, y_batch = next(iter(test_loader))
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    model.zero_grad()
    loss = criterion(model(X_batch), y_batch)
    loss.backward()

    print("── Gradient Norms After One Backward Pass ───────────────")
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm   = param.grad.norm().item()
            status = "OK        " if norm > 1e-6 else "⚠  VANISHING"
            print(f"  {name:<45}  norm = {norm:.6f}  [{status}]")
    print("─────────────────────────────────────────────────────────\n")

    model.eval()   # restore eval mode


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    save_path: str,
    title: str = "Training and Validation Loss",
) -> None:
    """
    Plot train vs validation MSE loss over epochs.
    LR reduction events appear as kinks in the curve — include this context in Loom.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, train_losses, label="Train MSE",      linewidth=2)
    ax.plot(epochs, val_losses,   label="Validation MSE", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ablation_curves(
    skip_train: list, skip_val: list,
    base_train: list, base_val: list,
    save_path: str,
) -> None:
    """
    Overlay validation loss curves for GRU+Skip vs GRU-only.
    The visual gap between the two lines is the empirical case for skip connections.
    """
    epochs = range(1, len(skip_val) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, skip_val, label="GRU + Skip (val)",  linewidth=2)
    ax.plot(epochs, base_val, label="GRU only (val)",    linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE Loss")
    ax.set_title("Ablation Study: Effect of Skip Connection on Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_predictions(
    model: GRUWithSkip,
    test_loader: torch.utils.data.DataLoader,
    norm_mean: float,
    norm_std: float,
    save_path: str,
    n_steps: int = 100,
) -> None:
    """
    Plot predicted vs actual values for the first n_steps of the test set.

    Predictions are denormalised back to the original signal scale before plotting,
    so the y-axis shows the actual wave amplitude rather than z-scores.
    """
    device = next(model.parameters()).device
    X_batch, y_batch = next(iter(test_loader))
    X_batch = X_batch.to(device)

    with torch.no_grad():
        preds = model(X_batch)

    # Take the first sample from the batch; flatten the time dimension
    actual    = denormalise(y_batch[0, :n_steps, 0], norm_mean, norm_std).numpy()
    predicted = denormalise(preds[0, :n_steps, 0].cpu(), norm_mean, norm_std).numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual,    label="Actual",    linewidth=2)
    ax.plot(predicted, label="Predicted", linewidth=2, linestyle="--", alpha=0.85)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Signal value")
    ax.set_title("GRU with Skip Connections — Predictions vs Actual (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(cfg: Config, include_ablation: bool = False) -> None:
    """
    Full evaluation pipeline for the trained GRU+Skip model.

    Expects:
        results/best_model.pth            — main model checkpoint
        results/best_model_history.pt     — loss history from training
        results/norm_stats.pt             — normalisation mean/std
        results/best_model_no_skip.pth    — (ablation only)
        results/best_model_no_skip_history.pt  — (ablation only)
    """
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # ── Load normalisation stats ────────────────────────────────────────
    if not os.path.exists(cfg.NORM_STATS_PATH):
        raise FileNotFoundError(
            f"Normalisation stats not found at {cfg.NORM_STATS_PATH}. "
            "Run  python train.py  first."
        )
    norm = torch.load(cfg.NORM_STATS_PATH)
    norm_mean, norm_std = norm["mean"], norm["std"]

    # ── Load model and data ─────────────────────────────────────────────
    model = load_model(cfg.CHECKPOINT_PATH, cfg, use_skip=True)
    _, _, test_loader, _, _ = create_dataloaders(cfg)

    print("\n── Evaluation: GRU + Skip Connection ────────────────────")

    # 1. Metrics
    metrics = compute_metrics(model, test_loader, cfg)

    # 2. Gradient norms
    check_gradient_norms(model, test_loader, cfg)

    # 3. Loss curves
    history_path = cfg.CHECKPOINT_PATH.replace(".pth", "_history.pt")
    if os.path.exists(history_path):
        history = torch.load(history_path)
        print("── Saving plots ──────────────────────────────────────────")
        plot_loss_curves(
            history["train"],
            history["val"],
            save_path=os.path.join(cfg.RESULTS_DIR, "loss_curve.png"),
        )
    else:
        print(f"  Warning: loss history not found at {history_path} — skipping loss plot")

    # 4. Predictions vs actual
    plot_predictions(
        model, test_loader, norm_mean, norm_std,
        save_path=os.path.join(cfg.RESULTS_DIR, "predictions.png"),
    )

    # 5. Ablation overlay (optional)
    if include_ablation:
        ablation_history_path = cfg.ABLATION_CHECKPOINT.replace(".pth", "_history.pt")
        if os.path.exists(ablation_history_path) and os.path.exists(history_path):
            skip_hist = torch.load(history_path)
            base_hist = torch.load(ablation_history_path)
            plot_ablation_curves(
                skip_hist["train"], skip_hist["val"],
                base_hist["train"], base_hist["val"],
                save_path=os.path.join(cfg.RESULTS_DIR, "ablation_comparison.png"),
            )

            # Print ablation summary table
            def epochs_to(losses, threshold=0.01):
                for i, v in enumerate(losses):
                    if v < threshold:
                        return i
                return "N/A"

            print("\n── Ablation Summary ─────────────────────────────────────")
            print(f"  {'Model':<25}  {'Final Val MSE':>14}  {'Epochs to <0.01':>16}")
            print("─" * 60)
            print(f"  {'GRU + Skip':<25}  {skip_hist['val'][-1]:>14.5f}  "
                  f"{str(epochs_to(skip_hist['val'])):>16}")
            print(f"  {'GRU only (baseline)':<25}  {base_hist['val'][-1]:>14.5f}  "
                  f"{str(epochs_to(base_hist['val'])):>16}")
            print("─" * 60)
        else:
            print(
                "  Ablation data not found. "
                "Run  python train.py --ablation  first."
            )

    print("\nEvaluation complete.  All plots saved to results/\n")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained GRU+Skip model")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Include ablation comparison plot (requires ablation model to be trained)",
    )
    args = parser.parse_args()

    cfg = Config()
    run_evaluation(cfg, include_ablation=args.ablation)

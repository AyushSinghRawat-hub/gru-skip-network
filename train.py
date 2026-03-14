"""
train.py — Training loop with gradient clipping, LR scheduling, and checkpointing.

Run:
    python train.py                 # trains GRU+Skip (main model)
    python train.py --ablation      # also trains GRU-only for comparison

Training pipeline per epoch:
    1. Forward pass through GRUWithSkip
    2. Compute MSE loss
    3. loss.backward() — runs BPTT (Backpropagation Through Time)
    4. clip_grad_norm_  — prevents exploding gradients
    5. optimizer.step() — update weights
    6. ReduceLROnPlateau — reduce LR if val loss stagnates
    7. Checkpoint        — save model state if val loss improved
"""

import argparse
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np

from config import Config
from dataset import create_dataloaders
from model import GRUWithSkip, init_weights


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random seeds so results are reproducible across runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float,
    device: torch.device,
) -> float:
    """
    Run one full pass over the training set.

    Key steps explained:
        optimizer.zero_grad()      — must clear accumulated gradients before each batch
        loss.backward()            — BPTT: unrolls the GRU and computes all gradients
        clip_grad_norm_            — rescales gradient vector if its norm > grad_clip;
                                     prevents the exploding gradient problem common in RNNs
        optimizer.step()           — apply Adam update to all parameters

    Returns:
        avg_loss (float): mean MSE over all batches in this epoch
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        # Gradient clipping — non-negotiable for RNNs trained with BPTT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model on a DataLoader without computing gradients.

    model.eval() disables dropout and switches BatchNorm/LayerNorm to inference mode.
    torch.no_grad() prevents gradient tape allocation, saving memory and compute.

    Returns:
        avg_loss (float): mean MSE over all batches
    """
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        total_loss += criterion(model(X_batch), y_batch).item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cfg: Config,
    use_skip: bool = True,
    verbose: bool = True,
) -> Tuple[GRUWithSkip, List[float], List[float]]:
    """
    Full training run for GRUWithSkip.

    Args:
        cfg      (Config): hyperparameter configuration
        use_skip (bool):   True → train GRU+Skip (main model)
                           False → train GRU-only (ablation baseline)
        verbose  (bool):   print progress every 10 epochs

    Returns:
        model        (GRUWithSkip): best model loaded from checkpoint
        train_losses (List[float]): per-epoch training MSE
        val_losses   (List[float]): per-epoch validation MSE
    """
    set_seed(cfg.SEED)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label  = "GRU + Skip" if use_skip else "GRU only (ablation)"
    checkpoint_path = cfg.CHECKPOINT_PATH if use_skip else cfg.ABLATION_CHECKPOINT

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training: {label}")
        print(f"  Device:   {device}")
        print(f"  Epochs:   {cfg.EPOCHS}  |  LR: {cfg.LEARNING_RATE}  |  "
              f"Hidden: {cfg.HIDDEN_SIZE}  |  Seq: {cfg.SEQ_LEN}")
        print(f"{'='*60}")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader, _, norm_mean, norm_std = create_dataloaders(cfg)

    # Save normalisation stats so evaluate.py can denormalise predictions
    if use_skip:
        torch.save({"mean": norm_mean, "std": norm_std}, cfg.NORM_STATS_PATH)

    # ── Model ───────────────────────────────────────────────────────────
    model = GRUWithSkip(
        input_size=cfg.INPUT_SIZE,
        hidden_size=cfg.HIDDEN_SIZE,
        output_size=cfg.OUTPUT_SIZE,
        use_skip=use_skip,
    ).to(device)
    init_weights(model)   # orthogonal init for recurrent weight stability

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  Parameters: {total_params:,}\n")

    # ── Optimiser & loss ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()

    # ReduceLROnPlateau: halve the LR after `patience` epochs with no val improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg.LR_PATIENCE,
        factor=cfg.LR_FACTOR,
        min_lr=cfg.LR_MIN,
    )

    # ── Training loop ────────────────────────────────────────────────────
    train_losses: List[float] = []
    val_losses:   List[float] = []
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(cfg.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.GRAD_CLIP, device
        )
        val_loss = evaluate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Step the LR scheduler based on validation loss
        scheduler.step(val_loss)

        # Checkpoint: save best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        if verbose and epoch % 10 == 0:
            elapsed = time.time() - t_start
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{cfg.EPOCHS}  |  "
                f"Train MSE: {train_loss:.5f}  |  Val MSE: {val_loss:.5f}  |  "
                f"LR: {current_lr:.2e}  |  {elapsed:.1f}s"
            )

    elapsed_total = time.time() - t_start
    if verbose:
        print(f"\n  Best val MSE: {best_val_loss:.5f}  (saved → {checkpoint_path})")
        print(f"  Total training time: {elapsed_total:.1f}s")

    # Reload best checkpoint before returning
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Save loss history alongside the checkpoint
    history_path = checkpoint_path.replace(".pth", "_history.pt")
    torch.save({"train": train_losses, "val": val_losses}, history_path)

    return model, train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(cfg: Config) -> None:
    """
    Train both GRU+Skip and GRU-only under identical conditions and print a
    comparison table to demonstrate the architectural benefit of skip connections.

    Same seed, same hyperparameters, same data — only the skip branch differs.
    """
    print("\n" + "="*60)
    print("  ABLATION STUDY")
    print("="*60)

    _, train_skip, val_skip = train(cfg, use_skip=True,  verbose=True)
    _, train_base, val_base = train(cfg, use_skip=False, verbose=True)

    final_train_skip = train_skip[-1]
    final_val_skip   = val_skip[-1]
    final_train_base = train_base[-1]
    final_val_base   = val_base[-1]

    # Epochs to reach val_loss < 0.01 (or report N/A)
    def epochs_to(losses, threshold=0.01):
        for i, v in enumerate(losses):
            if v < threshold:
                return i
        return "N/A"

    converge_skip = epochs_to(val_skip)
    converge_base = epochs_to(val_base)

    print("\n" + "─"*60)
    print(f"  {'Model':<25}  {'Final Train MSE':>16}  {'Final Val MSE':>14}  {'Epochs to <0.01':>16}")
    print("─"*60)
    print(f"  {'GRU + Skip (main)':<25}  {final_train_skip:>16.5f}  {final_val_skip:>14.5f}  {str(converge_skip):>16}")
    print(f"  {'GRU only (baseline)':<25}  {final_train_base:>16.5f}  {final_val_base:>14.5f}  {str(converge_base):>16}")
    print("─"*60)
    print("\n  Ablation complete.  Loss histories saved to results/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU + Skip Connection model")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Also train GRU-only baseline and print comparison table",
    )
    args = parser.parse_args()

    cfg = Config()

    if args.ablation:
        run_ablation(cfg)
    else:
        train(cfg, use_skip=True, verbose=True)
        print("\nTraining complete.  Run  python evaluate.py  to see results.")

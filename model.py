"""
model.py — Core model architecture for the GRU + Skip Connection project.

Contains:
    CustomGRUCell   — GRU cell implemented from first principles (no nn.GRU / nn.GRUCell).
    GRUWithSkip     — Full sequence model using CustomGRUCell with a skip connection.
    init_weights    — Orthogonal weight initialisation for recurrent stability.
    test_model_shapes       — Output shape sanity check.
    verify_against_pytorch  — Assert CustomGRUCell matches nn.GRUCell numerically.

Design notes
────────────
This file matches PyTorch's GRUCell exactly — two differences from Cho et al. (2014):

  1. Candidate gate formulation:
       Original paper:  ñ_t = tanh( W · [r_t ⊙ h_{t-1},  x_t] + b )
                               (reset applied to h before the linear)
       PyTorch variant: ñ_t = tanh( W_in·x_t + b_in  +  r_t*(W_hn·h_{t-1} + b_hn) )
                               (reset applied to the result of the hidden-state linear)

  2. Final hidden state convention (IMPORTANT — this is why naive reimplementations fail):
       Original paper:  h_t = (1 − z_t) ⊙ h_{t-1}  +  z_t ⊙ ñ_t
                               (z ≈ 1 → adopt new candidate)
       PyTorch variant: h_t = (1 − z_t) ⊙ ñ_t  +  z_t ⊙ h_{t-1}
                               (z ≈ 1 → keep old state; z ≈ 0 → adopt new candidate)

Both formulations represent the same interpolation mechanism — the weights learn the
appropriate gate values for the convention being used.  PyTorch's convention is used
here to enable exact numerical verification against nn.GRUCell (atol=1e-5).
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# CustomGRUCell
# ─────────────────────────────────────────────────────────────────────────────

class CustomGRUCell(nn.Module):
    """
    A single GRU time step implemented entirely from nn.Linear, sigmoid, and tanh.

    Equations (PyTorch-compatible — matches nn.GRUCell exactly):
        r_t = σ( W_ir · x_t + b_ir  +  W_hr · h_{t-1} + b_hr )        reset gate
        z_t = σ( W_iz · x_t + b_iz  +  W_hz · h_{t-1} + b_hz )        update gate
        ñ_t = tanh( W_in · x_t + b_in  +  r_t * (W_hn · h_{t-1} + b_hn) )  candidate
        h_t = (1 − z_t) * ñ_t  +  z_t * h_{t-1}                       new hidden state
              ↑ PyTorch convention: z≈1 → keep old state, z≈0 → adopt new candidate
              (original paper uses the opposite z labelling; same interpolation logic)

    Why separate input/hidden linears instead of a single concatenated linear:
        This mirrors PyTorch's internal layout, making weight remapping for the
        verification test straightforward and unambiguous.

    Args:
        input_size  (int): dimensionality of x_t
        hidden_size (int): dimensionality of h_t
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # Reset gate — two halves: one for input, one for previous hidden state
        self.W_ir = nn.Linear(input_size,  hidden_size)   # input  → reset
        self.W_hr = nn.Linear(hidden_size, hidden_size)   # hidden → reset

        # Update gate — same structure
        self.W_iz = nn.Linear(input_size,  hidden_size)   # input  → update
        self.W_hz = nn.Linear(hidden_size, hidden_size)   # hidden → update

        # Candidate hidden state — same structure
        self.W_in = nn.Linear(input_size,  hidden_size)   # input  → candidate
        self.W_hn = nn.Linear(hidden_size, hidden_size)   # hidden → candidate

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Compute one GRU time step.

        Args:
            x_t   (Tensor): current input,         shape (batch, input_size)
            h_prev(Tensor): previous hidden state,  shape (batch, hidden_size)

        Returns:
            h_new (Tensor): updated hidden state,   shape (batch, hidden_size)
        """
        # ── Reset gate ─────────────────────────────────────────────────
        # r ∈ (0,1):  near 0 → forget past,  near 1 → expose full past memory.
        # Controls how much of h_prev is visible during candidate computation.
        r = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_prev))

        # ── Update gate ────────────────────────────────────────────────
        # z ∈ (0,1):  near 0 → keep old state,  near 1 → adopt new candidate.
        # This is the primary gradient highway: when z≈0, ∂h_t/∂h_{t-1} ≈ 1.0
        # instead of a shrinking Jacobian — this is how GRUs defeat vanishing gradients.
        z = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_prev))

        # ── Candidate hidden state ─────────────────────────────────────
        # What the new memory *would* be if we accepted it fully.
        # r gates how much of the past hidden state feeds into this proposal.
        n = torch.tanh(self.W_in(x_t) + r * self.W_hn(h_prev))

        # ── Final hidden state — smooth interpolation ──────────────────
        # PyTorch convention:  h_t = (1-z)*ñ_t + z*h_{t-1}
        # z ≈ 1  →  h_t ≈ h_{t-1}  →  keep old memory (gradient of h_{t-1} ≈ 1)
        # z ≈ 0  →  h_t ≈ ñ_t      →  adopt new candidate
        # NOTE: original Cho et al. paper uses h_t = (1-z)*h_{t-1} + z*ñ_t (z labelling flipped)
        # The interpolation mechanism is identical; only the gate polarity differs.
        h_new = (1.0 - z) * n + z * h_prev
        return h_new


# ─────────────────────────────────────────────────────────────────────────────
# GRUWithSkip
# ─────────────────────────────────────────────────────────────────────────────

class GRUWithSkip(nn.Module):
    """
    Sequence-to-sequence model built on CustomGRUCell with a per-step skip connection.

    Architecture (per time step t):
        x_t  ──→  CustomGRUCell  ──→  h_t  ──┐
          │                                    ├── LayerNorm ── output_layer ── ŷ_t
          └──→  skip_proj (Linear)  ──────────┘

    Skip connection rationale:
        Provides a second gradient path that bypasses the recurrent computation.
        During backprop: ∂L/∂x_t = ∂L/∂ŷ_t × (∂ŷ_t/∂GRU_path + ∂ŷ_t/∂skip_path).
        Even when the GRU gradient vanishes over long sequences via BPTT, the skip
        gradient (one linear layer) remains non-zero — early-layer learning never stops.

    Projection layer (skip_proj):
        input_size ≠ hidden_size in general, so a direct addition is illegal.
        skip_proj maps x_t from input_size → hidden_size before the addition.

    Layer Normalization:
        Applied after the addition of GRU output + skip, before the output projection.
        LayerNorm is preferred over BatchNorm in recurrent models because it normalises
        per-sample over the feature dimension — it does not depend on batch statistics
        and works correctly at every time step regardless of sequence length.

    Args:
        input_size  (int):  features per time step
        hidden_size (int):  GRU hidden state / skip projection output size
        output_size (int):  prediction dimensionality
        use_skip    (bool): set False for ablation (disables skip branch entirely)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_skip = use_skip

        # Core recurrent computation — from scratch, no nn.GRU
        self.gru_cell = CustomGRUCell(input_size, hidden_size)

        # Skip branch: projects input to hidden_size so addition is dimension-safe
        # Only created when use_skip=True to keep ablation model architecturally clean
        if self.use_skip:
            self.skip_proj = nn.Linear(input_size, hidden_size)

        # Layer normalisation — stabilises training after the skip addition
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Output projection — maps from hidden space to prediction space
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a full sequence through the GRU + skip network.

        Args:
            x (Tensor): input sequence, shape (batch_size, seq_len, input_size)

        Returns:
            outputs (Tensor): per-step predictions, shape (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, _ = x.shape

        # Initialise hidden state as zeros — standard practice for sequence models
        # when no explicit initial context is available
        h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]                          # current step: (batch, input_size)

            # Recurrent update
            h = self.gru_cell(x_t, h)                 # (batch, hidden_size)

            if self.use_skip:
                # Skip branch: project input to hidden_size and add
                # This addition is the gradient highway — see docstring above
                skip = self.skip_proj(x_t)            # (batch, hidden_size)
                combined = h + skip                   # element-wise addition
            else:
                combined = h                          # ablation: no skip path

            # Layer norm → output projection
            out = self.output_layer(self.layer_norm(combined))   # (batch, output_size)
            outputs.append(out)

        # Stack along the time dimension
        return torch.stack(outputs, dim=1)            # (batch, seq_len, output_size)


# ─────────────────────────────────────────────────────────────────────────────
# Weight initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_weights(model: nn.Module) -> None:
    """
    Apply orthogonal initialisation to all weight matrices; zero out all biases.

    Why orthogonal for RNNs:
        Orthogonal matrices have eigenvalues of magnitude exactly 1.  When gradients
        flow backward through a recurrent weight matrix whose eigenvalues are ≈1,
        the gradient norm is preserved across time steps — no shrinkage, no explosion
        in the early phase of training.  This is more important for RNNs than for
        feedforward networks because the same weight matrix is applied T times via BPTT.

    PyTorch's default (Kaiming uniform) is fine for convergence but orthogonal
    init often gives faster, more stable early training on sequence tasks.
    """
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def test_model_shapes() -> None:
    """
    Verify that GRUWithSkip produces the correct output shape for a dummy batch.
    Run this immediately after any architecture change to catch dimension errors early.
    """
    print("Running shape test …")
    cfg_input, cfg_hidden, cfg_output = 1, 64, 1
    batch, seq = 32, 50

    for use_skip in (True, False):
        model = GRUWithSkip(cfg_input, cfg_hidden, cfg_output, use_skip=use_skip)
        dummy = torch.randn(batch, seq, cfg_input)
        out = model(dummy)
        expected = (batch, seq, cfg_output)
        assert out.shape == expected, (
            f"[use_skip={use_skip}] Expected {expected}, got {out.shape}"
        )
        label = "GRUWithSkip" if use_skip else "GRU (no skip)"
        print(f"  {label}: output shape {tuple(out.shape)}  ✓")

    print("Shape test passed.\n")


def verify_against_pytorch() -> None:
    """
    Load identical weights into CustomGRUCell and PyTorch's nn.GRUCell.
    Run the same input and assert outputs match to atol=1e-5.

    Weight mapping (PyTorch packs weights in a specific order):
        weight_ih  shape (3*H, input_size):  rows [0:H]   → W_ir
                                              rows [H:2H]  → W_iz
                                              rows [2H:3H] → W_in
        weight_hh  shape (3*H, hidden_size): rows [0:H]   → W_hr
                                              rows [H:2H]  → W_hz
                                              rows [2H:3H] → W_hn
        bias_ih and bias_hh follow the same row ordering.

    If this assertion passes, CustomGRUCell is mathematically identical to
    PyTorch's implementation — the custom code is not an approximation.
    """
    print("Running PyTorch verification test …")
    input_size, hidden_size, batch = 4, 8, 3
    x = torch.randn(batch, input_size)
    h = torch.randn(batch, hidden_size)

    custom_cell  = CustomGRUCell(input_size, hidden_size)
    pytorch_cell = nn.GRUCell(input_size, hidden_size)

    H = hidden_size
    with torch.no_grad():
        # ── Input-side weights ───────────────────────────────────────────
        custom_cell.W_ir.weight.copy_(pytorch_cell.weight_ih[0:H])
        custom_cell.W_iz.weight.copy_(pytorch_cell.weight_ih[H:2*H])
        custom_cell.W_in.weight.copy_(pytorch_cell.weight_ih[2*H:3*H])

        # ── Hidden-side weights ──────────────────────────────────────────
        custom_cell.W_hr.weight.copy_(pytorch_cell.weight_hh[0:H])
        custom_cell.W_hz.weight.copy_(pytorch_cell.weight_hh[H:2*H])
        custom_cell.W_hn.weight.copy_(pytorch_cell.weight_hh[2*H:3*H])

        # ── Biases ───────────────────────────────────────────────────────
        custom_cell.W_ir.bias.copy_(pytorch_cell.bias_ih[0:H])
        custom_cell.W_iz.bias.copy_(pytorch_cell.bias_ih[H:2*H])
        custom_cell.W_in.bias.copy_(pytorch_cell.bias_ih[2*H:3*H])

        custom_cell.W_hr.bias.copy_(pytorch_cell.bias_hh[0:H])
        custom_cell.W_hz.bias.copy_(pytorch_cell.bias_hh[H:2*H])
        custom_cell.W_hn.bias.copy_(pytorch_cell.bias_hh[2*H:3*H])

    custom_out  = custom_cell(x, h)
    pytorch_out = pytorch_cell(x, h)

    max_diff = (custom_out - pytorch_out).abs().max().item()
    assert torch.allclose(custom_out, pytorch_out, atol=1e-5), (
        f"Verification failed — max diff: {max_diff:.2e}"
    )
    print(f"  CustomGRUCell ≡ nn.GRUCell  (max diff = {max_diff:.2e})  ✓")
    print("Verification test passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Run checks when executed directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_model_shapes()
    verify_against_pytorch()

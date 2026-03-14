"""
config.py — Central hyperparameter store for the GRU + Skip Connection project.

Every tunable value lives here. No magic numbers anywhere else in the codebase.
Import Config and pass a Config() instance down to every function that needs settings.
"""


class Config:
    # ------------------------------------------------------------------ #
    #  Reproducibility                                                     #
    # ------------------------------------------------------------------ #
    SEED: int = 42

    # ------------------------------------------------------------------ #
    #  Data                                                                #
    # ------------------------------------------------------------------ #
    N_SAMPLES: int = 5000        # number of sliding-window sequences
    SEQ_LEN: int = 50            # length of each input sequence (time steps)
    TRAIN_SPLIT: float = 0.8     # fraction of data used for training
    VAL_SPLIT: float = 0.1       # fraction used for validation (rest = test)

    # ------------------------------------------------------------------ #
    #  Model                                                               #
    # ------------------------------------------------------------------ #
    INPUT_SIZE: int = 1          # univariate time series → 1 feature per step
    HIDDEN_SIZE: int = 64        # GRU hidden state dimension
    OUTPUT_SIZE: int = 1         # predict one value per time step

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3  # Adam default; ReduceLROnPlateau will lower it
    EPOCHS: int = 100
    GRAD_CLIP: float = 1.0       # max gradient norm before clipping kicks in

    # ------------------------------------------------------------------ #
    #  Learning Rate Scheduling (ReduceLROnPlateau)                        #
    # ------------------------------------------------------------------ #
    LR_PATIENCE: int = 10        # epochs of no improvement before LR is reduced
    LR_FACTOR: float = 0.5       # multiply LR by this factor on plateau
    LR_MIN: float = 1e-5         # floor — LR will never drop below this

    # ------------------------------------------------------------------ #
    #  Paths                                                               #
    # ------------------------------------------------------------------ #
    RESULTS_DIR: str = "results/"
    CHECKPOINT_PATH: str = "results/best_model.pth"
    ABLATION_CHECKPOINT: str = "results/best_model_no_skip.pth"
    NORM_STATS_PATH: str = "results/norm_stats.pt"   # saved mean/std for eval

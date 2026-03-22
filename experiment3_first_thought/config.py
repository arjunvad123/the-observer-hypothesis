"""
Experiment 3 Configuration

Standalone first-thought study on the liquid/CfC substrate.
Paths, model sizes, and training defaults intentionally mirror
Experiment 6 where possible so results remain comparable.
"""

from pathlib import Path
import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get('EXP3_DATA_DIR', str(BASE_DIR / 'data')))
SYSTEM_DATA_PATH = DATA_DIR / 'system_data.h5'
TRAJECTORIES_PATH = DATA_DIR / 'executor_trajectories.h5'
TRAJECTORIES_SECONDARY_PATH = DATA_DIR / 'executor_trajectories_secondary.h5'
RESULTS_DIR = DATA_DIR / 'results'
CHECKPOINTS_DIR = DATA_DIR / 'checkpoints'

# ── Dynamical Systems ─────────────────────────────────────────────────

SYSTEM_DIM = 8
N_SYSTEMS = 8
N_TRAJECTORIES_PER_SYSTEM = 100
TRAJECTORY_LENGTH = 500
DT = 0.02
TOTAL_TRAJECTORIES = N_SYSTEMS * N_TRAJECTORIES_PER_SYSTEM
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SYSTEM_TYPES = [
    'lorenz', 'rossler', 'double_pendulum', 'coupled_oscillators',
    'van_der_pol', 'damped_sine', 'step_function', 'logistic_map',
]

# ── Executor (CfC) ───────────────────────────────────────────────────

EXECUTOR_HIDDEN_SIZE = 64
EXECUTOR_OUTPUT_SIZE = SYSTEM_DIM

# ── Observer (CfC) ───────────────────────────────────────────────────

OBSERVER_DEFAULT_SIZE = 50
OBSERVER_OUTPUT_SIZE = EXECUTOR_HIDDEN_SIZE

# ── Training ─────────────────────────────────────────────────────────

EXECUTOR_LR = 1e-3
EXECUTOR_EPOCHS = 50
OBSERVER_LR = 1e-3
OBSERVER_EPOCHS = 30
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_STEPS = 100
MIN_LR_RATIO = 0.1

# ── Probe Study ──────────────────────────────────────────────────────

K_SWEEP_PASSES = [1, 2, 3, 4, 6, 8]
DEFAULT_DELIBERATION_PASSES = 4
PROBE_MAX_SAMPLES = 50
OVERTHINKING_MARGIN = 0.01
STABILITY_CORR_THRESHOLD = 0.10
FIRST_THOUGHT_MARGIN = 0.0

# ── Device ───────────────────────────────────────────────────────────

def get_device():
    """Get the best available device. Prefer CUDA over MPS."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = get_device()

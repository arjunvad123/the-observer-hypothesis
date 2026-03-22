"""
Experiment 3 Control Baselines

Controls mirror the training setup but restrict evaluation to the
first-thought study. Each control is run through the same probe schema
as the trained observer so JSON outputs are directly comparable.
"""

from __future__ import annotations

from pathlib import Path

from . import config
from .observer_model import LiquidObserver, LinearBaseline
from .probes import run_all_probes
from .result_utils import write_json
from .trainer import load_model


def _load_or_init_model(checkpoint_path, fallback_model, label, device=None):
    """Load a trained control model when present, otherwise build a fallback."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        print(f'Loading {label} from {checkpoint_path}')
        return load_model(str(checkpoint_path), device)

    print(f'Warning: No trained {label} found. Using untrained.')
    return fallback_model


def run_control_probes(
    model,
    control_name,
    h5_path=None,
    device=None,
    max_samples=None,
    normalization_h5_path=None,
):
    """Run the Experiment 3 probes on a control model."""
    print(f"\n{'=' * 60}")
    print(f"CONTROL: {control_name}")
    print(f"{'=' * 60}\n")

    results = run_all_probes(
        model=model,
        h5_path=h5_path,
        device=device,
        save_dir=None,
        max_samples=max_samples,
        normalization_h5_path=normalization_h5_path,
    )
    results['summary']['control_name'] = control_name
    return results


def control_untrained(h5_path=None, device=None, max_samples=None, normalization_h5_path=None):
    """Random-weight observer with the same architecture as the trained model."""
    print('Creating untrained observer (random weights)...')
    model = LiquidObserver()
    return run_control_probes(
        model,
        'Untrained Observer',
        h5_path,
        device,
        max_samples,
        normalization_h5_path=normalization_h5_path,
    )


def control_linear(
    checkpoint_path=None,
    h5_path=None,
    device=None,
    max_samples=None,
    normalization_h5_path=None,
):
    """Single-matrix baseline tested under the same deliberation probe."""
    checkpoint_path = checkpoint_path or str(config.CHECKPOINTS_DIR / 'linear_baseline.pt')
    model = _load_or_init_model(checkpoint_path, LinearBaseline(), 'linear baseline', device)

    return run_control_probes(
        model,
        'Linear Baseline',
        h5_path,
        device,
        max_samples,
        normalization_h5_path=normalization_h5_path,
    )


def control_shuffled(
    checkpoint_path=None,
    h5_path=None,
    device=None,
    max_samples=None,
    normalization_h5_path=None,
):
    """Observer trained on temporally shuffled hidden-state sequences."""
    checkpoint_path = checkpoint_path or str(config.CHECKPOINTS_DIR / 'shuffled_observer.pt')
    model = _load_or_init_model(checkpoint_path, LiquidObserver(), 'shuffled observer', device)

    return run_control_probes(
        model,
        'Shuffled Observer',
        h5_path,
        device,
        max_samples,
        normalization_h5_path=normalization_h5_path,
    )


def control_wrong_executor(
    observer_path=None,
    h5_path=None,
    device=None,
    max_samples=None,
    normalization_h5_path=None,
):
    """Primary observer tested on hidden states from a different executor seed."""
    observer_path = observer_path or str(config.CHECKPOINTS_DIR / 'observer.pt')
    model = _load_or_init_model(observer_path, LiquidObserver(), 'observer for wrong-executor control', device)

    secondary_h5 = str(config.TRAJECTORIES_SECONDARY_PATH)
    if not Path(secondary_h5).exists():
        print('Warning: No secondary executor trajectories. Using primary trajectories as fallback.')
        secondary_h5 = h5_path or str(config.TRAJECTORIES_PATH)

    return run_control_probes(
        model,
        'Wrong-Executor (Different Seed)',
        secondary_h5,
        device,
        max_samples,
        normalization_h5_path=normalization_h5_path or str(config.TRAJECTORIES_PATH),
    )


def run_all_controls(observer_path=None, h5_path=None, device=None, save_dir=None, max_samples=None):
    """Run all Experiment 3 control baselines and save the comparison."""
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    save_dir = save_dir or str(config.RESULTS_DIR)
    normalization_h5_path = str(config.TRAJECTORIES_PATH)

    print('\n' + '=' * 70)
    print('RUNNING EXPERIMENT 3 CONTROL BASELINES')
    print('=' * 70)

    results = {
        'untrained': control_untrained(
            h5_path=h5_path,
            device=device,
            max_samples=max_samples,
            normalization_h5_path=normalization_h5_path,
        ),
        'linear': control_linear(
            h5_path=h5_path,
            device=device,
            max_samples=max_samples,
            normalization_h5_path=normalization_h5_path,
        ),
        'shuffled': control_shuffled(
            h5_path=h5_path,
            device=device,
            max_samples=max_samples,
            normalization_h5_path=normalization_h5_path,
        ),
        'wrong_executor': control_wrong_executor(
            observer_path=observer_path,
            h5_path=h5_path,
            device=device,
            max_samples=max_samples,
            normalization_h5_path=normalization_h5_path,
        ),
    }

    print('\n' + '=' * 60)
    print('CONTROL COMPARISON SUMMARY')
    print('=' * 60)
    for name, result in results.items():
        summary = result.get('summary', {})
        print(
            f"  {name:>25}: {summary.get('positive_count', 0)}/{summary.get('total_count', 0)} positive indicators"
        )

    if save_dir:
        output_path = Path(save_dir) / 'control_results.json'
        write_json(results, output_path)
        print(f"\nControl results saved to {output_path}")

    return results

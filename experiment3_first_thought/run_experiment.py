"""
Experiment 3: First Thought vs. Reasoned Explanation
====================================================

Standalone liquid/CfC implementation of the first-thought study.
A CfC observer watches a CfC executor trained on eight dynamical
systems, then is evaluated on whether immediate predictions outperform
iterative deliberation as K increases.

Phases:
    1. Extract — Generate dynamical systems, train executors, extract hidden states
    2. Train   — Train observer, linear baseline, and shuffled control
    3. Probe   — Run the first-thought study and controls

Usage:
    python -m experiment3_first_thought.run_experiment
    python -m experiment3_first_thought.run_experiment --extract-only
    python -m experiment3_first_thought.run_experiment --train-only
    python -m experiment3_first_thought.run_experiment --probe-only
"""

from __future__ import annotations

import argparse

import h5py
import torch

from . import config
from .result_utils import write_json


def phase_extract(device=None):
    """Phase 1: Generate systems, train executors, and extract hidden states."""
    from .dynamical_systems import generate_all_systems, save_system_data
    from .executor_model import (
        extract_hidden_trajectories,
        load_executor,
        train_executor,
        train_secondary_executor,
    )

    device = device or config.DEVICE

    print('=' * 70)
    print('PHASE 1: SYSTEM GENERATION & EXECUTOR TRAINING')
    print('=' * 70)

    print('\n--- Generating Dynamical Systems ---')
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not config.SYSTEM_DATA_PATH.exists():
        data = generate_all_systems()
        save_system_data(data)
    else:
        print(f'  System data already exists at {config.SYSTEM_DATA_PATH}')

    print('\n--- Training Primary Executor ---')
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    executor_path = config.CHECKPOINTS_DIR / 'executor.pt'
    if not executor_path.exists():
        train_executor(device=device)
    else:
        print(f'  Executor already trained at {executor_path}')

    print('\n--- Extracting Hidden State Trajectories ---')
    if not config.TRAJECTORIES_PATH.exists():
        executor = load_executor(str(executor_path), device)
        extract_hidden_trajectories(executor, output_h5_path=str(config.TRAJECTORIES_PATH), device=device)
    else:
        print(f'  Trajectories already exist at {config.TRAJECTORIES_PATH}')

    print('\n--- Training Secondary Executor (wrong-executor control) ---')
    secondary_path = config.CHECKPOINTS_DIR / 'executor_secondary.pt'
    if not secondary_path.exists():
        train_secondary_executor(device=device)
    else:
        print(f'  Secondary executor already trained at {secondary_path}')

    print('\n--- Extracting Secondary Hidden State Trajectories ---')
    if not config.TRAJECTORIES_SECONDARY_PATH.exists():
        secondary_executor = load_executor(str(secondary_path), device)
        extract_hidden_trajectories(
            secondary_executor,
            output_h5_path=str(config.TRAJECTORIES_SECONDARY_PATH),
            device=device,
        )
    else:
        print(f'  Secondary trajectories already exist at {config.TRAJECTORIES_SECONDARY_PATH}')

    with h5py.File(str(config.TRAJECTORIES_PATH), 'r') as handle:
        print(f"\n  Primary hidden states: {handle['hidden_states'].shape}")
        print(f"  System states: {handle['system_states'].shape}")

    print('\nPhase 1 complete.')


def phase_train(device=None):
    """Phase 2: Train the observer and Experiment 3 controls."""
    from .trainer import train_linear_baseline, train_observer, train_shuffled_observer

    device = device or config.DEVICE

    print('\n' + '=' * 70)
    print('PHASE 2: OBSERVER TRAINING')
    print('=' * 70)

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print('\n--- Primary Observer ---')
    observer_result = train_observer(device=device)

    print('\n--- Linear Baseline ---')
    linear_result = train_linear_baseline(device=device)

    print('\n--- Shuffled Observer ---')
    shuffled_result = train_shuffled_observer(device=device)

    summary = {
        'observer_final_train_loss': observer_result['history']['train_loss'][-1],
        'observer_final_val_loss': observer_result['history']['val_loss'][-1]
        if observer_result['history']['val_loss']
        else None,
        'linear_final_train_loss': linear_result['history']['train_loss'][-1],
        'linear_final_val_loss': linear_result['history']['val_loss'][-1]
        if linear_result['history']['val_loss']
        else None,
        'shuffled_final_train_loss': shuffled_result['history']['train_loss'][-1],
        'shuffled_final_val_loss': shuffled_result['history']['val_loss'][-1]
        if shuffled_result['history']['val_loss']
        else None,
    }

    write_json(summary, config.RESULTS_DIR / 'training_summary.json')

    print('\n' + '-' * 60)
    print('Training Summary:')
    for key, value in summary.items():
        print(f'  {key}: {value}')

    return summary


def phase_probe(device=None):
    """Phase 3: Run Experiment 3 probes and control comparisons."""
    from .controls import run_all_controls
    from .probes import run_all_probes
    from .trainer import load_model

    device = device or config.DEVICE

    print('\n' + '=' * 70)
    print('PHASE 3: FIRST-THOUGHT PROBES')
    print('=' * 70)

    observer_path = config.CHECKPOINTS_DIR / 'observer.pt'
    if not observer_path.exists():
        print(f'ERROR: No trained observer found at {observer_path}')
        print('Run with --train-only first.')
        return None

    model = load_model(str(observer_path), device)

    probe_results = run_all_probes(
        model=model,
        h5_path=str(config.TRAJECTORIES_PATH),
        device=device,
        save_dir=str(config.RESULTS_DIR),
        max_samples=config.PROBE_MAX_SAMPLES,
    )
    control_results = run_all_controls(
        observer_path=str(observer_path),
        h5_path=str(config.TRAJECTORIES_PATH),
        device=device,
        save_dir=str(config.RESULTS_DIR),
        max_samples=config.PROBE_MAX_SAMPLES,
    )

    print('\n' + '=' * 70)
    print('FINAL COMPARISON: OBSERVER vs CONTROLS')
    print('=' * 70)

    observer_pos = probe_results['summary']['positive_count']
    observer_total = probe_results['summary']['total_count']
    print(f"  {'Trained Observer':>25}: {observer_pos}/{observer_total} positive indicators")
    for name, result in control_results.items():
        summary = result.get('summary', {})
        print(
            f"  {name:>25}: {summary.get('positive_count', 0)}/{summary.get('total_count', 0)} positive indicators"
        )

    best_control = max(
        result.get('summary', {}).get('positive_count', 0)
        for result in control_results.values()
    ) if control_results else 0
    if observer_pos > best_control:
        print('\n  Interpretation: trained observer exceeds every control on the Experiment 3 indicators.')
    elif observer_pos == best_control:
        print('\n  Interpretation: trained observer matches the best control; the first-thought effect may not be specific.')
    else:
        print('\n  Interpretation: at least one control exceeds the trained observer on this probe suite.')

    return {'probes': probe_results, 'controls': control_results}

def run(extract=True, train=True, probe=True, device=None):
    """Run the full Experiment 3 pipeline."""
    device = device or config.DEVICE

    print('=' * 70)
    print('EXPERIMENT 3: FIRST THOUGHT VS. REASONED EXPLANATION')
    print('=' * 70)
    print(f'Device: {device}')
    print(f'Executor: CfC {config.EXECUTOR_HIDDEN_SIZE} hidden neurons')
    print(f'Observer: CfC {config.OBSERVER_DEFAULT_SIZE} hidden neurons')
    print(f'Data: {config.TOTAL_TRAJECTORIES} trajectories x {config.TRAJECTORY_LENGTH} timesteps')
    print(f'K sweep: {config.K_SWEEP_PASSES}')
    print()

    if extract:
        phase_extract(device)
    if train:
        phase_train(device)
    if probe:
        results = phase_probe(device)
    else:
        results = None

    if results is not None:
        write_json(results, config.RESULTS_DIR / 'run_summary.json')

    print('\n' + '=' * 70)
    print('EXPERIMENT 3 COMPLETE')
    print('=' * 70)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment 3: First Thought vs. Reasoned Explanation'
    )
    parser.add_argument('--extract-only', action='store_true', help='Only generate systems and train executors')
    parser.add_argument('--train-only', action='store_true', help='Only train observers (executor data must exist)')
    parser.add_argument('--probe-only', action='store_true', help='Only run Experiment 3 probes (trained models must exist)')
    parser.add_argument('--device', type=str, default=None, help='Torch device (auto-detected if not set)')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else config.DEVICE

    if args.extract_only:
        phase_extract(device)
    elif args.train_only:
        phase_train(device)
    elif args.probe_only:
        phase_probe(device)
    else:
        run(device=device)

"""
Experiment 3 Probes

A compact probe suite focused on the first-thought hypothesis.
The observer is only trained on next-hidden-state prediction; these
analyses ask whether immediate predictions outperform iterative
self-refinement and how that tradeoff changes as deliberation depth grows.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config
from .dataset import TrajectoryDataset
from .result_utils import write_json


def _get_test_dataset(h5_path=None, normalization_h5_path=None):
    """Load the held-out split with normalization stats from a reference HDF5."""
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    normalization_h5_path = normalization_h5_path or str(config.TRAJECTORIES_PATH)

    train_ds = TrajectoryDataset(normalization_h5_path, split='train', normalize=True)
    test_ds = TrajectoryDataset(h5_path, split='test', normalize=True, precompute_stats=False)
    test_ds.set_normalization_stats(train_ds.neuron_means, train_ds.neuron_stds)
    return test_ds


def _prediction_delta(current, previous):
    """Root-mean-square delta between consecutive final-step predictions."""
    return torch.sqrt(torch.mean((current - previous) ** 2)).item()


def _pearson_corr(xs, ys):
    """Safe Pearson correlation returning 0.0 for degenerate inputs."""
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _predict_with_deliberation(model, input_t, total_passes):
    """Run K total passes, feeding each final-step prediction back in."""
    if total_passes < 1:
        raise ValueError('total_passes must be >= 1')

    current_input = input_t.clone()
    final_predictions = []
    pass_durations_ms = []

    for pass_idx in range(total_passes):
        start = time.perf_counter()
        pred = model(current_input)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if isinstance(pred, tuple):
            pred = pred[0]

        final_predictions.append(pred[:, -1:, :].detach())
        pass_durations_ms.append(elapsed_ms)

        if pass_idx < total_passes - 1:
            current_input = torch.cat([current_input[:, 1:, :], pred[:, -1:, :]], dim=1)

    return final_predictions, pass_durations_ms


def evaluate_deliberation_sweep(
    model,
    h5_path=None,
    device=None,
    max_samples=None,
    k_values=None,
    normalization_h5_path=None,
):
    """Evaluate the observer across a sweep of deliberation depths K."""
    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    max_samples = max_samples or config.PROBE_MAX_SAMPLES
    k_values = sorted(set(k_values or config.K_SWEEP_PASSES))
    normalization_h5_path = normalization_h5_path or str(config.TRAJECTORIES_PATH)

    model = model.to(device)
    model.eval()

    dataset = _get_test_dataset(h5_path, normalization_h5_path=normalization_h5_path)
    loss_fn = nn.MSELoss(reduction='none')

    max_k = max(k_values)
    per_k_errors = {k: [] for k in k_values}
    per_k_wall_clock_ms = {k: [] for k in k_values}
    per_k_final_deltas = {k: [] for k in k_values}
    per_k_mean_path_deltas = {k: [] for k in k_values}

    all_sample_errors = []
    all_sample_stabilities = []

    with torch.no_grad():
        for idx in range(min(max_samples, len(dataset))):
            input_hidden, target_hidden = dataset[idx]
            input_t = input_hidden.unsqueeze(0).to(device)
            target_t = target_hidden.unsqueeze(0).to(device)
            target_final = target_t[:, -1:, :]

            predictions, pass_durations_ms = _predict_with_deliberation(model, input_t, max_k)

            running_deltas = []
            for pass_idx, final_pred in enumerate(predictions, start=1):
                error = loss_fn(final_pred, target_final).mean().item()

                if pass_idx == 1:
                    final_delta = 0.0
                    mean_path_delta = 0.0
                else:
                    delta = _prediction_delta(final_pred, predictions[pass_idx - 2])
                    running_deltas.append(delta)
                    final_delta = delta
                    mean_path_delta = float(np.mean(running_deltas))

                if pass_idx in per_k_errors:
                    per_k_errors[pass_idx].append(error)
                    per_k_wall_clock_ms[pass_idx].append(float(np.sum(pass_durations_ms[:pass_idx])))
                    per_k_final_deltas[pass_idx].append(final_delta)
                    per_k_mean_path_deltas[pass_idx].append(mean_path_delta)

                    if pass_idx > 1:
                        all_sample_errors.append(error)
                        all_sample_stabilities.append(-final_delta)

    curve = []
    for k in k_values:
        curve.append({
            'k': int(k),
            'error_mean': float(np.mean(per_k_errors[k])),
            'error_std': float(np.std(per_k_errors[k])),
            'latency_units': int(k),
            'wall_clock_ms_mean': float(np.mean(per_k_wall_clock_ms[k])),
            'wall_clock_ms_std': float(np.std(per_k_wall_clock_ms[k])),
            'final_delta_mean': float(np.mean(per_k_final_deltas[k])),
            'final_delta_std': float(np.std(per_k_final_deltas[k])),
            'mean_path_delta_mean': float(np.mean(per_k_mean_path_deltas[k])),
            'mean_path_delta_std': float(np.std(per_k_mean_path_deltas[k])),
        })

    error_values = np.array([entry['error_mean'] for entry in curve], dtype=np.float64)
    latency_values = np.array([entry['latency_units'] for entry in curve], dtype=np.float64)
    deliberative_entries = [entry for entry in curve if entry['k'] > 1]

    best_idx = int(np.argmin(error_values))
    best_k = int(curve[best_idx]['k'])
    is_monotonic_degradation = bool(np.all(np.diff(error_values) >= -1e-12))
    overthinking_detected = bool(
        best_k < int(curve[-1]['k']) and error_values[-1] > np.min(error_values) * (1.0 + config.OVERTHINKING_MARGIN)
    )

    reaction_time = {
        'curve': [
            {
                'latency_units': entry['latency_units'],
                'error_mean': entry['error_mean'],
                'wall_clock_ms_mean': entry['wall_clock_ms_mean'],
            }
            for entry in curve
        ],
        'latency_error_slope': float(np.polyfit(latency_values, error_values, deg=1)[0]) if len(curve) > 1 else 0.0,
        'latency_error_corr': _pearson_corr(latency_values, error_values),
        'best_latency_units': best_k,
        'deliberation_hurts': bool(error_values[-1] > error_values[0] * (1.0 + config.OVERTHINKING_MARGIN)),
    }

    stability = {
        'curve': [
            {
                'k': entry['k'],
                'final_delta_mean': entry['final_delta_mean'],
                'final_delta_std': entry['final_delta_std'],
                'mean_path_delta_mean': entry['mean_path_delta_mean'],
                'mean_path_delta_std': entry['mean_path_delta_std'],
                'error_mean': entry['error_mean'],
            }
            for entry in curve
        ],
        'stability_accuracy_corr': _pearson_corr(all_sample_stabilities, [-err for err in all_sample_errors]),
        'delta_error_corr': _pearson_corr([-score for score in all_sample_stabilities], all_sample_errors),
        'most_stable_k': int(
            min(deliberative_entries, key=lambda entry: entry['final_delta_mean'])['k']
        ) if deliberative_entries else 1,
    }

    k_sweep = {
        'curve': curve,
        'best_k': best_k,
        'is_monotonic_degradation': is_monotonic_degradation,
        'overthinking_detected': overthinking_detected,
    }

    return {
        'k_sweep': k_sweep,
        'reaction_time': reaction_time,
        'stability': stability,
        'sample_count': int(min(max_samples, len(dataset))),
    }


def run_all_probes(
    model,
    h5_path=None,
    device=None,
    save_dir=None,
    max_samples=None,
    normalization_h5_path=None,
):
    """Run the full Experiment 3 probe suite and optionally save it."""
    device = device or config.DEVICE
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    max_samples = max_samples or config.PROBE_MAX_SAMPLES
    normalization_h5_path = normalization_h5_path or str(config.TRAJECTORIES_PATH)

    print('=' * 60)
    print('EXPERIMENT 3 PROBES — FIRST THOUGHT STUDY')
    print('=' * 60)
    print()

    sweep_results = evaluate_deliberation_sweep(
        model,
        h5_path=h5_path,
        device=device,
        max_samples=max_samples,
        k_values=config.K_SWEEP_PASSES,
        normalization_h5_path=normalization_h5_path,
    )

    curve_by_k = {entry['k']: entry for entry in sweep_results['k_sweep']['curve']}
    first_entry = curve_by_k[1]
    default_entry = curve_by_k[config.DEFAULT_DELIBERATION_PASSES]

    first_thought = {
        'first_pass_error': first_entry['error_mean'],
        'default_deliberation_error': default_entry['error_mean'],
        'first_pass_std': first_entry['error_std'],
        'default_deliberation_std': default_entry['error_std'],
        'default_deliberation_passes': int(config.DEFAULT_DELIBERATION_PASSES),
        'first_better_than_deliberation': bool(
            first_entry['error_mean'] <= default_entry['error_mean'] * (1.0 + config.FIRST_THOUGHT_MARGIN)
        ),
        'deliberation_ratio': default_entry['error_mean'] / max(first_entry['error_mean'], 1e-8),
        'latency_units': {
            'first_pass': 1,
            'default_deliberation': int(config.DEFAULT_DELIBERATION_PASSES),
        },
    }

    k_sweep = sweep_results['k_sweep']
    reaction_time = sweep_results['reaction_time']
    stability = sweep_results['stability']

    indicators = {
        'first_thought_advantage': first_thought['first_better_than_deliberation'],
        'overthinking_detected': k_sweep['overthinking_detected'],
        'latency_tradeoff_detected': reaction_time['latency_error_slope'] > 0.0,
        'stability_tracks_accuracy': stability['stability_accuracy_corr'] > config.STABILITY_CORR_THRESHOLD,
    }

    positive = sum(1 for value in indicators.values() if value)
    results = {
        'metadata': {
            'evaluated_h5_path': str(h5_path),
            'normalization_h5_path': str(normalization_h5_path),
            'max_samples': int(max_samples),
        },
        'first_thought': first_thought,
        'k_sweep': k_sweep,
        'reaction_time': reaction_time,
        'stability': stability,
        'summary': {
            'indicators': {key: bool(value) for key, value in indicators.items()},
            'positive_count': positive,
            'total_count': len(indicators),
            'sample_count': sweep_results['sample_count'],
        },
    }

    print('SUMMARY')
    print('-' * 50)
    print(f"  First thought advantage:    {results['summary']['indicators']['first_thought_advantage']}")
    print(f"  Overthinking detected:      {results['summary']['indicators']['overthinking_detected']}")
    print(f"  Latency tradeoff detected:  {results['summary']['indicators']['latency_tradeoff_detected']}")
    print(f"  Stability tracks accuracy:  {results['summary']['indicators']['stability_tracks_accuracy']}")
    print(f"  Positive indicators:        {positive}/{len(indicators)}")
    print()

    if save_dir:
        _save_results(results, save_dir)

    return results


def _save_results(results, save_dir):
    """Save probe results to JSON."""
    write_json(results, Path(save_dir) / 'probe_results.json')

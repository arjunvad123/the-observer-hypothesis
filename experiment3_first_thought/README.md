# Experiment 3: First Thought vs. Reasoned Explanation

> **Status: Implemented**

Experiment 3 asks a narrow question:

When the observer predicts the executor's next hidden state, is the first answer better than the answer it reaches after repeatedly feeding its own predictions back into itself?

This directory contains a standalone version of that experiment on the same liquid neural network setup used in Experiment 6.

## What this experiment is doing

The observer watches the executor's hidden states and tries to predict the next hidden state.

At probe time, we compare two modes:

- **First thought**: one forward pass, no self-feedback
- **Deliberation**: run the observer multiple times, each time feeding the last prediction back in as the newest input step

The point is not to ask whether the observer can explain itself in words. The point is to measure whether extra internal processing helps or hurts prediction.

## What is included here

This implementation focuses on the core experiment only.

Included:
- single-pass vs. multi-pass prediction
- `K` sweep over multiple deliberation depths
- reaction-time analog using pass count as latency
- stability analysis based on how much the prediction changes from pass to pass
- control comparisons

Not included:
- language head
- explanation faithfulness analysis
- dual-path architecture

## Substrate

The substrate matches Experiment 6 so the results are comparable.

- **Executor**: CfC network trained on 8 dynamical systems
- **Observer**: CfC network trained to predict next executor hidden state
- **Systems**: Lorenz, Rossler, double pendulum, coupled oscillators, Van der Pol, damped sine, step function, logistic map
- **Dataset size**: 800 trajectories, 500 timesteps each

## Primary and secondary executor data

Two executor trajectory files are produced.

- **Primary**: hidden states from the main executor. This is what the observer is trained on.
- **Secondary**: hidden states from a different executor trained with a different seed. This is used for the wrong-executor control.

All probe evaluations use normalization statistics from the primary training split. That keeps the observer's input scaling fixed across the main evaluation and the wrong-executor control.

## Phases

### `--extract-only`

This phase builds the data used by the experiment.

It will:
- generate the dynamical-system trajectories
- train the primary executor
- extract primary executor hidden states
- train a secondary executor with a different seed
- extract secondary executor hidden states

Outputs:
- `system_data.h5`
- `executor.pt`
- `executor_trajectories.h5`
- `executor_secondary.pt`
- `executor_trajectories_secondary.h5`

### `--train-only`

This phase trains the models used in the probe phase.

It will:
- train the main observer
- train the linear baseline
- train the shuffled observer
- write `training_summary.json`

Outputs:
- `observer.pt`
- `linear_baseline.pt`
- `shuffled_observer.pt`
- `training_summary.json`

### `--probe-only`

This phase runs the first-thought study on the trained observer and the controls.

It writes:
- `probe_results.json`
- `control_results.json`

## Result files

### `probe_results.json`

This is the trained observer.

Main sections:
- `first_thought`: one-pass result vs. default deliberation depth
- `k_sweep`: error at each tested `K`
- `reaction_time`: latency/error tradeoff summary
- `stability`: how much the prediction drifts as deliberation continues
- `summary`: compact verdict fields

### `control_results.json`

This runs the same probe logic on four controls.

- **Untrained observer**: same architecture, random weights
- **Linear baseline**: simple linear predictor instead of a CfC observer
- **Shuffled observer**: trained on temporally shuffled hidden-state sequences
- **Wrong-executor**: trained observer tested on the secondary executor's hidden states

These controls answer different questions:
- Is the effect just an architecture artifact?
- Do liquid dynamics matter?
- Does temporal order matter?
- Is the observer specific to its own executor?

## What “error” means

All main scores are mean squared error on the next hidden state.

- lower error is better
- a ratio above `1.0` means deliberation made the prediction worse

This is a regression experiment, not a classification task, so there is no accuracy percentage.

## Running

Install dependencies:

```bash
pip install -r experiment3_first_thought/requirements.txt
```

Run the full pipeline:

```bash
python -m experiment3_first_thought.run_experiment
```

Run individual phases:

```bash
python -m experiment3_first_thought.run_experiment --extract-only
python -m experiment3_first_thought.run_experiment --train-only
python -m experiment3_first_thought.run_experiment --probe-only
```

Use a different data directory if needed:

```bash
EXP3_DATA_DIR=/path/to/exp3-data python -m experiment3_first_thought.run_experiment
```

## Results

### Probes

| Probe | Metric | Value | Interpretation |
|-------|--------|-------|---------|
| First Thought | `K=1` vs `K=4` | `0.000317` vs `0.000343` | First better |
| K Sweep | Best `K`, monotonicity | `best_k=1`, monotonic degradation | Overthinking present |
| Reaction Time | latency/error correlation | `0.9704` | More passes hurt |
| Stability | stability/error correlation | `0.8106` | Drift tracks worse error |

**Score: 4/4 positive indicators**

### Controls

| Control | Positive Indicators | Key Difference from Trained |
|---------|--------------------|-----------------------------|
| Untrained | 2/4 | Much higher base error, weak first-thought effect |
| Linear | 3/4 | Lower one-pass error than the trained observer |
| Shuffled | 3/4 | Temporal order is not needed for the overthinking pattern |
| Wrong-Executor | 3/4 | Error jumps from `0.000317` to `17.2251` on foreign trajectories |

## Key findings

1. The trained observer does show the intended Experiment 3 pattern. One pass is best, and every extra pass in the `K` sweep makes the prediction worse.
2. The new `K` sweep matters because it shows this is not a narrow `K=1` vs `K=4` artifact. The curve keeps getting worse through `K=8`.
3. The reaction-time analog and stability analysis point in the same direction. More processing depth leads to more drift, and more drift goes with higher error.
4. The wrong-executor control is the clearest positive result. The observer is strongly tied to its own executor.
5. The first-thought effect is not unique to the trained liquid observer. The linear baseline and shuffled observer also show it, and the linear model has lower one-pass error on this task.

## Files

- `config.py`: paths and experiment settings
- `dynamical_systems.py`: data generation
- `executor_model.py`: executor training and hidden-state extraction
- `observer_model.py`: observer and linear baseline
- `trainer.py`: observer/control training
- `probes.py`: first-thought probe logic
- `controls.py`: control runs
- `run_experiment.py`: entry point

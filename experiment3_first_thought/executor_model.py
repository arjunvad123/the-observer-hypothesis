"""
CfC Executor Model

A Closed-form Continuous-time neural network trained on multi-system
next-step prediction. The executor's hidden states — the continuous-time
"thoughts" of the network — are what the observer will learn to predict.

Architecture: Input projection -> CfC with AutoNCP wiring -> Output head
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

from ncps.torch import CfC
from ncps.wirings import AutoNCP

from . import config


class LiquidExecutor(nn.Module):
    """CfC executor trained to predict next states of diverse dynamical systems."""

    def __init__(
        self,
        input_size=None,
        hidden_size=None,
        output_size=None,
    ):
        super().__init__()
        input_size = input_size or config.SYSTEM_DIM
        hidden_size = hidden_size or config.EXECUTOR_HIDDEN_SIZE
        output_size = output_size or config.EXECUTOR_OUTPUT_SIZE

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Input projection to match CfC input expectations
        self.input_proj = nn.Linear(input_size, hidden_size)

        # NCP-structured CfC
        self.wiring = AutoNCP(hidden_size, output_size)
        self.cfc = CfC(
            hidden_size,
            self.wiring,
            batch_first=True,
            return_sequences=True,
        )

        # Output head to map from CfC output to system dim
        self.output_head = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

        self.config = {
            'type': 'liquid_executor',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
        }

    def forward(self, x, h0=None):
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size)
            h0: optional initial hidden state

        Returns:
            predictions: (batch, seq_len, output_size)
        """
        projected = self.input_proj(x)  # (batch, seq_len, hidden_size)

        if h0 is not None:
            cfc_out, _ = self.cfc(projected, h0)
        else:
            cfc_out, _ = self.cfc(projected)

        predictions = self.output_head(cfc_out)
        return predictions

    def forward_with_states(self, x):
        """Step-by-step forward, capturing hidden state at every timestep.

        This is the key extraction method — captures the executor's "thoughts".

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            predictions: (batch, seq_len, output_size)
            all_hidden: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        projected = self.input_proj(x)

        all_hidden = []
        h = None

        for t in range(seq_len):
            inp = projected[:, t:t + 1, :]  # (batch, 1, hidden_size)
            if h is not None:
                out, h = self.cfc(inp, h)
            else:
                out, h = self.cfc(inp)
            all_hidden.append(h.detach())

        # Stack hidden states
        all_hidden = torch.stack(all_hidden, dim=1)  # (batch, seq_len, hidden_size)

        # Get predictions via normal forward
        predictions = self.forward(x)

        return predictions, all_hidden

    def count_parameters(self):
        """Total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def train_executor(system_data_path=None, save_path=None, device=None, seed=0):
    """Train executor on dynamical system prediction.

    Input: x(0..T-1), Target: x(1..T)
    Loss: MSE

    Returns training history dict.
    """
    from .dynamical_systems import load_system_data

    system_data_path = system_data_path or str(config.SYSTEM_DATA_PATH)
    save_path = save_path or str(config.CHECKPOINTS_DIR / 'executor.pt')
    device = device or config.DEVICE

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("TRAINING CfC EXECUTOR")
    print("=" * 60)

    # Load data
    data = load_system_data(system_data_path)
    trajectories = torch.tensor(data['trajectories'], dtype=torch.float32)

    # Split
    n_total = len(trajectories)
    n_train = int(n_total * config.TRAIN_RATIO)
    n_val = int(n_total * config.VAL_RATIO)

    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    train_traj = trajectories[train_idx]
    val_traj = trajectories[val_idx]

    # Create model
    model = LiquidExecutor().to(device)
    total, trainable = model.count_parameters()
    print(f"Executor parameters: {total:,} total, {trainable:,} trainable")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.EXECUTOR_LR, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()

    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(config.EXECUTOR_EPOCHS):
        model.train()
        epoch_losses = []

        # Mini-batch training
        perm = torch.randperm(len(train_traj))
        for start in range(0, len(train_traj), config.BATCH_SIZE):
            batch_idx = perm[start:start + config.BATCH_SIZE]
            batch = train_traj[batch_idx].to(device)

            # Input: x(0..T-2), Target: x(1..T-1)
            input_seq = batch[:, :-1, :]
            target_seq = batch[:, 1:, :]

            predictions = model(input_seq)
            loss = loss_fn(predictions, target_seq)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train = np.mean(epoch_losses)
        history['train_loss'].append(avg_train)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(val_traj), config.BATCH_SIZE):
                batch = val_traj[start:start + config.BATCH_SIZE].to(device)
                input_seq = batch[:, :-1, :]
                target_seq = batch[:, 1:, :]
                predictions = model(input_seq)
                loss = loss_fn(predictions, target_seq)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        history['val_loss'].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model.config,
                'optimizer_state_dict': optimizer.state_dict(),
                'seed': seed,
            }, save_path)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{config.EXECUTOR_EPOCHS} | "
                  f"Train: {avg_train:.6f} | Val: {avg_val:.6f}")

    print(f"  Best val loss: {best_val_loss:.6f}")
    return {'history': history, 'model': model, 'best_val_loss': best_val_loss}


def extract_hidden_trajectories(model, system_data_path=None, output_h5_path=None, device=None):
    """Run trained executor on all trajectories, save hidden states to HDF5.

    HDF5 structure:
        hidden_states: (800, 499, 64) — executor hidden state at each step
        system_states: (800, 500, 8)  — original dynamical system states
        system_types:  (800,)         — string labels
        interest_masks: (800, 500)    — boolean interest indicators
        predictions:    (800, 499, 8) — executor's predictions
    """
    from .dynamical_systems import load_system_data

    system_data_path = system_data_path or str(config.SYSTEM_DATA_PATH)
    output_h5_path = output_h5_path or str(config.TRAJECTORIES_PATH)
    device = device or config.DEVICE

    print("Extracting executor hidden state trajectories...")

    data = load_system_data(system_data_path)
    trajectories = torch.tensor(data['trajectories'], dtype=torch.float32)
    n_traj = len(trajectories)
    seq_len = trajectories.shape[1]

    model = model.to(device)
    model.eval()

    # Process in batches
    all_hidden = []
    all_predictions = []

    with torch.no_grad():
        for start in tqdm(range(0, n_traj, config.BATCH_SIZE), desc="Extracting"):
            batch = trajectories[start:start + config.BATCH_SIZE].to(device)
            input_seq = batch[:, :-1, :]  # (batch, seq_len-1, 8)

            predictions, hidden = model.forward_with_states(input_seq)

            all_hidden.append(hidden.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_hidden = np.concatenate(all_hidden, axis=0)      # (800, 499, 64)
    all_predictions = np.concatenate(all_predictions, axis=0)  # (800, 499, 8)

    # Save to HDF5
    Path(output_h5_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('hidden_states', data=all_hidden.astype(np.float32),
                         compression='gzip', compression_opts=4)
        f.create_dataset('system_states', data=data['trajectories'],
                         compression='gzip', compression_opts=4)
        f.create_dataset('predictions', data=all_predictions.astype(np.float32),
                         compression='gzip', compression_opts=4)
        f.create_dataset('interest_masks', data=data['interest_masks'],
                         compression='gzip', compression_opts=4)

        dt_str = h5py.string_dtype('ascii', 30)
        f.create_dataset('system_types',
                         data=data['system_types'].astype('S30'), dtype=dt_str)

        # Metadata
        f.attrs['n_trajectories'] = n_traj
        f.attrs['seq_len'] = all_hidden.shape[1]
        f.attrs['hidden_size'] = all_hidden.shape[2]
        f.attrs['system_dim'] = config.SYSTEM_DIM
        f.attrs['trajectory_length'] = seq_len

    print(f"  Hidden states shape: {all_hidden.shape}")
    print(f"  Saved to {output_h5_path}")

    return all_hidden


def train_secondary_executor(system_data_path=None, save_path=None, device=None):
    """Train a SECOND executor with different random seed.
    For the wrong-executor control.
    """
    save_path = save_path or str(config.CHECKPOINTS_DIR / 'executor_secondary.pt')

    print("\n" + "=" * 60)
    print("TRAINING SECONDARY CfC EXECUTOR (different seed)")
    print("=" * 60)

    return train_executor(
        system_data_path=system_data_path,
        save_path=save_path,
        device=device,
        seed=42,  # Different seed from primary (seed=0)
    )


def load_executor(checkpoint_path, device=None):
    """Load a saved executor from checkpoint."""
    device = device or config.DEVICE
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint['config']
    model = LiquidExecutor(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        output_size=model_config['output_size'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model

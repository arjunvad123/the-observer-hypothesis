"""
HDF5 Dataset Loader

Memory-mapped access to extracted CfC executor hidden states with:
- Per-neuron z-score normalization
- Train/val/test split
- Interest mask access for surprise probe
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from . import config


class TrajectoryDataset(Dataset):
    """Dataset of executor hidden state trajectories.

    Each item: (input_hidden, target_hidden)
    - input_hidden: (seq_len-1, hidden_size) — h(0..T-2)
    - target_hidden: (seq_len-1, hidden_size) — h(1..T-1) shifted by 1

    Per-neuron z-score normalization ensures balanced input to observer.
    """

    def __init__(
        self,
        h5_path=None,
        split='train',
        normalize=True,
        precompute_stats=True,
    ):
        self.h5_path = Path(h5_path or str(config.TRAJECTORIES_PATH))
        self.split = split
        self.normalize = normalize

        # Open HDF5 (keep handle open for fast repeated access)
        self._h5_file = h5py.File(self.h5_path, 'r')
        self.n_trajectories = self._h5_file.attrs['n_trajectories']
        self.seq_len = self._h5_file.attrs['seq_len']
        self.hidden_size = self._h5_file.attrs['hidden_size']
        self._hidden_ds = self._h5_file['hidden_states']
        self._interest_ds = self._h5_file['interest_masks']
        self._system_ds = self._h5_file['system_types']

        # Compute split indices
        n_train = int(self.n_trajectories * config.TRAIN_RATIO)
        n_val = int(self.n_trajectories * config.VAL_RATIO)

        if split == 'train':
            self.start_idx = 0
            self.end_idx = n_train
        elif split == 'val':
            self.start_idx = n_train
            self.end_idx = n_train + n_val
        elif split == 'test':
            self.start_idx = n_train + n_val
            self.end_idx = self.n_trajectories
        else:
            raise ValueError(f"Unknown split: {split}")

        self.n_items = self.end_idx - self.start_idx

        # Normalization statistics (per-neuron mean and std)
        self.neuron_means = None
        self.neuron_stds = None

        if normalize and precompute_stats:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute per-neuron z-score statistics from the training split."""
        print(f"Computing normalization statistics from training split...")

        n_sample = min(200, self.n_items)
        sample_indices = np.random.RandomState(42).choice(
            self.n_items, n_sample, replace=False
        ) + self.start_idx

        means = np.zeros(self.hidden_size, dtype=np.float64)
        sq_means = np.zeros(self.hidden_size, dtype=np.float64)
        count = 0

        for idx in sample_indices:
            data = self._hidden_ds[idx]  # (seq_len, hidden_size)
            means += data.mean(axis=0)
            sq_means += (data ** 2).mean(axis=0)
            count += 1

        means /= count
        sq_means /= count
        stds = np.sqrt(np.maximum(sq_means - means ** 2, 1e-8))

        self.neuron_means = torch.tensor(means, dtype=torch.float32)  # (hidden_size,)
        self.neuron_stds = torch.tensor(stds, dtype=torch.float32)

        print(f"  Neuron mean range: [{means.min():.4f}, {means.max():.4f}]")
        print(f"  Neuron std range:  [{stds.min():.4f}, {stds.max():.4f}]")

    def __del__(self):
        """Close HDF5 file handle."""
        try:
            if hasattr(self, '_h5_file') and self._h5_file and self._h5_file.id.valid:
                self._h5_file.close()
        except Exception:
            pass

    def set_normalization_stats(self, means, stds):
        """Set normalization stats externally (for val/test using train stats)."""
        self.neuron_means = means
        self.neuron_stds = stds

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        """Return a single training example.

        Returns:
            input_hidden: (seq_len-1, hidden_size) — h(0..T-2)
            target_hidden: (seq_len-1, hidden_size) — h(1..T-1)
        """
        actual_idx = self.start_idx + idx
        hidden = torch.tensor(self._hidden_ds[actual_idx], dtype=torch.float32)

        # Normalize per neuron
        if self.normalize and self.neuron_means is not None:
            hidden = (hidden - self.neuron_means) / self.neuron_stds

        # Input: h(0..T-2), Target: h(1..T-1)
        input_hidden = hidden[:-1, :]   # (seq_len-1, hidden_size)
        target_hidden = hidden[1:, :]   # (seq_len-1, hidden_size)

        return input_hidden, target_hidden

    def get_system_type(self, idx):
        """Return the system type string for this trajectory."""
        actual_idx = self.start_idx + idx
        s = self._system_ds[actual_idx]
        return s.decode() if isinstance(s, bytes) else s

    def get_interest_mask(self, idx):
        """Return the interest mask for this trajectory.
        Sliced to match hidden state seq_len (original minus 1).
        """
        actual_idx = self.start_idx + idx
        mask = self._interest_ds[actual_idx]  # (trajectory_length,)
        # Hidden states are from input seq (0..T-2), so mask aligns with [1:T-1]
        return mask[1:self.seq_len + 1]

    def get_raw_hidden(self, idx):
        """Return unnormalized hidden states for a trajectory."""
        actual_idx = self.start_idx + idx
        return torch.tensor(self._hidden_ds[actual_idx], dtype=torch.float32)


class ShuffledTrajectoryDataset(Dataset):
    """Control dataset: temporally shuffled hidden states.

    Same statistics, but temporal structure destroyed.
    """

    def __init__(self, base_dataset, seed=42):
        self.base = base_dataset
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        input_hidden, target_hidden = self.base[idx]

        seq_len = input_hidden.shape[0]
        perm = torch.tensor(self.rng.permutation(seq_len))

        input_shuffled = input_hidden[perm, :]
        target_shuffled = target_hidden[perm, :]

        return input_shuffled, target_shuffled


def create_dataloaders(h5_path=None, batch_size=None, num_workers=0):
    """Create train/val/test dataloaders with shared normalization stats.

    Returns:
        dict with 'train', 'val', 'test' DataLoaders and datasets
    """
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    batch_size = batch_size or config.BATCH_SIZE

    train_ds = TrajectoryDataset(h5_path, split='train', normalize=True)
    val_ds = TrajectoryDataset(h5_path, split='val', normalize=True, precompute_stats=False)
    test_ds = TrajectoryDataset(h5_path, split='test', normalize=True, precompute_stats=False)

    # Share normalization stats from training set
    val_ds.set_normalization_stats(train_ds.neuron_means, train_ds.neuron_stds)
    test_ds.set_normalization_stats(train_ds.neuron_means, train_ds.neuron_stds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_ds,
        'val_dataset': val_ds,
        'test_dataset': test_ds,
    }

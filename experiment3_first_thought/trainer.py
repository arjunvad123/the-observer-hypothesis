"""
Observer Trainer

Trains the CfC observer on executor hidden state prediction.
The observer's ONLY objective is prediction: given executor h(0..T),
predict h(T+1). All consciousness-like properties must emerge from
this objective alone.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .observer_model import LiquidObserver, LinearBaseline
from .dataset import TrajectoryDataset, ShuffledTrajectoryDataset, create_dataloaders


class Trainer:
    """Unified trainer for CfC observer models.

    Cosine LR schedule with warmup, gradient clipping, validation, checkpointing.
    """

    def __init__(
        self,
        model=None,
        lr=None,
        weight_decay=None,
        device=None,
    ):
        self.device = device or config.DEVICE
        self.model = model or LiquidObserver()
        self.model = self.model.to(self.device)

        lr = lr or config.OBSERVER_LR

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay or config.WEIGHT_DECAY,
        )

        total, trainable = self.model.count_parameters()
        print(f"Model parameters: {total:,} total, {trainable:,} trainable")

    def _create_scheduler(self, n_epochs, steps_per_epoch):
        """Cosine schedule with linear warmup."""
        total_steps = n_epochs * steps_per_epoch
        warmup_steps = min(config.WARMUP_STEPS, total_steps // 5)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return config.MIN_LR_RATIO + (1 - config.MIN_LR_RATIO) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(
        self,
        train_loader,
        val_loader=None,
        n_epochs=None,
        save_path=None,
        print_every=2,
    ):
        """Train the model.

        Returns:
            history dict with train/val losses
        """
        n_epochs = n_epochs or config.OBSERVER_EPOCHS

        print(f"Training for {n_epochs} epochs")
        print(f"Device: {self.device}")
        print("-" * 60)

        scheduler = self._create_scheduler(n_epochs, len(train_loader))
        loss_fn = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        for epoch in range(n_epochs):
            # ── Train ───────────────────────────────────────────
            self.model.train()
            epoch_losses = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}",
                        leave=False) if len(train_loader) > 5 else train_loader

            for input_hidden, target_hidden in pbar:
                input_hidden = input_hidden.to(self.device)
                target_hidden = target_hidden.to(self.device)

                predictions = self.model(input_hidden)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]

                loss = loss_fn(predictions, target_hidden)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)
                self.optimizer.step()
                scheduler.step()

                epoch_losses.append(loss.item())

                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(loss=f"{loss.item():.6f}")

            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)

            # ── Validate ────────────────────────────────────────
            avg_val_loss = None
            if val_loader is not None:
                avg_val_loss = self._validate(val_loader, loss_fn)
                history['val_loss'].append(avg_val_loss)

                if save_path and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save(save_path)

            # ── Log ─────────────────────────────────────────────
            if (epoch + 1) % print_every == 0:
                msg = f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {avg_train_loss:.6f}"
                if avg_val_loss is not None:
                    msg += f" | Val Loss: {avg_val_loss:.6f}"
                msg += f" | LR: {scheduler.get_last_lr()[0]:.2e}"
                print(msg)

        # Save final model if no validation
        if save_path and val_loader is None:
            self._save(save_path)

        return history

    def _validate(self, val_loader, loss_fn):
        """Run validation pass."""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for input_hidden, target_hidden in val_loader:
                input_hidden = input_hidden.to(self.device)
                target_hidden = target_hidden.to(self.device)

                predictions = self.model(input_hidden)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]

                loss = loss_fn(predictions, target_hidden)
                val_losses.append(loss.item())

        return np.mean(val_losses)

    def _save(self, save_path):
        """Save model checkpoint."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)


# ── Training functions ────────────────────────────────────────────────

def train_observer(h5_path=None, save_path=None, device=None, hidden_size=None):
    """Train primary observer model."""
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    save_path = save_path or str(config.CHECKPOINTS_DIR / 'observer.pt')
    device = device or config.DEVICE
    hidden_size = hidden_size or config.OBSERVER_DEFAULT_SIZE

    print("\n" + "=" * 60)
    print(f"Training CfC Observer (hidden_size={hidden_size})")
    print("=" * 60)

    dataloaders = create_dataloaders(h5_path)

    model = LiquidObserver(hidden_size=hidden_size)
    trainer = Trainer(model=model, device=device)

    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        save_path=save_path,
    )

    return {
        'history': history,
        'model': trainer.model,
        'train_dataset': dataloaders['train_dataset'],
        'val_dataset': dataloaders['val_dataset'],
        'test_dataset': dataloaders['test_dataset'],
    }


def train_linear_baseline(h5_path=None, save_path=None, device=None):
    """Train the linear baseline model."""
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    save_path = save_path or str(config.CHECKPOINTS_DIR / 'linear_baseline.pt')
    device = device or config.DEVICE

    print("\n" + "=" * 60)
    print("Training Linear Baseline")
    print("=" * 60)

    dataloaders = create_dataloaders(h5_path)

    model = LinearBaseline()
    trainer = Trainer(model=model, device=device)

    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        save_path=save_path,
    )

    return {'history': history, 'model': trainer.model}


def train_shuffled_observer(h5_path=None, save_path=None, device=None, hidden_size=None):
    """Train observer on temporally shuffled data."""
    h5_path = h5_path or str(config.TRAJECTORIES_PATH)
    save_path = save_path or str(config.CHECKPOINTS_DIR / 'shuffled_observer.pt')
    device = device or config.DEVICE
    hidden_size = hidden_size or config.OBSERVER_DEFAULT_SIZE

    print("\n" + "=" * 60)
    print("Training Shuffled Observer (Control)")
    print("=" * 60)

    base_train = TrajectoryDataset(h5_path, split='train')
    base_val = TrajectoryDataset(h5_path, split='val', precompute_stats=False)
    base_val.set_normalization_stats(base_train.neuron_means, base_train.neuron_stds)

    shuffled_train = ShuffledTrajectoryDataset(base_train)
    shuffled_val = ShuffledTrajectoryDataset(base_val)

    train_loader = DataLoader(shuffled_train, batch_size=config.BATCH_SIZE,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(shuffled_val, batch_size=config.BATCH_SIZE,
                            shuffle=False, pin_memory=True)

    model = LiquidObserver(hidden_size=hidden_size)
    trainer = Trainer(model=model, device=device)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=save_path,
    )

    return {'history': history, 'model': trainer.model}


def load_model(checkpoint_path, device=None):
    """Load a saved model from checkpoint."""
    device = device or config.DEVICE
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint['config']

    if model_config.get('type') == 'linear_baseline':
        model = LinearBaseline(hidden_size=model_config['hidden_size'])
    elif model_config.get('type') == 'liquid_observer':
        model = LiquidObserver(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            output_size=model_config['output_size'],
        )
    else:
        # Fallback: try as observer
        model = LiquidObserver(
            hidden_size=model_config.get('hidden_size', config.OBSERVER_DEFAULT_SIZE),
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model

"""
CfC Observer Model

A Closed-form Continuous-time observer that watches executor hidden states
and predicts the next hidden state. The observer exists in the same
continuous-time domain as the executor.
"""

import torch
import torch.nn as nn

from ncps.torch import CfC
from ncps.wirings import AutoNCP

from . import config


class LiquidObserver(nn.Module):
    """CfC observer that predicts the executor's next hidden state."""

    def __init__(
        self,
        input_size=None,
        hidden_size=None,
        output_size=None,
    ):
        super().__init__()
        input_size = input_size or config.EXECUTOR_HIDDEN_SIZE
        hidden_size = hidden_size or config.OBSERVER_DEFAULT_SIZE
        output_size = output_size or config.OBSERVER_OUTPUT_SIZE

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # CfC output neurons (motor neurons in NCP)
        # AutoNCP requires output_size < units - 2
        self.cfc_output = min(output_size, hidden_size - 3)

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # NCP-structured CfC
        self.wiring = AutoNCP(hidden_size, self.cfc_output)
        self.cfc = CfC(
            hidden_size,
            self.wiring,
            batch_first=True,
            return_sequences=True,
        )

        # Prediction head: map from CfC output to executor hidden size
        self.prediction_head = nn.Sequential(
            nn.Linear(self.cfc_output, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

        self.config = {
            'type': 'liquid_observer',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
        }

    def forward(self, executor_hidden, h0=None):
        """Forward pass over executor hidden states."""
        projected = self.input_proj(executor_hidden)

        if h0 is not None:
            cfc_out, _ = self.cfc(projected, h0)
        else:
            cfc_out, _ = self.cfc(projected)

        return self.prediction_head(cfc_out)

    def count_parameters(self):
        """Total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class LinearBaseline(nn.Module):
    """Linear baseline: predict the next hidden state with one linear map."""

    def __init__(self, hidden_size=None):
        super().__init__()
        hidden_size = hidden_size or config.EXECUTOR_HIDDEN_SIZE
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)

        self.config = {
            'type': 'linear_baseline',
            'hidden_size': hidden_size,
        }

    def forward(self, executor_hidden, h0=None):
        return self.linear(executor_hidden)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

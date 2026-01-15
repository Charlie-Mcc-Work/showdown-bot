"""Proximal Policy Optimization (PPO) algorithm implementation."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from showdown_bot.config import training_config, TrainingConfig
from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.training.buffer import RolloutBuffer


@dataclass
class PPOStats:
    """Statistics from a PPO update."""

    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float


class PPO:
    """Proximal Policy Optimization algorithm.

    Implements the clipped surrogate objective with value function clipping
    and entropy bonus for exploration.
    """

    def __init__(
        self,
        model: PolicyValueNetwork,
        learning_rate: float = 3e-4,
        final_learning_rate: float = 3e-5,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        target_kl: float | None = None,
        device: torch.device | None = None,
    ):
        """Initialize PPO.

        Args:
            model: Policy and value network
            learning_rate: Initial learning rate for optimizer
            final_learning_rate: Final learning rate (for linear decay)
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of epochs per update
            target_kl: Optional early stopping based on KL divergence
            device: Device to train on
        """
        self.model = model
        self.initial_lr = learning_rate
        self.final_lr = final_learning_rate
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.target_kl = target_kl
        self.device = device or torch.device("cpu")

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

        # Mixed precision training - disabled for now, overhead exceeds benefit for small models
        self.use_amp = False  # self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

    def update(self, buffer: RolloutBuffer, batch_size: int) -> PPOStats:
        """Perform a PPO update using data from the buffer.

        Args:
            buffer: Rollout buffer containing experiences
            batch_size: Minibatch size for updates

        Returns:
            Statistics from the update
        """
        # Handle empty buffer (e.g., during shutdown)
        if buffer.size == 0:
            return PPOStats(
                policy_loss=0.0,
                value_loss=0.0,
                entropy_loss=0.0,
                total_loss=0.0,
                approx_kl=0.0,
                clip_fraction=0.0,
                explained_variance=0.0,
            )

        # Track statistics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        for epoch in range(self.num_epochs):
            batches = buffer.get_batches(batch_size, shuffle=True)

            for batch in batches:
                # Mixed precision forward pass and loss computation
                with autocast("cuda", enabled=self.use_amp):
                    # Get current policy outputs
                    _, log_probs, entropy, values = self.model.get_action_and_value(
                        player_pokemon=batch["player_pokemon"],
                        opponent_pokemon=batch["opponent_pokemon"],
                        player_active_idx=batch["player_active_idx"],
                        opponent_active_idx=batch["opponent_active_idx"],
                        field_state=batch["field_state"],
                        action_mask=batch["action_mask"],
                        action=batch["actions"],
                    )

                    # Compute ratio for PPO
                    log_ratio = log_probs - batch["old_log_probs"]
                    ratio = torch.exp(log_ratio)

                    # Clipped surrogate objective
                    advantages = batch["advantages"]
                    surrogate1 = ratio * advantages
                    surrogate2 = torch.clamp(
                        ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                    ) * advantages
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()

                    # Value loss with clipping
                    values_clipped = batch["old_values"] + torch.clamp(
                        values - batch["old_values"],
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    value_loss1 = F.mse_loss(values, batch["returns"])
                    value_loss2 = F.mse_loss(values_clipped, batch["returns"])
                    value_loss = torch.min(value_loss1, value_loss2)

                    # Entropy bonus (for exploration)
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss
                    )

                # Approximate KL divergence for early stopping (outside autocast)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                # Optimize with gradient scaling for mixed precision
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Track statistics
                with torch.no_grad():
                    clip_fraction = (
                        (torch.abs(ratio - 1) > self.clip_epsilon).float().mean().item()
                    )

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                num_updates += 1

            # Early stopping based on KL divergence
            if self.target_kl is not None and total_approx_kl / num_updates > self.target_kl:
                break

        # Compute explained variance
        with torch.no_grad():
            # Get all returns and values for explained variance
            batches = buffer.get_batches(buffer.size, shuffle=False)
            all_returns = batches[0]["returns"].cpu().numpy()
            all_old_values = batches[0]["old_values"].cpu().numpy()

            var_returns = all_returns.var()
            if var_returns > 0:
                explained_var = 1 - (all_returns - all_old_values).var() / var_returns
            else:
                explained_var = 0.0

        return PPOStats(
            policy_loss=total_policy_loss / num_updates,
            value_loss=total_value_loss / num_updates,
            entropy_loss=total_entropy_loss / num_updates,
            total_loss=total_loss / num_updates,
            approx_kl=total_approx_kl / num_updates,
            clip_fraction=total_clip_fraction / num_updates,
            explained_variance=float(explained_var),
        )

    def update_learning_rate(self, progress: float) -> float:
        """Update learning rate based on training progress (linear decay).

        Args:
            progress: Training progress from 0.0 to 1.0

        Returns:
            The new learning rate
        """
        progress = max(0.0, min(1.0, progress))
        new_lr = self.initial_lr + progress * (self.final_lr - self.initial_lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        return new_lr

    @classmethod
    def from_config(
        cls,
        model: PolicyValueNetwork,
        device: torch.device | None = None,
        config: TrainingConfig | None = None,
    ) -> "PPO":
        """Create PPO from config.

        Args:
            model: Policy and value network
            device: Device to train on
            config: Training config (uses global config if not provided)
        """
        cfg = config or training_config
        return cls(
            model=model,
            learning_rate=cfg.learning_rate,
            final_learning_rate=cfg.final_learning_rate,
            clip_epsilon=cfg.clip_epsilon,
            value_coef=cfg.value_coef,
            entropy_coef=cfg.entropy_coef,
            max_grad_norm=cfg.max_grad_norm,
            num_epochs=cfg.num_epochs,
            device=device,
        )

    def save(self, path: str) -> None:
        """Save model, optimizer, and scaler state."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model, optimizer, and scaler state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

"""PPO trainer for the OU player network.

This trainer handles:
1. Collecting experience from battles
2. Computing advantages using GAE
3. Updating the policy using PPO
4. Managing self-play opponents
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from showdown_bot.ou.training.config import OUTrainingConfig
from showdown_bot.ou.training.buffer import OUExperienceBuffer, OUTransition
from showdown_bot.ou.player.network import OUPlayerNetwork, OUActorCritic
from showdown_bot.ou.player.state_encoder import OUStateEncoder, OUEncodedState
from showdown_bot.ou.shared.data_loader import Team

logger = logging.getLogger(__name__)


class OUPlayerTrainer:
    """PPO trainer for the OU battle player.

    Uses the same PPO algorithm as the random battles trainer,
    but with OU-specific state encoding and team handling.
    """

    def __init__(
        self,
        network: OUPlayerNetwork,
        encoder: OUStateEncoder,
        config: OUTrainingConfig,
        device: torch.device | None = None,
    ):
        """Initialize the trainer.

        Args:
            network: The player network to train
            encoder: State encoder for battles
            config: Training configuration
            device: Device for training
        """
        self.network = network
        self.encoder = encoder
        self.config = config
        self.device = device or torch.device(config.device)

        # Move network to device
        self.network = self.network.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.player_lr,
        )

        # Experience buffer
        self.buffer = OUExperienceBuffer(
            gamma=config.player_gamma,
            gae_lambda=config.player_gae_lambda,
        )

        # Logging
        self.writer: SummaryWriter | None = None
        self.global_step = 0
        self.episode_count = 0

        # Training statistics
        self.total_wins = 0
        self.total_losses = 0
        self.skill_rating = 1000.0

    def setup_logging(self, log_dir: str | Path) -> None:
        """Set up TensorBoard logging.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def add_transition(
        self,
        state: OUEncodedState,
        action: int,
        reward: float,
        next_state: OUEncodedState | None,
        done: bool,
        log_prob: float,
        value: float,
        team: Team | None = None,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if terminal)
            done: Whether episode ended
            log_prob: Log probability of action
            value: Value estimate
            team: Team being used
        """
        transition = OUTransition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value,
            team=team,
        )
        self.buffer.add(transition)

        if done:
            self.episode_count += 1
            if reward > 0:
                self.total_wins += 1
            else:
                self.total_losses += 1

    def update(self) -> dict[str, float]:
        """Perform a PPO update.

        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) == 0:
            return {}

        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages()

        # PPO update
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }

        for epoch in range(self.config.player_num_epochs):
            for transitions, returns, advantages in self.buffer.iterate_batches(
                self.config.player_batch_size
            ):
                batch_metrics = self._update_batch(transitions, returns, advantages)

                for key in metrics:
                    metrics[key] += batch_metrics.get(key, 0.0)

        # Average metrics
        num_batches = self.config.player_num_epochs * (
            len(self.buffer) // self.config.player_batch_size + 1
        )
        for key in metrics:
            metrics[key] /= max(num_batches, 1)

        # Log metrics
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"train/{key}", value, self.global_step)

            win_rate = self.total_wins / max(self.total_wins + self.total_losses, 1)
            self.writer.add_scalar("train/win_rate", win_rate, self.global_step)
            self.writer.add_scalar("train/skill_rating", self.skill_rating, self.global_step)

        self.global_step += 1

        # Clear buffer for next rollout
        self.buffer.clear()

        return metrics

    def _update_batch(
        self,
        transitions: list[OUTransition],
        returns: list[float],
        advantages: list[float],
    ) -> dict[str, float]:
        """Update on a single batch.

        Args:
            transitions: Batch of transitions
            returns: Computed returns
            advantages: Computed advantages

        Returns:
            Batch metrics
        """
        # Prepare batch tensors
        old_log_probs = torch.tensor(
            [t.log_prob for t in transitions],
            device=self.device,
        )
        old_values = torch.tensor(
            [t.value for t in transitions],
            device=self.device,
        )
        actions = torch.tensor(
            [t.action for t in transitions],
            device=self.device,
            dtype=torch.long,
        )
        returns_tensor = torch.tensor(returns, device=self.device)
        advantages_tensor = torch.tensor(advantages, device=self.device)

        # Forward pass for each state
        log_probs = []
        values = []
        entropies = []

        for trans in transitions:
            state = trans.state.to_device(self.device)
            output = self.network(state)

            # Get log prob of taken action
            policy = output["policy"]
            if policy.dim() == 1:
                policy = policy.unsqueeze(0)

            action_idx = trans.action
            log_prob = torch.log(policy[0, action_idx] + 1e-8)
            log_probs.append(log_prob)

            values.append(output["value"].squeeze())

            # Entropy
            entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=-1)
            entropies.append(entropy.squeeze())

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        entropies = torch.stack(entropies)

        # PPO loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(
            ratio,
            1 - self.config.player_clip_epsilon,
            1 + self.config.player_clip_epsilon,
        ) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns_tensor)

        # Entropy bonus
        entropy_loss = -entropies.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.config.player_value_coef * value_loss
            + self.config.player_entropy_coef * entropy_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config.player_max_grad_norm,
        )

        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean().item()
            clip_fraction = (
                (torch.abs(ratio - 1) > self.config.player_clip_epsilon)
                .float()
                .mean()
                .item()
            )

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropies.mean().item(),
            "total_loss": total_loss.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def update_skill_rating(self, won: bool, opponent_rating: float) -> None:
        """Update skill rating after a game.

        Args:
            won: Whether we won
            opponent_rating: Opponent's skill rating
        """
        expected = 1 / (1 + 10 ** ((opponent_rating - self.skill_rating) / 400))
        actual = 1.0 if won else 0.0
        self.skill_rating += self.config.skill_k_factor * (actual - expected)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save a training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "skill_rating": self.skill_rating,
            "config": self.config.to_dict(),
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.episode_count = checkpoint.get("episode_count", 0)
        self.total_wins = checkpoint.get("total_wins", 0)
        self.total_losses = checkpoint.get("total_losses", 0)
        self.skill_rating = checkpoint.get("skill_rating", 1000.0)

        logger.info(f"Loaded checkpoint from {path}")

    def get_action(
        self,
        state: OUEncodedState,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Get an action from the policy.

        Args:
            state: Current state
            deterministic: Whether to use greedy action

        Returns:
            (action, log_prob, value)
        """
        state = state.to_device(self.device)

        with torch.no_grad():
            output = self.network(state)
            policy = output["policy"]
            value = output["value"]

            if policy.dim() == 1:
                policy = policy.unsqueeze(0)

            if deterministic:
                action = policy.argmax(dim=-1).item()
            else:
                action = torch.multinomial(policy, 1).item()

            log_prob = torch.log(policy[0, action] + 1e-8).item()
            value = value.item() if value.dim() == 0 else value.squeeze().item()

        return action, log_prob, value

    @property
    def win_rate(self) -> float:
        """Current win rate."""
        total = self.total_wins + self.total_losses
        if total == 0:
            return 0.5
        return self.total_wins / total

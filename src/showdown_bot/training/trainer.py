"""Training loop for the Pokemon Showdown RL bot."""

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env.player import Player, RandomPlayer
from poke_env.environment import AbstractBattle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from showdown_bot.config import training_config
from showdown_bot.environment.battle_env import calculate_reward
from showdown_bot.environment.state_encoder import StateEncoder
from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.training.buffer import RolloutBuffer
from showdown_bot.training.ppo import PPO, PPOStats


class TrainablePlayer(Player):
    """A player that collects experiences for training."""

    def __init__(
        self,
        model: PolicyValueNetwork,
        state_encoder: StateEncoder,
        device: torch.device,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.state_encoder = state_encoder
        self.device = device

        # Experience storage for current battle
        self.current_experiences: list[dict[str, Any]] = []
        self.prev_hp_fraction: float | None = None
        self.prev_opp_hp_fraction: float | None = None

        # Statistics
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def choose_move(self, battle: AbstractBattle) -> str:
        """Choose a move and store the experience."""
        # Encode state
        state = self.state_encoder.encode_battle(battle)

        # Add batch dimension and move to device
        player_pokemon = state.player_pokemon.unsqueeze(0).to(self.device)
        opponent_pokemon = state.opponent_pokemon.unsqueeze(0).to(self.device)
        player_active_idx = torch.tensor([state.player_active_idx], device=self.device)
        opponent_active_idx = torch.tensor([state.opponent_active_idx], device=self.device)
        field_state = state.field_state.unsqueeze(0).to(self.device)
        action_mask = state.action_mask.unsqueeze(0).to(self.device)

        # Get action from model
        with torch.no_grad():
            self.model.eval()
            action, log_prob, _, value = self.model.get_action_and_value(
                player_pokemon,
                opponent_pokemon,
                player_active_idx,
                opponent_active_idx,
                field_state,
                action_mask,
            )

        action_idx = action.item()
        log_prob_val = log_prob.item()
        value_val = value.item()

        # Calculate reward
        reward = calculate_reward(
            battle,
            self.prev_hp_fraction,
            self.prev_opp_hp_fraction,
        )

        # Update HP tracking
        self.prev_hp_fraction = sum(
            p.current_hp_fraction for p in battle.team.values() if not p.fainted
        ) / 6
        self.prev_opp_hp_fraction = sum(
            p.current_hp_fraction for p in battle.opponent_team.values() if not p.fainted
        ) / 6

        # Store experience
        self.current_experiences.append({
            "player_pokemon": state.player_pokemon.numpy(),
            "opponent_pokemon": state.opponent_pokemon.numpy(),
            "player_active_idx": state.player_active_idx,
            "opponent_active_idx": state.opponent_active_idx,
            "field_state": state.field_state.numpy(),
            "action_mask": state.action_mask.numpy(),
            "action": action_idx,
            "log_prob": log_prob_val,
            "value": value_val,
            "reward": reward,
            "done": False,
        })

        # Convert action to battle order
        order = self.state_encoder.action_to_battle_order(action_idx, battle)
        if order:
            return order

        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        """Called when a battle finishes."""
        # Calculate final reward
        if battle.won:
            final_reward = 1.0
        elif battle.lost:
            final_reward = -1.0
        else:
            final_reward = 0.0

        # Update last experience with terminal reward
        if self.current_experiences:
            self.current_experiences[-1]["reward"] = final_reward
            self.current_experiences[-1]["done"] = True

            # Track episode stats
            total_reward = sum(exp["reward"] for exp in self.current_experiences)
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(len(self.current_experiences))

        # Reset for next battle
        self.prev_hp_fraction = None
        self.prev_opp_hp_fraction = None

    def get_experiences(self) -> list[dict[str, Any]]:
        """Get collected experiences and clear the buffer."""
        experiences = self.current_experiences.copy()
        self.current_experiences = []
        return experiences

    def reset_stats(self) -> None:
        """Reset episode statistics."""
        self.episode_rewards = []
        self.episode_lengths = []


@dataclass
class TrainingStats:
    """Statistics from training."""

    total_timesteps: int = 0
    total_episodes: int = 0
    total_updates: int = 0
    best_win_rate: float = 0.0
    training_start_time: float = field(default_factory=time.time)


class Trainer:
    """Main trainer class that orchestrates the training loop."""

    def __init__(
        self,
        model: PolicyValueNetwork,
        device: torch.device,
        save_dir: str = "data/checkpoints",
        log_dir: str = "runs",
    ):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)

        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.state_encoder = StateEncoder(device=device)
        self.ppo = PPO.from_config(model, device=device)
        self.buffer = RolloutBuffer(
            buffer_size=training_config.rollout_steps,
            num_envs=1,  # Single env for now
            gamma=training_config.gamma,
            gae_lambda=training_config.gae_lambda,
            device=device,
        )

        # TensorBoard writer
        run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(self.log_dir / run_name))

        # Stats
        self.stats = TrainingStats()

    async def collect_rollout(
        self,
        player: TrainablePlayer,
        opponent: Player,
        num_steps: int,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Collect experiences by running battles.

        Args:
            player: The training player
            opponent: The opponent to play against
            num_steps: Target number of steps to collect

        Returns:
            Tuple of (experiences, stats dict)
        """
        all_experiences: list[dict[str, Any]] = []
        battles_played = 0
        wins = 0

        while len(all_experiences) < num_steps:
            # Play a battle
            await player.battle_against(opponent, n_battles=1)
            battles_played += 1

            # Get experiences from the battle
            experiences = player.get_experiences()
            all_experiences.extend(experiences)

            # Track wins
            if player.n_won_battles > wins:
                wins = player.n_won_battles

        # Compute stats
        stats = {
            "battles": battles_played,
            "wins": wins,
            "win_rate": wins / battles_played if battles_played > 0 else 0.0,
            "avg_reward": np.mean(player.episode_rewards) if player.episode_rewards else 0.0,
            "avg_length": np.mean(player.episode_lengths) if player.episode_lengths else 0.0,
        }

        return all_experiences, stats

    def add_experiences_to_buffer(self, experiences: list[dict[str, Any]]) -> None:
        """Add collected experiences to the rollout buffer."""
        for exp in experiences:
            if self.buffer.ptr >= self.buffer.buffer_size:
                break

            self.buffer.add(
                player_pokemon=exp["player_pokemon"][np.newaxis, ...],
                opponent_pokemon=exp["opponent_pokemon"][np.newaxis, ...],
                player_active_idx=np.array([exp["player_active_idx"]]),
                opponent_active_idx=np.array([exp["opponent_active_idx"]]),
                field_state=exp["field_state"][np.newaxis, ...],
                action_mask=exp["action_mask"][np.newaxis, ...],
                action=np.array([exp["action"]]),
                log_prob=np.array([exp["log_prob"]]),
                reward=np.array([exp["reward"]]),
                done=np.array([exp["done"]]),
                value=np.array([exp["value"]]),
            )

    async def train(
        self,
        total_timesteps: int | None = None,
        eval_interval: int | None = None,
        save_interval: int | None = None,
    ) -> None:
        """Main training loop.

        Args:
            total_timesteps: Total timesteps to train for
            eval_interval: Steps between evaluations
            save_interval: Steps between saving checkpoints
        """
        total_timesteps = total_timesteps or training_config.total_timesteps
        eval_interval = eval_interval or training_config.eval_interval
        save_interval = save_interval or training_config.checkpoint_interval

        print("=" * 60)
        print("Pokemon Showdown RL Bot - Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Rollout steps: {training_config.rollout_steps}")
        print(f"Batch size: {training_config.batch_size}")
        print(f"Learning rate: {training_config.learning_rate}")
        print("=" * 60)

        # Create players
        player = TrainablePlayer(
            model=self.model,
            state_encoder=self.state_encoder,
            device=self.device,
            battle_format=training_config.battle_format,
            max_concurrent_battles=1,
        )

        opponent = RandomPlayer(
            battle_format=training_config.battle_format,
            max_concurrent_battles=1,
        )

        # Training loop
        pbar = tqdm(total=total_timesteps, desc="Training")

        while self.stats.total_timesteps < total_timesteps:
            # Collect rollout
            self.buffer.reset()
            player.reset_stats()

            experiences, rollout_stats = await self.collect_rollout(
                player, opponent, training_config.rollout_steps
            )

            # Add to buffer
            self.add_experiences_to_buffer(experiences)

            # Compute advantages (using 0 as terminal value since battles end)
            last_value = np.zeros((1,), dtype=np.float32)
            last_done = np.ones((1,), dtype=np.float32)
            self.buffer.compute_advantages(last_value, last_done)

            # PPO update
            self.model.train()
            ppo_stats = self.ppo.update(self.buffer, training_config.batch_size)

            # Update stats
            self.stats.total_timesteps += len(experiences)
            self.stats.total_episodes += rollout_stats["battles"]
            self.stats.total_updates += 1

            # Log to TensorBoard
            self._log_training(rollout_stats, ppo_stats)

            # Update progress bar
            pbar.update(len(experiences))
            pbar.set_postfix({
                "win_rate": f"{rollout_stats['win_rate']:.1%}",
                "reward": f"{rollout_stats['avg_reward']:.2f}",
                "loss": f"{ppo_stats.total_loss:.3f}",
            })

            # Save checkpoint
            if self.stats.total_timesteps % save_interval < len(experiences):
                self._save_checkpoint()

            # Track best win rate
            if rollout_stats["win_rate"] > self.stats.best_win_rate:
                self.stats.best_win_rate = rollout_stats["win_rate"]
                self._save_checkpoint("best_model.pt")

        pbar.close()
        self.writer.close()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Total timesteps: {self.stats.total_timesteps:,}")
        print(f"Total episodes: {self.stats.total_episodes:,}")
        print(f"Best win rate: {self.stats.best_win_rate:.1%}")
        print("=" * 60)

    def _log_training(self, rollout_stats: dict[str, float], ppo_stats: PPOStats) -> None:
        """Log training metrics to TensorBoard."""
        step = self.stats.total_timesteps

        # Rollout stats
        self.writer.add_scalar("rollout/win_rate", rollout_stats["win_rate"], step)
        self.writer.add_scalar("rollout/avg_reward", rollout_stats["avg_reward"], step)
        self.writer.add_scalar("rollout/avg_length", rollout_stats["avg_length"], step)
        self.writer.add_scalar("rollout/battles", rollout_stats["battles"], step)

        # PPO stats
        self.writer.add_scalar("train/policy_loss", ppo_stats.policy_loss, step)
        self.writer.add_scalar("train/value_loss", ppo_stats.value_loss, step)
        self.writer.add_scalar("train/entropy_loss", ppo_stats.entropy_loss, step)
        self.writer.add_scalar("train/total_loss", ppo_stats.total_loss, step)
        self.writer.add_scalar("train/approx_kl", ppo_stats.approx_kl, step)
        self.writer.add_scalar("train/clip_fraction", ppo_stats.clip_fraction, step)
        self.writer.add_scalar("train/explained_variance", ppo_stats.explained_variance, step)

        # General stats
        self.writer.add_scalar("stats/total_episodes", self.stats.total_episodes, step)
        self.writer.add_scalar("stats/total_updates", self.stats.total_updates, step)

    def _save_checkpoint(self, filename: str | None = None) -> None:
        """Save a training checkpoint."""
        if filename is None:
            filename = f"checkpoint_{self.stats.total_timesteps}.pt"

        path = self.save_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.ppo.optimizer.state_dict(),
                "stats": {
                    "total_timesteps": self.stats.total_timesteps,
                    "total_episodes": self.stats.total_episodes,
                    "total_updates": self.stats.total_updates,
                    "best_win_rate": self.stats.best_win_rate,
                },
            },
            path,
        )
        print(f"\nSaved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        stats = checkpoint.get("stats", {})
        self.stats.total_timesteps = stats.get("total_timesteps", 0)
        self.stats.total_episodes = stats.get("total_episodes", 0)
        self.stats.total_updates = stats.get("total_updates", 0)
        self.stats.best_win_rate = stats.get("best_win_rate", 0.0)

        print(f"Loaded checkpoint: {path}")
        print(f"  Timesteps: {self.stats.total_timesteps:,}")
        print(f"  Episodes: {self.stats.total_episodes:,}")

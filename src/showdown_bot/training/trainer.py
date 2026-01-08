"""Training loop for the Pokemon Showdown RL bot."""

import asyncio
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import BattleOrder
from poke_env.battle import AbstractBattle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

from showdown_bot.config import training_config, TrainingConfig
from showdown_bot.environment.battle_env import calculate_reward
from showdown_bot.environment.state_encoder import StateEncoder
from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.training.buffer import RolloutBuffer
from showdown_bot.training.ppo import PPO, PPOStats
from showdown_bot.training.self_play import SelfPlayManager


# Maximum retries for websocket connection errors
MAX_ROLLOUT_RETRIES = 3
RETRY_DELAY_SECONDS = 5.0


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

        # Statistics - track per rollout, not cumulative
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._rollout_wins: int = 0
        self._rollout_battles: int = 0

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
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
            self._rollout_wins += 1
        elif battle.lost:
            final_reward = -1.0
        else:
            final_reward = 0.0

        self._rollout_battles += 1

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
        """Reset episode statistics for new rollout."""
        self.episode_rewards = []
        self.episode_lengths = []
        self._rollout_wins = 0
        self._rollout_battles = 0

    def get_rollout_stats(self) -> dict[str, float]:
        """Get stats for the current rollout."""
        return {
            "battles": self._rollout_battles,
            "wins": self._rollout_wins,
            "win_rate": self._rollout_wins / self._rollout_battles if self._rollout_battles > 0 else 0.0,
            "avg_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "avg_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
        }


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
        self_play_dir: str = "data/opponents",
        use_self_play: bool = True,
        config: TrainingConfig | None = None,
        num_envs: int | None = None,
    ):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.use_self_play = use_self_play
        self.config = config or training_config  # Use provided config or default
        self.num_envs = num_envs or self.config.num_envs

        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.state_encoder = StateEncoder(device=device)
        self.ppo = PPO.from_config(model, device=device, config=self.config)
        self.buffer = RolloutBuffer(
            buffer_size=self.config.rollout_steps,
            num_envs=self.num_envs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            device=device,
        )

        # Self-play manager
        self.self_play_manager = SelfPlayManager(
            pool_dir=self_play_dir,
            max_pool_size=20,
            self_play_ratio=self.config.self_play_ratio,
            checkpoint_interval=self.config.checkpoint_interval,
            sampling_strategy="elo_matched",
            device=device,
        ) if use_self_play else None

        # TensorBoard writer - will be set on train start
        self.writer: SummaryWriter | None = None
        self._run_name: str | None = None

        # Stats
        self.stats = TrainingStats()
        self._best_win_rate: float = 0.0  # Track for tuning

        # Graceful shutdown
        self._shutdown_requested = False

        # Track active players for cleanup
        self._active_players: list[TrainablePlayer] = []
        self._active_opponents: list[Player] = []

    def _create_players(self) -> list[TrainablePlayer]:
        """Create fresh player instances for training.

        Returns a list of TrainablePlayer instances, one per environment.
        """
        players = [
            TrainablePlayer(
                model=self.model,
                state_encoder=self.state_encoder,
                device=self.device,
                battle_format=self.config.battle_format,
                max_concurrent_battles=1,
            )
            for _ in range(self.num_envs)
        ]
        return players

    async def _cleanup_players(self, players: list[Player]) -> None:
        """Clean up player websocket connections gracefully."""
        for player in players:
            try:
                # poke-env players may have a stop_listening method
                if hasattr(player, 'stop_listening'):
                    await player.stop_listening()
            except Exception:
                pass  # Ignore cleanup errors

    def _setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            if self._shutdown_requested:
                print("\nForce quitting...")
                sys.exit(1)
            print("\n\nShutdown requested. Finishing current rollout and saving checkpoint...")
            print("Press Ctrl+C again to force quit.")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def collect_rollout_single(
        self,
        player: TrainablePlayer,
        opponent: Player,
        num_steps: int,
    ) -> list[dict[str, Any]]:
        """Collect experiences from a single environment.

        Args:
            player: The training player
            opponent: The opponent to play against
            num_steps: Target number of steps to collect

        Returns:
            List of experiences
        """
        all_experiences: list[dict[str, Any]] = []

        while len(all_experiences) < num_steps:
            # Play a battle
            await player.battle_against(opponent, n_battles=1)

            # Get experiences from the battle
            experiences = player.get_experiences()
            all_experiences.extend(experiences)

        return all_experiences

    async def collect_rollout_parallel(
        self,
        players: list[TrainablePlayer],
        opponents: list[Player],
        num_steps: int,
    ) -> list[list[dict[str, Any]]]:
        """Collect experiences from multiple environments in parallel.

        Args:
            players: List of training players
            opponents: List of opponents
            num_steps: Target steps per environment

        Returns:
            List of experience lists (one per environment)
        """
        # Run all environments in parallel
        tasks = [
            self.collect_rollout_single(player, opponent, num_steps)
            for player, opponent in zip(players, opponents)
        ]
        return await asyncio.gather(*tasks)

    async def collect_rollout_with_retry(
        self,
        players: list[TrainablePlayer],
        opponents: list[Player],
        num_steps: int,
    ) -> tuple[list[list[dict[str, Any]]], list[TrainablePlayer], list[Player]]:
        """Collect rollouts with automatic retry on websocket errors.

        Args:
            players: List of training players
            opponents: List of opponents
            num_steps: Target steps per environment

        Returns:
            Tuple of (experiences, players, opponents) - players/opponents may be
            recreated if connection errors occurred
        """
        for attempt in range(MAX_ROLLOUT_RETRIES):
            try:
                experiences = await self.collect_rollout_parallel(
                    players, opponents, num_steps
                )
                return experiences, players, opponents

            except (ConnectionClosed, ConnectionClosedError, OSError) as e:
                # Websocket connection error - retry with fresh players
                error_name = type(e).__name__
                tqdm.write(
                    f"\n  Connection error ({error_name}): {e}"
                )

                if attempt < MAX_ROLLOUT_RETRIES - 1:
                    tqdm.write(
                        f"  Retrying ({attempt + 1}/{MAX_ROLLOUT_RETRIES}) "
                        f"in {RETRY_DELAY_SECONDS}s..."
                    )

                    # Clean up old players
                    await self._cleanup_players(players)
                    await self._cleanup_players(opponents)

                    # Wait before retrying
                    await asyncio.sleep(RETRY_DELAY_SECONDS)

                    # Create fresh players
                    players = self._create_players()

                    # Create fresh opponents
                    opponents = []
                    for _ in range(self.num_envs):
                        if self.self_play_manager:
                            opponent, _ = self.self_play_manager.get_opponent(
                                self.config.battle_format
                            )
                            opponents.append(opponent)
                        else:
                            opponents.append(RandomPlayer(
                                battle_format=self.config.battle_format,
                                max_concurrent_battles=1,
                            ))

                    # Reset stats on new players
                    for player in players:
                        player.reset_stats()
                else:
                    tqdm.write(
                        f"  Max retries ({MAX_ROLLOUT_RETRIES}) exceeded. "
                        "Saving checkpoint and exiting..."
                    )
                    raise

            except asyncio.CancelledError:
                # Don't retry on cancellation
                raise

        # Should not reach here, but just in case
        raise RuntimeError("Rollout collection failed after all retries")

    def add_experiences_to_buffer_parallel(
        self, all_env_experiences: list[list[dict[str, Any]]]
    ) -> int:
        """Add experiences from multiple environments to the buffer.

        Interleaves experiences from different environments for better mixing.

        Args:
            all_env_experiences: List of experience lists (one per environment)

        Returns:
            Total number of experiences added
        """
        num_envs = len(all_env_experiences)
        total_added = 0

        # Find the minimum length to ensure we don't go out of bounds
        min_len = min(len(exp) for exp in all_env_experiences)
        max_steps = min(min_len, self.buffer.buffer_size)

        for step in range(max_steps):
            if self.buffer.ptr >= self.buffer.buffer_size:
                break

            # Gather data from all environments for this step
            player_pokemon = np.stack([
                all_env_experiences[env][step]["player_pokemon"]
                for env in range(num_envs)
            ])
            opponent_pokemon = np.stack([
                all_env_experiences[env][step]["opponent_pokemon"]
                for env in range(num_envs)
            ])
            player_active_idx = np.array([
                all_env_experiences[env][step]["player_active_idx"]
                for env in range(num_envs)
            ])
            opponent_active_idx = np.array([
                all_env_experiences[env][step]["opponent_active_idx"]
                for env in range(num_envs)
            ])
            field_state = np.stack([
                all_env_experiences[env][step]["field_state"]
                for env in range(num_envs)
            ])
            action_mask = np.stack([
                all_env_experiences[env][step]["action_mask"]
                for env in range(num_envs)
            ])
            actions = np.array([
                all_env_experiences[env][step]["action"]
                for env in range(num_envs)
            ])
            log_probs = np.array([
                all_env_experiences[env][step]["log_prob"]
                for env in range(num_envs)
            ])
            rewards = np.array([
                all_env_experiences[env][step]["reward"]
                for env in range(num_envs)
            ])
            dones = np.array([
                all_env_experiences[env][step]["done"]
                for env in range(num_envs)
            ])
            values = np.array([
                all_env_experiences[env][step]["value"]
                for env in range(num_envs)
            ])

            self.buffer.add(
                player_pokemon=player_pokemon,
                opponent_pokemon=opponent_pokemon,
                player_active_idx=player_active_idx,
                opponent_active_idx=opponent_active_idx,
                field_state=field_state,
                action_mask=action_mask,
                action=actions,
                log_prob=log_probs,
                reward=rewards,
                done=dones,
                value=values,
            )
            total_added += num_envs

        return total_added

    def add_experiences_to_buffer(self, experiences: list[dict[str, Any]]) -> None:
        """Add collected experiences to the rollout buffer (single env)."""
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
        self._setup_signal_handlers()

        total_timesteps = total_timesteps or self.config.total_timesteps
        eval_interval = eval_interval or self.config.eval_interval
        save_interval = save_interval or self.config.checkpoint_interval

        # Setup TensorBoard if not resuming
        if self.writer is None:
            self._run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir=str(self.log_dir / self._run_name))

        print("=" * 60)
        print("Pokemon Showdown RL Bot - Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Starting from: {self.stats.total_timesteps:,}")
        print(f"Parallel environments: {self.num_envs}")
        print(f"Rollout steps: {self.config.rollout_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Self-play: {'Enabled' if self.use_self_play else 'Disabled'}")
        if self.self_play_manager:
            print(f"Agent Elo: {self.self_play_manager.agent_elo:.0f}")
            print(f"Opponent pool size: {self.self_play_manager.opponent_pool.size}")
        print("=" * 60)
        print("Press Ctrl+C to stop and save checkpoint")
        print("=" * 60)

        # Create multiple players for parallel environments
        players = self._create_players()

        # Training loop
        pbar = tqdm(
            total=total_timesteps,
            initial=self.stats.total_timesteps,
            desc="Training",
        )

        try:
            while self.stats.total_timesteps < total_timesteps and not self._shutdown_requested:
                # Reset buffer and all player stats
                self.buffer.reset()
                for player in players:
                    player.reset_stats()

                # Create opponents for each environment
                opponents = []
                opponent_infos = []
                for _ in range(self.num_envs):
                    if self.self_play_manager:
                        opponent, opponent_info = self.self_play_manager.get_opponent(
                            self.config.battle_format
                        )
                        opponents.append(opponent)
                        opponent_infos.append(opponent_info)
                    else:
                        opponents.append(RandomPlayer(
                            battle_format=self.config.battle_format,
                            max_concurrent_battles=1,
                        ))
                        opponent_infos.append(None)

                # Collect rollouts in parallel with automatic retry on connection errors
                steps_per_env = self.config.rollout_steps // self.num_envs
                all_env_experiences, players, opponents = await self.collect_rollout_with_retry(
                    players, opponents, steps_per_env
                )

                # Aggregate rollout stats from all players
                total_wins = sum(p._rollout_wins for p in players)
                total_battles = sum(p._rollout_battles for p in players)
                all_rewards = [r for p in players for r in p.episode_rewards]
                all_lengths = [l for p in players for l in p.episode_lengths]

                rollout_stats = {
                    "battles": total_battles,
                    "wins": total_wins,
                    "win_rate": total_wins / total_battles if total_battles > 0 else 0.0,
                    "avg_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
                    "avg_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
                }

                # Update self-play stats (use aggregate win rate)
                if self.self_play_manager:
                    for info in opponent_infos:
                        if info is not None:
                            won = rollout_stats["win_rate"] > 0.5
                            self.self_play_manager.update_after_game(info, won)

                # Add to buffer
                total_added = self.add_experiences_to_buffer_parallel(all_env_experiences)

                # Compute advantages
                last_value = np.zeros((self.num_envs,), dtype=np.float32)
                last_done = np.ones((self.num_envs,), dtype=np.float32)
                self.buffer.compute_advantages(last_value, last_done)

                # PPO update
                self.model.train()
                ppo_stats = self.ppo.update(self.buffer, self.config.batch_size)

                # Update stats
                self.stats.total_timesteps += total_added
                self.stats.total_episodes += rollout_stats["battles"]
                self.stats.total_updates += 1

                # Log to TensorBoard
                self._log_training(rollout_stats, ppo_stats)

                # Update progress bar
                pbar.update(total_added)
                postfix = {
                    "win_rate": f"{rollout_stats['win_rate']:.1%}",
                    "reward": f"{rollout_stats['avg_reward']:.2f}",
                    "loss": f"{ppo_stats.total_loss:.3f}",
                }
                if self.self_play_manager:
                    postfix["elo"] = f"{self.self_play_manager.agent_elo:.0f}"
                pbar.set_postfix(postfix)

                # Add checkpoint to self-play pool
                if self.self_play_manager and self.self_play_manager.should_add_checkpoint(
                    self.stats.total_timesteps
                ):
                    self.self_play_manager.add_checkpoint(self.model, self.stats.total_timesteps)
                    print(f"\n  Added checkpoint to opponent pool (size: {self.self_play_manager.opponent_pool.size})")

                # Save checkpoint periodically
                if self.stats.total_timesteps % save_interval < total_added:
                    self._save_checkpoint()

                # Track best win rate
                if rollout_stats["win_rate"] > self.stats.best_win_rate:
                    self.stats.best_win_rate = rollout_stats["win_rate"]
                    self._best_win_rate = rollout_stats["win_rate"]  # For tuning
                    self._save_checkpoint("best_model.pt")

                # Clean up opponent connections to prevent file descriptor leak
                await self._cleanup_players(opponents)

        except Exception as e:
            print(f"\nError during training: {e}")
            print("Saving emergency checkpoint...")
            self._save_checkpoint("emergency_checkpoint.pt")
            raise
        finally:
            # Clean up player connections
            try:
                await self._cleanup_players(players)
                await self._cleanup_players(opponents)
            except Exception:
                pass  # Ignore cleanup errors

            # Always save on exit
            pbar.close()
            if self._shutdown_requested or self.stats.total_timesteps > 0:
                self._save_checkpoint("latest.pt")
                print(f"\nCheckpoint saved to: {self.save_dir / 'latest.pt'}")

            if self.writer:
                self.writer.close()

        print("\n" + "=" * 60)
        if self._shutdown_requested:
            print("Training Stopped (checkpoint saved)")
        else:
            print("Training Complete!")
        print(f"Total timesteps: {self.stats.total_timesteps:,}")
        print(f"Total episodes: {self.stats.total_episodes:,}")
        print(f"Best win rate: {self.stats.best_win_rate:.1%}")
        if self.self_play_manager:
            print(f"Final Elo: {self.self_play_manager.agent_elo:.0f}")
        print("=" * 60)
        print(f"\nTo resume training, run:")
        print(f"  python scripts/train.py --resume {self.save_dir / 'latest.pt'}")

    def _log_training(self, rollout_stats: dict[str, float], ppo_stats: PPOStats) -> None:
        """Log training metrics to TensorBoard."""
        if self.writer is None:
            return

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

        # Self-play stats
        if self.self_play_manager:
            sp_stats = self.self_play_manager.get_stats()
            self.writer.add_scalar("self_play/agent_elo", sp_stats["agent_elo"], step)
            self.writer.add_scalar("self_play/pool_size", sp_stats["pool_size"], step)
            self.writer.add_scalar("self_play/pool_avg_elo", sp_stats["pool_avg_elo"], step)
            self.writer.add_scalar("self_play/games_vs_self_play", sp_stats["games_vs_self_play"], step)
            self.writer.add_scalar("self_play/games_vs_random", sp_stats["games_vs_random"], step)

    def _save_checkpoint(self, filename: str | None = None) -> None:
        """Save a training checkpoint."""
        if filename is None:
            filename = f"checkpoint_{self.stats.total_timesteps}.pt"

        path = self.save_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.ppo.optimizer.state_dict(),
            "stats": {
                "total_timesteps": self.stats.total_timesteps,
                "total_episodes": self.stats.total_episodes,
                "total_updates": self.stats.total_updates,
                "best_win_rate": self.stats.best_win_rate,
            },
            "run_name": self._run_name,
        }

        # Save self-play state
        if self.self_play_manager:
            checkpoint["self_play"] = {
                "agent_elo": self.self_play_manager.agent_elo,
                "last_checkpoint_timestep": self.self_play_manager.last_checkpoint_timestep,
                "games_vs_self_play": self.self_play_manager.games_vs_self_play,
                "games_vs_random": self.self_play_manager.games_vs_random,
            }

        torch.save(checkpoint, path)
        tqdm.write(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        stats = checkpoint.get("stats", {})
        self.stats.total_timesteps = stats.get("total_timesteps", 0)
        self.stats.total_episodes = stats.get("total_episodes", 0)
        self.stats.total_updates = stats.get("total_updates", 0)
        self.stats.best_win_rate = stats.get("best_win_rate", 0.0)

        # Restore run name for TensorBoard continuity
        self._run_name = checkpoint.get("run_name")
        if self._run_name:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / self._run_name))

        # Restore self-play state
        if self.self_play_manager and "self_play" in checkpoint:
            sp_state = checkpoint["self_play"]
            self.self_play_manager.agent_elo = sp_state.get("agent_elo", 1000.0)
            self.self_play_manager.last_checkpoint_timestep = sp_state.get("last_checkpoint_timestep", 0)
            self.self_play_manager.games_vs_self_play = sp_state.get("games_vs_self_play", 0)
            self.self_play_manager.games_vs_random = sp_state.get("games_vs_random", 0)

        print(f"Loaded checkpoint: {path}")
        print(f"  Timesteps: {self.stats.total_timesteps:,}")
        print(f"  Episodes: {self.stats.total_episodes:,}")
        print(f"  Best win rate: {self.stats.best_win_rate:.1%}")
        if self.self_play_manager and "self_play" in checkpoint:
            print(f"  Agent Elo: {self.self_play_manager.agent_elo:.0f}")

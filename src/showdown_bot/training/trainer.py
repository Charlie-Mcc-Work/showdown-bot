"""Training loop for the Pokemon Showdown RL bot."""

import asyncio
import gc
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import autocast
from poke_env.player import Player, RandomPlayer
from poke_env.ps_client import ServerConfiguration
from poke_env.player.battle_order import BattleOrder
from poke_env.battle import AbstractBattle
from torch.utils.tensorboard import SummaryWriter
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

# Memory thresholds for automatic shutdown
MEMORY_SOFT_LIMIT_PERCENT = 80.0  # Graceful shutdown with checkpoint
MEMORY_HARD_LIMIT_PERCENT = 95.0  # Force quit


class MemoryMonitor:
    """Monitors system memory usage and triggers shutdown if thresholds exceeded."""

    def __init__(
        self,
        soft_limit_percent: float = MEMORY_SOFT_LIMIT_PERCENT,
        hard_limit_percent: float = MEMORY_HARD_LIMIT_PERCENT,
    ):
        """Initialize memory monitor.

        Args:
            soft_limit_percent: Memory % that triggers graceful shutdown (default 80%)
            hard_limit_percent: Memory % that triggers force quit (default 95%)
        """
        self.soft_limit_percent = soft_limit_percent
        self.hard_limit_percent = hard_limit_percent
        self._total_memory: int | None = None

    def get_memory_info(self) -> tuple[int, int, float]:
        """Get current memory usage from /proc/meminfo.

        Returns:
            Tuple of (used_bytes, total_bytes, percent_used)
        """
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        # Values are in kB
                        key = parts[0].rstrip(":")
                        value = int(parts[1]) * 1024  # Convert to bytes
                        meminfo[key] = value

            total = meminfo.get("MemTotal", 0)
            available = meminfo.get("MemAvailable", 0)
            used = total - available
            percent = (used / total * 100) if total > 0 else 0.0

            self._total_memory = total
            return used, total, percent

        except (OSError, ValueError, KeyError):
            # Fallback if /proc/meminfo is unavailable
            return 0, 0, 0.0

    def check_memory(self) -> tuple[str, float]:
        """Check memory and return status.

        Returns:
            Tuple of (status, percent_used) where status is one of:
            - "ok": Memory usage is acceptable
            - "soft_limit": Memory exceeds soft limit, should gracefully shutdown
            - "hard_limit": Memory exceeds hard limit, must force quit
        """
        used, total, percent = self.get_memory_info()

        if percent >= self.hard_limit_percent:
            return "hard_limit", percent
        elif percent >= self.soft_limit_percent:
            return "soft_limit", percent
        else:
            return "ok", percent

    def format_memory(self, bytes_val: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_val < 1024:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024
        return f"{bytes_val:.1f} TB"

    def get_memory_status_string(self) -> str:
        """Get a formatted string of current memory status."""
        used, total, percent = self.get_memory_info()
        return f"{self.format_memory(used)}/{self.format_memory(total)} ({percent:.1f}%)"


class TrainingDisplay:
    """Training display that updates in place using ANSI escape codes."""

    def __init__(self, total_timesteps: int, start_timesteps: int = 0):
        self.total_timesteps = total_timesteps
        self.start_timesteps = start_timesteps
        self.current_timesteps = start_timesteps
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_timesteps = start_timesteps
        self.iteration = 0

        # Smoothed speed calculation
        self._speed_history: list[float] = []
        self._speed_window = 10

        # Track if we're in the middle of an in-place update
        self._in_place_active = False

    def _format_number(self, n: int) -> str:
        """Format large numbers with K/M suffix."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h{mins}m"

    def _get_speed(self) -> float:
        """Get smoothed iterations per second."""
        if not self._speed_history:
            return 0.0
        return sum(self._speed_history) / len(self._speed_history)

    def initialize(self):
        """Print header (static, not updated in place)."""
        pass  # Header will be part of the update line

    def close(self):
        """Move to new line after in-place updates."""
        if self._in_place_active:
            print()  # Move past the in-place line
            self._in_place_active = False

    def update(
        self,
        timesteps: int,
        rollout_stats: dict | None = None,
        ppo_stats: "PPOStats | None" = None,
        skill: float | None = None,
        pool_size: int | None = None,
        memory_str: str = "",
        memory_percent: float = 0.0,
        best_skill: float = 0.0,
    ):
        """Update the display in place."""
        now = time.time()
        self.current_timesteps = timesteps
        self.iteration += 1

        # Calculate speed
        time_delta = now - self.last_update_time
        if time_delta > 0:
            step_delta = timesteps - self.last_timesteps
            instant_speed = step_delta / time_delta
            self._speed_history.append(instant_speed)
            if len(self._speed_history) > self._speed_window:
                self._speed_history.pop(0)

        self.last_update_time = now
        self.last_timesteps = timesteps

        # Calculate progress and ETA
        speed = self._get_speed()
        steps_done = timesteps - self.start_timesteps
        steps_total = self.total_timesteps - self.start_timesteps
        progress = steps_done / steps_total if steps_total > 0 else 0.0

        # ETA calculation
        if speed > 0 and progress < 1.0:
            remaining_steps = steps_total - steps_done
            eta_seconds = remaining_steps / speed
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "--"

        win_rate = rollout_stats.get("win_rate", 0.0) if rollout_stats else 0.0
        skill_val = skill if skill is not None else 1000.0

        # Build the status line
        # Format: [progress bar] 7.5M/12M (62%) | 1234/s | Win:85% | Skill:30066 | ETA:2h15m
        bar_width = 20
        filled = int(bar_width * progress)
        bar = "=" * filled + "-" * (bar_width - filled)

        status = (
            f"\r[{bar}] {self._format_number(timesteps)}/{self._format_number(self.total_timesteps)} "
            f"({progress:>5.1%}) | {speed:>5.0f}/s | Win:{win_rate:>3.0%} | "
            f"Skill:{skill_val:>5.0f} | Best:{best_skill:>5.0f} | ETA:{eta_str}"
        )

        # Pad to overwrite any previous longer line
        status = status.ljust(120)

        # Print in place (no newline)
        print(status, end="", flush=True)
        self._in_place_active = True

    def message(self, msg: str):
        """Print an important message on a new line."""
        if self._in_place_active:
            print()  # Move to new line first
            self._in_place_active = False
        print(f">> {msg}")


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

        # Mixed precision inference - disabled for now, overhead exceeds benefit for small models
        self.use_amp = False  # device.type == "cuda"

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

        # Get action from model with mixed precision inference
        # Note: model.eval() is set once per rollout in the training loop, not per action
        with torch.no_grad(), autocast("cuda", enabled=self.use_amp):
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
        server_ports: list[int] | None = None,
    ):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.use_self_play = use_self_play
        self.config = config or training_config  # Use provided config or default
        self.num_envs = num_envs or self.config.num_envs

        # Server configurations for distributing players across multiple servers
        # This allows parallel training to scale beyond a single server's capacity
        if server_ports:
            self.server_configs = [
                ServerConfiguration(
                    websocket_url=f"ws://localhost:{port}/showdown/websocket",
                    authentication_url="https://play.pokemonshowdown.com/action.php?",
                )
                for port in server_ports
            ]
        else:
            self.server_configs = [None]  # Use default server

        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.state_encoder = StateEncoder(device=device)
        self.ppo = PPO.from_config(model, device=device, config=self.config)
        # Buffer uses num_envs=1 since experiences are added one at a time
        # from the shared queue (fast envs contribute more, slow ones less)
        self.buffer = RolloutBuffer(
            buffer_size=self.config.rollout_steps,
            num_envs=1,
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
            sampling_strategy="skill_matched",
            device=device,
        ) if use_self_play else None

        # TensorBoard writer - will be set on train start
        self.writer: SummaryWriter | None = None
        self._run_name: str | None = None

        # Stats
        self.stats = TrainingStats()
        self._best_win_rate: float = 0.0  # Track for tuning

        # Load existing best model's skill to avoid overwriting better models
        self._global_best_skill = self._load_existing_best_skill()

        # Graceful shutdown
        self._shutdown_requested = False
        self._memory_shutdown = False  # Triggered by memory monitor

        # Memory monitor for automatic shutdown on high memory usage
        self.memory_monitor = MemoryMonitor()

        # Track active players for cleanup
        self._active_players: list[TrainablePlayer] = []
        self._active_opponents: list[Player] = []

    def _create_players(self) -> list[TrainablePlayer]:
        """Create fresh player instances for training.

        Returns a list of TrainablePlayer instances, one per environment.
        Players are distributed across available servers in round-robin fashion.
        """
        players = []
        for i in range(self.num_envs):
            # Distribute players across servers round-robin
            server_config = self.server_configs[i % len(self.server_configs)]
            players.append(
                TrainablePlayer(
                    model=self.model,
                    state_encoder=self.state_encoder,
                    device=self.device,
                    battle_format=self.config.battle_format,
                    max_concurrent_battles=1,
                    server_configuration=server_config,
                )
            )
        return players

    async def _cleanup_single_opponent(self, opponent: Player) -> None:
        """Clean up a single opponent after its battle finishes naturally.

        This is called from collect_rollout_single after each opponent's battles
        complete, ensuring no race conditions with pending messages.
        """
        try:
            if hasattr(opponent, 'ps_client') and opponent.ps_client is not None:
                await opponent.ps_client.stop_listening()

            # Free model memory for HistoricalPlayer opponents
            if hasattr(opponent, 'model') and opponent.model is not None:
                try:
                    opponent.model.cpu()
                    opponent.model = None
                except Exception:
                    pass
            if hasattr(opponent, 'state_encoder'):
                opponent.state_encoder = None
        except Exception:
            pass  # Ignore cleanup errors

    async def _cleanup_players(self, players: list[Player], full_cleanup: bool = False) -> None:
        """Clean up player websocket connections and optionally free GPU memory.

        Args:
            players: List of players to clean up
            full_cleanup: If True, wait for websocket drain and free model memory.
                         If False, just stop listening (faster, for most iterations).
        """
        # Stop listening on websockets
        cleanup_tasks = []
        for player in players:
            try:
                if hasattr(player, 'ps_client') and player.ps_client is not None:
                    cleanup_tasks.append(player.ps_client.stop_listening())
            except Exception:
                pass

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            if full_cleanup:
                await asyncio.sleep(0.3)

        # Only free model memory during full cleanup
        if full_cleanup:
            for player in players:
                if hasattr(player, 'model') and player.model is not None:
                    try:
                        player.model.cpu()
                        player.model = None
                    except Exception:
                        pass
                if hasattr(player, 'state_encoder'):
                    player.state_encoder = None

    def _setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            if self._shutdown_requested:
                print("\nForce quitting...")
                sys.exit(1)
            print("\n\nShutdown requested. Saving checkpoint...")
            print("Press Ctrl+C again to force quit.")
            self._shutdown_requested = True

            # Suppress poke-env websocket error messages during shutdown
            # These are harmless "ConnectionClosedOK" messages from closing mid-battle
            logging.getLogger("poke_env").setLevel(logging.CRITICAL)
            logging.getLogger("websockets").setLevel(logging.CRITICAL)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_asyncio_exception_handler(self) -> None:
        """Set up asyncio exception handler to suppress ConnectionClosedOK errors.

        These errors are harmless and occur during normal cleanup when we close
        websocket connections while battles are still in progress. They also occur
        during shutdown. Suppressing them keeps the terminal output clean.
        """
        def silent_exception_handler(loop, context):
            exception = context.get("exception")
            if exception:
                exc_name = type(exception).__name__
                # Suppress all websocket connection closed errors
                if "ConnectionClosed" in exc_name or "ConnectionReset" in exc_name:
                    return  # Silently ignore
            # Also check the message for connection-related errors
            message = context.get("message", "")
            if "ConnectionClosed" in message:
                return
            # For other exceptions, use default handler
            loop.default_exception_handler(context)

        try:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(silent_exception_handler)
        except RuntimeError:
            pass  # No running loop, nothing to do

    def _setup_logging_filter(self) -> None:
        """Suppress noisy poke-env messages during training.

        poke-env logs various non-critical messages (ConnectionClosedOK, Invalid choice, etc.)
        that don't affect training. We disable logging at CRITICAL level since poke-env
        uses CRITICAL for non-critical errors that it recovers from automatically.
        """
        # Disable all logging - poke-env uses CRITICAL for recoverable errors
        # which clutters the output without providing actionable information
        logging.disable(logging.CRITICAL)

    async def collect_rollout_single(
        self,
        player: TrainablePlayer,
        opponent: Player,
        experience_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ) -> None:
        """Collect experiences from a single environment into shared queue.

        Runs battles continuously until stop_event is set. Each battle completes
        fully before checking stop_event, ensuring clean opponent cleanup.

        Args:
            player: The training player
            opponent: The opponent to play against
            experience_queue: Shared queue to put experiences into
            stop_event: Event that signals when to stop collecting
        """
        while not self._shutdown_requested:
            # Check stop_event BEFORE starting a new battle (not during)
            # This ensures battles always complete fully
            if stop_event.is_set():
                break

            # Play a battle with timeout to prevent hanging
            try:
                await asyncio.wait_for(
                    player.battle_against(opponent, n_battles=1),
                    timeout=120.0  # 2 minute timeout per battle
                )
            except asyncio.TimeoutError:
                continue  # Skip this battle, try another
            except asyncio.CancelledError:
                break

            # Get experiences from the battle and add to shared queue
            experiences = player.get_experiences()
            for exp in experiences:
                await experience_queue.put((player, exp))

        # Clean up this specific opponent after its battles are done
        await self._cleanup_single_opponent(opponent)

    async def collect_rollout_parallel(
        self,
        players: list[TrainablePlayer],
        opponents: list[Player],
        total_steps: int,
    ) -> list[list[dict[str, Any]]]:
        """Collect experiences from multiple environments in parallel.

        Uses a shared queue so fast environments can contribute more while
        slow ones are still battling. Stops when total_steps is reached.
        Each battle completes fully before stopping, ensuring clean cleanup.

        Args:
            players: List of training players
            opponents: List of opponents
            total_steps: Total steps to collect across ALL environments

        Returns:
            List of experience lists (one per environment)
        """
        experience_queue: asyncio.Queue = asyncio.Queue()
        stop_event = asyncio.Event()

        # Start all environments collecting into shared queue
        tasks = [
            asyncio.create_task(
                self.collect_rollout_single(player, opponent, experience_queue, stop_event)
            )
            for player, opponent in zip(players, opponents)
        ]

        # Collect experiences from queue until we have enough
        all_env_experiences: list[list[dict[str, Any]]] = [[] for _ in players]
        player_to_idx = {id(p): i for i, p in enumerate(players)}
        total_collected = 0

        try:
            while total_collected < total_steps and not self._shutdown_requested:
                try:
                    # Wait for experience with timeout to check stop conditions
                    player, exp = await asyncio.wait_for(experience_queue.get(), timeout=1.0)
                    idx = player_to_idx[id(player)]
                    all_env_experiences[idx].append(exp)
                    total_collected += 1
                except asyncio.TimeoutError:
                    # Check if all tasks are done (no more experiences coming)
                    if all(task.done() for task in tasks):
                        break
                    continue

        finally:
            # Signal collectors to stop after their current battle finishes
            stop_event.set()

            # Wait for battles to complete, but with a timeout
            # If shutdown was requested, give a short timeout then force cancel
            timeout = 5.0 if self._shutdown_requested else 30.0
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                # Force cancel any stuck tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Wait briefly for cancellation to complete
                await asyncio.gather(*tasks, return_exceptions=True)

        return all_env_experiences

    async def collect_rollout_with_retry(
        self,
        players: list[TrainablePlayer],
        opponents: list[Player],
        total_steps: int,
    ) -> tuple[list[list[dict[str, Any]]], list[TrainablePlayer], list[Player]]:
        """Collect rollouts with automatic retry on websocket errors.

        Args:
            players: List of training players
            opponents: List of opponents
            total_steps: Total steps to collect across ALL environments

        Returns:
            Tuple of (experiences, players, opponents) - players/opponents may be
            recreated if connection errors occurred
        """
        for attempt in range(MAX_ROLLOUT_RETRIES):
            try:
                experiences = await self.collect_rollout_parallel(
                    players, opponents, total_steps
                )
                return experiences, players, opponents

            except (ConnectionClosed, ConnectionClosedError, OSError) as e:
                # Websocket connection error - retry with fresh players
                error_name = type(e).__name__
                print(
                    f"\n  Connection error ({error_name}): {e}"
                )

                if attempt < MAX_ROLLOUT_RETRIES - 1:
                    print(
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
                    print(
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

        Handles uneven experience counts - fast environments contribute more.
        Experiences are added in round-robin order for mixing.

        Args:
            all_env_experiences: List of experience lists (one per environment)

        Returns:
            Total number of experiences added
        """
        total_added = 0
        num_envs = len(all_env_experiences)

        # Get lengths and create indices for round-robin iteration
        lengths = [len(exps) for exps in all_env_experiences]
        indices = [0] * num_envs  # Current index for each environment
        max_len = max(lengths) if lengths else 0

        # Round-robin through environments, skipping exhausted ones
        for _ in range(max_len):
            if self.buffer.ptr >= self.buffer.buffer_size:
                break

            for env in range(num_envs):
                if indices[env] >= lengths[env]:
                    continue  # This env is exhausted
                if self.buffer.ptr >= self.buffer.buffer_size:
                    break

                exp = all_env_experiences[env][indices[env]]
                indices[env] += 1

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
                total_added += 1

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
        self._setup_asyncio_exception_handler()
        self._setup_logging_filter()

        timesteps_to_train = total_timesteps or self.config.total_timesteps
        eval_interval = eval_interval or self.config.eval_interval
        save_interval = save_interval or self.config.checkpoint_interval

        # Calculate target: train for timesteps_to_train MORE steps from current position
        starting_timesteps = self.stats.total_timesteps
        target_timesteps = starting_timesteps + timesteps_to_train

        # Setup TensorBoard if not resuming
        if self.writer is None:
            self._run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir=str(self.log_dir / self._run_name))

        # Print startup info (one-time, not part of updating display)
        print("=" * 60)
        print("Pokemon Showdown RL Bot - Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Training for: {timesteps_to_train:,} steps")
        print(f"Starting from: {starting_timesteps:,}")
        print(f"Target: {target_timesteps:,}")
        print(f"Parallel environments: {self.num_envs}")
        print(f"Self-play: {'Enabled' if self.use_self_play else 'Disabled'}")
        print("=" * 60)
        print()

        # Create multiple players for parallel environments
        players = self._create_players()

        # Create display
        display = TrainingDisplay(
            total_timesteps=target_timesteps,
            start_timesteps=starting_timesteps,
        )
        display.initialize()

        try:
            while self.stats.total_timesteps < target_timesteps and not self._shutdown_requested:
                # Check memory usage at the start of each iteration
                mem_status, mem_percent = self.memory_monitor.check_memory()
                if mem_status == "hard_limit":
                    display.close()
                    print(f"\n[CRITICAL] Memory usage at {mem_percent:.1f}% - FORCE QUITTING!")
                    print("Attempting emergency checkpoint save...")
                    self._save_checkpoint("emergency_memory_checkpoint.pt")
                    print("Emergency checkpoint saved.")
                    sys.exit(1)
                elif mem_status == "soft_limit":
                    display.message(f"Memory at {mem_percent:.1f}% - initiating graceful shutdown")
                    self._shutdown_requested = True
                    self._memory_shutdown = True
                    break

                # Reset buffer and all player stats
                self.buffer.reset()
                for player in players:
                    player.reset_stats()

                # Create opponents for each environment
                # Opponents must connect to the same server as their corresponding player
                opponents = []
                opponent_infos = []
                for i in range(self.num_envs):
                    server_config = self.server_configs[i % len(self.server_configs)]
                    if self.self_play_manager:
                        opponent, opponent_info = self.self_play_manager.get_opponent(
                            self.config.battle_format,
                            server_configuration=server_config,
                        )
                        opponents.append(opponent)
                        opponent_infos.append(opponent_info)
                    else:
                        opponents.append(RandomPlayer(
                            battle_format=self.config.battle_format,
                            max_concurrent_battles=1,
                            server_configuration=server_config,
                        ))
                        opponent_infos.append(None)

                # Set model to eval mode once for the entire rollout (not per action)
                self.model.eval()

                # Collect rollouts in parallel with automatic retry on connection errors
                # Pass total steps - fast envs contribute more while slow ones are still battling
                all_env_experiences, players, opponents = await self.collect_rollout_with_retry(
                    players, opponents, self.config.rollout_steps
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

                # Compute advantages (buffer uses num_envs=1)
                last_value = np.zeros((1,), dtype=np.float32)
                last_done = np.ones((1,), dtype=np.float32)
                self.buffer.compute_advantages(last_value, last_done)

                # PPO update
                self.model.train()
                ppo_stats = self.ppo.update(self.buffer, self.config.batch_size)

                # Update stats
                self.stats.total_timesteps += total_added
                self.stats.total_episodes += rollout_stats["battles"]
                self.stats.total_updates += 1

                # Log to TensorBoard (every 5 iterations to reduce I/O overhead)
                if self.stats.total_updates % 5 == 0:
                    self._log_training(rollout_stats, ppo_stats)

                # Update display
                _, mem_percent = self.memory_monitor.check_memory()
                display.update(
                    timesteps=self.stats.total_timesteps,
                    rollout_stats=rollout_stats,
                    ppo_stats=ppo_stats,
                    skill=self.self_play_manager.agent_skill if self.self_play_manager else None,
                    pool_size=self.self_play_manager.opponent_pool.size if self.self_play_manager else 0,
                    memory_str=self.memory_monitor.get_memory_status_string(),
                    memory_percent=mem_percent,
                    best_skill=self._global_best_skill,
                )

                # Add checkpoint to self-play pool
                if self.self_play_manager and self.self_play_manager.should_add_checkpoint(
                    self.stats.total_timesteps
                ):
                    self.self_play_manager.add_checkpoint(self.model, self.stats.total_timesteps)
                    display.message(f"Added checkpoint to opponent pool (size: {self.self_play_manager.opponent_pool.size})")

                # Save checkpoint periodically
                if self.stats.total_timesteps % save_interval < total_added:
                    self._save_checkpoint()

                # Track best win rate (session best)
                if rollout_stats["win_rate"] > self.stats.best_win_rate:
                    self.stats.best_win_rate = rollout_stats["win_rate"]
                    self._best_win_rate = rollout_stats["win_rate"]  # For tuning

                # Save best_model.pt based on skill rating (not win rate, since win rate caps at 100%)
                # Only save if we beat the global best from existing best_model.pt
                if self.self_play_manager:
                    current_skill = self.self_play_manager.agent_skill
                    if current_skill > self._global_best_skill:
                        old_best = self._global_best_skill
                        self._global_best_skill = current_skill
                        self._save_checkpoint("best_model.pt")
                        display.message(f"NEW BEST MODEL! Skill: {old_best:.0f} -> {current_skill:.0f}")

                # Opponents are now cleaned up individually in collect_rollout_single
                # after each battle finishes naturally. Just do periodic GC.
                if self.stats.total_updates % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except asyncio.CancelledError:
            # Task was cancelled during shutdown - this is expected
            display.close()
            print("Training cancelled.")
        except Exception as e:
            display.close()
            print(f"Error during training: {e}")
            print("Saving emergency checkpoint...")
            self._save_checkpoint("emergency_checkpoint.pt")
            raise
        finally:
            # Clean up player connections (full cleanup since we're shutting down)
            try:
                await self._cleanup_players(players, full_cleanup=True)
            except Exception:
                pass  # Ignore cleanup errors

            # Close display and save checkpoint
            display.close()
            if self._shutdown_requested or self.stats.total_timesteps > 0:
                self._save_checkpoint("latest.pt")

            if self.writer:
                self.writer.close()

        # Print final summary
        print()
        print("=" * 60)
        if self._memory_shutdown:
            print("Training Stopped - HIGH MEMORY USAGE (checkpoint saved)")
            print(f"Memory at shutdown: {self.memory_monitor.get_memory_status_string()}")
        elif self._shutdown_requested:
            print("Training Stopped (checkpoint saved)")
        else:
            print("Training Complete!")
        print(f"Total timesteps: {self.stats.total_timesteps:,}")
        print(f"Total episodes: {self.stats.total_episodes:,}")
        print(f"Best win rate: {self.stats.best_win_rate:.1%}")
        if self.self_play_manager:
            print(f"Final Skill: {self.self_play_manager.agent_skill:.0f}")
        print(f"Checkpoint saved: {self.save_dir / 'latest.pt'}")
        print("=" * 60)

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
            self.writer.add_scalar("self_play/agent_skill", sp_stats["agent_skill"], step)
            self.writer.add_scalar("self_play/pool_size", sp_stats["pool_size"], step)
            self.writer.add_scalar("self_play/pool_avg_skill", sp_stats["pool_avg_skill"], step)
            self.writer.add_scalar("self_play/games_vs_self_play", sp_stats["games_vs_self_play"], step)
            self.writer.add_scalar("self_play/games_vs_random", sp_stats["games_vs_random"], step)

    def _load_existing_best_skill(self) -> float:
        """Load the skill rating from existing best_model.pt if it exists.

        This prevents overwriting a better model from a previous training run
        with a worse model from a new run that starts fresh. Skill rating is
        used instead of win rate because win rate caps at 100%, while skill
        continues to grow as the model beats stronger opponents.
        """
        best_model_path = self.save_dir / "best_model.pt"
        if best_model_path.exists():
            try:
                checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
                skill = checkpoint.get("self_play", {}).get("agent_skill", 0.0)
                steps = checkpoint.get("stats", {}).get("total_timesteps", 0)
                print(f"Loaded best_model.pt: Skill={skill:.0f}, Steps={steps:,}")
                return skill
            except Exception as e:
                print(f"Warning: Could not load best_model.pt: {e}")
                return 0.0
        print("No existing best_model.pt found, starting fresh")
        return 0.0

    def _save_checkpoint(self, filename: str | None = None) -> None:
        """Save a training checkpoint.

        Always updates latest.pt to point to the most recent checkpoint,
        ensuring training can resume even after unexpected termination.
        """
        is_numbered = filename is None
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
                "agent_skill": self.self_play_manager.agent_skill,
                "last_checkpoint_timestep": self.self_play_manager.last_checkpoint_timestep,
                "games_vs_self_play": self.self_play_manager.games_vs_self_play,
                "games_vs_random": self.self_play_manager.games_vs_random,
            }

        torch.save(checkpoint, path)

        # Verify best_model.pt saves to ensure we don't lose the best model
        if filename == "best_model.pt":
            # Verify the save by reading it back
            try:
                verify = torch.load(path, map_location="cpu", weights_only=False)
                saved_skill = verify.get("self_play", {}).get("agent_skill", 0)
                saved_steps = verify.get("stats", {}).get("total_timesteps", 0)
                expected_skill = checkpoint.get("self_play", {}).get("agent_skill", 0)
                if abs(saved_skill - expected_skill) > 0.1:
                    print(f"ERROR: best_model.pt verification failed! Expected skill {expected_skill}, got {saved_skill}")
                else:
                    print(f"Saved best_model.pt: Skill={saved_skill:.0f}, Steps={saved_steps:,} [VERIFIED]")
            except Exception as e:
                print(f"ERROR: Could not verify best_model.pt: {e}")
        else:
            print(f"Saved checkpoint: {path}")

        # Always update latest.pt when saving numbered checkpoints
        # This ensures resume works even after unexpected termination (kill/crash)
        if is_numbered:
            latest_path = self.save_dir / "latest.pt"
            torch.save(checkpoint, latest_path)

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
            # Support both new "agent_skill" and old "agent_elo" keys for backwards compatibility
            self.self_play_manager.agent_skill = sp_state.get("agent_skill", sp_state.get("agent_elo", 1000.0))
            self.self_play_manager.last_checkpoint_timestep = sp_state.get("last_checkpoint_timestep", 0)
            self.self_play_manager.games_vs_self_play = sp_state.get("games_vs_self_play", 0)
            self.self_play_manager.games_vs_random = sp_state.get("games_vs_random", 0)

        print(f"Loaded checkpoint: {path}")
        print(f"  Timesteps: {self.stats.total_timesteps:,}")
        print(f"  Episodes: {self.stats.total_episodes:,}")
        print(f"  Best win rate: {self.stats.best_win_rate:.1%}")
        if self.self_play_manager and "self_play" in checkpoint:
            print(f"  Agent Skill: {self.self_play_manager.agent_skill:.0f}")

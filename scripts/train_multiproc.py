#!/usr/bin/env python3
"""Multi-process training script that bypasses Python's GIL.

Architecture:
- Main process (coordinator): Owns the model, manages opponent pool, does PPO updates
- Worker processes: Run battles, collect experiences, return them to coordinator

This allows true parallelism across CPU cores for experience collection.

Usage:
    # Start servers first (one per worker recommended)
    for i in {0..1}; do
        node ~/pokemon-showdown/pokemon-showdown start --no-security --port $((8000+i)) &
    done

    # Run multi-process training (2 workers, 6 envs each = 12 total)
    python scripts/train_multiproc.py --workers 2 --envs-per-worker 6 --server-ports 8000 8001

    # Resume from checkpoint
    python scripts/train_multiproc.py --workers 2 --envs-per-worker 6 --resume data/checkpoints/latest.pt
"""

import argparse
import asyncio
import multiprocessing as mp
import os
import signal
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from queue import Empty

# Suppress warnings before torch import
warnings.filterwarnings("ignore", message=".*hipBLASLt.*")
warnings.filterwarnings("ignore", message=".*experimental.*")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from showdown_bot.config import TrainingConfig


def worker_process(
    worker_id: int,
    server_port: int,
    num_envs: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    control_queue: mp.Queue,
    device_str: str,
    battle_format: str,
):
    """Worker process that runs battles and collects experiences.

    This runs in a separate process, bypassing the GIL.
    Each worker gets its own PS server for optimal performance.
    """
    # Delay imports to avoid issues with multiprocessing
    import torch
    from poke_env.player import Player, RandomPlayer
    from poke_env.battle import AbstractBattle
    from poke_env.ps_client import ServerConfiguration

    from showdown_bot.models.network import PolicyValueNetwork
    from showdown_bot.environment.state_encoder import StateEncoder
    from showdown_bot.training.trainer import calculate_reward
    from showdown_bot.environment.battle_env import MaxDamagePlayer

    print(f"[Worker {worker_id}] Starting on port {server_port} with {num_envs} envs")

    device = torch.device(device_str)

    # Create network and encoder
    encoder = StateEncoder(device=device)
    network = PolicyValueNetwork.from_config().to(device)
    network.eval()

    # Opponent network for self-play (loaded when needed)
    opponent_network = PolicyValueNetwork.from_config().to(device)
    opponent_network.eval()

    server_config = ServerConfiguration(
        websocket_url=f"ws://localhost:{server_port}/showdown/websocket",
        authentication_url="https://play.pokemonshowdown.com/action.php?",
    )

    class WorkerTrainablePlayer(Player):
        """Player that collects experiences for this worker.

        IMPORTANT: Rewards are attributed to the action that caused them.
        - At turn t, we take action A_t
        - At turn t+1, we see the result and assign the HP-delta reward to experience t
        - Terminal reward (win/loss) is assigned to the last experience
        """

        def __init__(self, model, state_encoder, dev, **kwargs):
            super().__init__(**kwargs)
            self.model = model
            self.state_encoder = state_encoder
            self.device = dev
            self.current_experiences = []
            self.prev_hp_fraction = None
            self.prev_opp_hp_fraction = None
            self.prev_our_fainted = 0
            self.prev_opp_fainted = 0
            self._won = None

        def choose_move(self, battle: AbstractBattle):
            state = self.state_encoder.encode_battle(battle)

            player_pokemon = state.player_pokemon.unsqueeze(0).to(self.device)
            opponent_pokemon = state.opponent_pokemon.unsqueeze(0).to(self.device)
            player_active_idx = torch.tensor([state.player_active_idx], device=self.device)
            opponent_active_idx = torch.tensor([state.opponent_active_idx], device=self.device)
            field_state = state.field_state.unsqueeze(0).to(self.device)
            action_mask = state.action_mask.unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(
                    player_pokemon, opponent_pokemon,
                    player_active_idx, opponent_active_idx,
                    field_state, action_mask,
                )

            action_idx = action.item()
            log_prob_val = log_prob.item()
            value_val = value.item()

            # Calculate current HP state
            our_remaining = [p for p in battle.team.values() if not p.fainted]
            opp_remaining = [p for p in battle.opponent_team.values() if not p.fainted]
            current_our_hp = sum(p.current_hp_fraction for p in our_remaining) / max(1, len(our_remaining))
            current_opp_hp = sum(p.current_hp_fraction for p in opp_remaining) / max(1, len(opp_remaining))
            current_our_fainted = sum(1 for p in battle.team.values() if p.fainted)
            current_opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)

            # FIXED: Attribute reward to the PREVIOUS experience (the action that caused this state)
            if self.current_experiences and self.prev_hp_fraction is not None:
                # HP differential reward - attributed to previous action
                our_hp_delta = current_our_hp - self.prev_hp_fraction
                opp_hp_delta = current_opp_hp - self.prev_opp_hp_fraction

                # KO differential (only the delta, not cumulative)
                our_ko_delta = current_our_fainted - self.prev_our_fainted
                opp_ko_delta = current_opp_fainted - self.prev_opp_fainted

                # Reward for the action that caused this transition
                intermediate_reward = 0.0
                intermediate_reward += 0.1 * (-opp_hp_delta)  # Positive when opponent loses HP
                intermediate_reward += 0.1 * our_hp_delta  # Positive when we don't lose HP
                intermediate_reward += 0.1 * (opp_ko_delta - our_ko_delta)  # Bonus for KOs

                # Update the PREVIOUS experience with this reward
                self.current_experiences[-1]["reward"] = intermediate_reward

            # Update tracking for next turn
            self.prev_hp_fraction = current_our_hp
            self.prev_opp_hp_fraction = current_opp_hp
            self.prev_our_fainted = current_our_fainted
            self.prev_opp_fainted = current_opp_fainted

            # Store current experience with reward=0 (will be filled next turn or at game end)
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
                "reward": 0.0,  # Will be filled when we see the result
                "done": False,
            })

            order = self.state_encoder.action_to_battle_order(action_idx, battle)
            return order if order else self.choose_random_move(battle)

        def _battle_finished_callback(self, battle: AbstractBattle):
            if battle.won:
                final_reward = 1.0
                self._won = True
            elif battle.lost:
                final_reward = -1.0
                self._won = False
            else:
                final_reward = 0.0
                self._won = None

            if self.current_experiences:
                # Add terminal reward to the last experience
                self.current_experiences[-1]["reward"] += final_reward
                self.current_experiences[-1]["done"] = True

            self.prev_hp_fraction = None
            self.prev_opp_hp_fraction = None
            self.prev_our_fainted = 0
            self.prev_opp_fainted = 0

        def get_experiences(self):
            exps = self.current_experiences.copy()
            self.current_experiences = []
            return exps

        def reset(self):
            self.current_experiences = []
            self.prev_hp_fraction = None
            self.prev_opp_hp_fraction = None
            self.prev_our_fainted = 0
            self.prev_opp_fainted = 0
            self._won = None

    class SelfPlayOpponent(Player):
        """Opponent using a historical model checkpoint."""

        def __init__(self, model, state_encoder, dev, **kwargs):
            super().__init__(**kwargs)
            self.model = model
            self.state_encoder = state_encoder
            self.device = dev

        def choose_move(self, battle: AbstractBattle):
            state = self.state_encoder.encode_battle(battle)

            player_pokemon = state.player_pokemon.unsqueeze(0).to(self.device)
            opponent_pokemon = state.opponent_pokemon.unsqueeze(0).to(self.device)
            player_active_idx = torch.tensor([state.player_active_idx], device=self.device)
            opponent_active_idx = torch.tensor([state.opponent_active_idx], device=self.device)
            field_state = state.field_state.unsqueeze(0).to(self.device)
            action_mask = state.action_mask.unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, _, _, _ = self.model.get_action_and_value(
                    player_pokemon, opponent_pokemon,
                    player_active_idx, opponent_active_idx,
                    field_state, action_mask,
                )

            order = self.state_encoder.action_to_battle_order(action.item(), battle)
            return order if order else self.choose_random_move(battle)

    async def run_battles(task: dict) -> dict:
        """Run battles based on task from coordinator."""
        model_weights = task["model_weights"]
        opponent_types = task["opponent_types"]  # List of opponent types for each env
        opponent_weights = task.get("opponent_weights")  # Weights for self-play opponent

        # Load current model weights
        network.load_state_dict(model_weights)

        # Load opponent weights if provided (for self-play)
        if opponent_weights is not None:
            opponent_network.load_state_dict(opponent_weights)

        players = []
        opponents = []
        env_opponent_types = []

        for i in range(num_envs):
            opp_type = opponent_types[i % len(opponent_types)]
            env_opponent_types.append(opp_type)

            player = WorkerTrainablePlayer(
                model=network,
                state_encoder=encoder,
                dev=device,
                battle_format=battle_format,
                max_concurrent_battles=1,
                server_configuration=server_config,
            )
            players.append(player)

            # Create appropriate opponent
            # "best" type uses either MaxDamage (no weights) or neural net (with weights)
            if opp_type == "best" and opponent_weights is not None:
                # Best checkpoint exists - use neural network opponent
                opponent = SelfPlayOpponent(
                    model=opponent_network,
                    state_encoder=encoder,
                    dev=device,
                    battle_format=battle_format,
                    max_concurrent_battles=1,
                    server_configuration=server_config,
                )
            elif opp_type == "max_damage" or (opp_type == "best" and opponent_weights is None):
                # No best checkpoint yet, or explicit MaxDamage - use MaxDamage
                opponent = MaxDamagePlayer(
                    battle_format=battle_format,
                    max_concurrent_battles=1,
                    server_configuration=server_config,
                )
                env_opponent_types[i] = "max_damage"
            elif opp_type == "self_play" and opponent_weights is not None:
                # Legacy self-play support
                opponent = SelfPlayOpponent(
                    model=opponent_network,
                    state_encoder=encoder,
                    dev=device,
                    battle_format=battle_format,
                    max_concurrent_battles=1,
                    server_configuration=server_config,
                )
            else:
                # Fallback to max_damage
                opponent = MaxDamagePlayer(
                    battle_format=battle_format,
                    max_concurrent_battles=1,
                    server_configuration=server_config,
                )
                env_opponent_types[i] = "max_damage"

            opponents.append(opponent)

        async def run_single(idx, player, opponent):
            try:
                await asyncio.wait_for(
                    player.battle_against(opponent, n_battles=1),
                    timeout=120.0
                )
                return idx, player.get_experiences(), player._won, env_opponent_types[idx]
            except Exception as e:
                print(f"[Worker {worker_id}] Battle {idx} error: {e}")
                return idx, [], None, env_opponent_types[idx]

        tasks = [run_single(i, p, o) for i, (p, o) in enumerate(zip(players, opponents))]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_experiences = []
        battle_results = []  # (won, opponent_type) for each battle

        for result in results:
            if isinstance(result, Exception):
                continue
            idx, exps, won, opp_type = result
            all_experiences.extend(exps)
            if won is not None:
                battle_results.append((won, opp_type))

        # Cleanup connections
        cleanup_tasks = []
        for p in players + opponents:
            if hasattr(p, 'ps_client') and p.ps_client:
                cleanup_tasks.append(p.ps_client.stop_listening())
        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                pass

        return {
            "experiences": all_experiences,
            "steps": len(all_experiences),
            "battle_results": battle_results,
        }

    # Main worker loop - use os._exit for immediate termination on signal
    import os as _os

    def handle_signal(signum, frame):
        print(f"[Worker {worker_id}] Shutdown")
        _os._exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while True:
        try:
            # Check control queue
            try:
                msg = control_queue.get_nowait()
                if msg == "stop":
                    break
            except Empty:
                pass

            # Wait for task
            try:
                task = task_queue.get(timeout=1.0)
            except Empty:
                continue

            if task is None:
                break

            # Run battles in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_battles(task))
                result_queue.put(result)
            finally:
                loop.close()

        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"[Worker {worker_id}] Shutdown")


class MultiProcessTrainer:
    """Coordinates multi-process training with self-play and curriculum learning."""

    def __init__(
        self,
        num_workers: int,
        envs_per_worker: int,
        server_ports: list[int],
        device: torch.device,
        config: TrainingConfig,
        benchmark_games: int = 50,
    ):
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.server_ports = server_ports
        self.device = device
        self.config = config
        self.benchmark_games = benchmark_games

        # Import here to ensure proper initialization
        from showdown_bot.models.network import PolicyValueNetwork
        from showdown_bot.training.ppo import PPO
        from showdown_bot.training.buffer import RolloutBuffer
        from showdown_bot.training.self_play import SelfPlayManager

        self.network = PolicyValueNetwork.from_config().to(device)

        # Buffer sized for combined experiences from all workers
        self.buffer = RolloutBuffer(
            buffer_size=config.rollout_steps,
            num_envs=1,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=device,
        )

        self.ppo = PPO.from_config(self.network, device=device, config=config)

        # Self-play manager
        self.self_play_manager = SelfPlayManager(
            pool_dir=Path("data/self_play_pool"),
            max_pool_size=20,
            self_play_ratio=config.self_play_ratio,
            checkpoint_interval=config.checkpoint_interval,
            sampling_strategy="diverse",
            device=device,
            curriculum_enabled=config.curriculum_enabled,
            curriculum_bench_min=config.curriculum_bench_min,
            curriculum_bench_max=config.curriculum_bench_max,
            curriculum_early_self_play=config.curriculum_early_self_play,
            curriculum_early_max_damage=config.curriculum_early_max_damage,
            curriculum_late_self_play=config.curriculum_late_self_play,
            curriculum_late_max_damage=config.curriculum_late_max_damage,
        )

        # Stats
        self.total_timesteps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.best_bench_rate = 0.0
        self.last_bench_rate = None  # From periodic evaluation
        self.last_best_eval_rate = None  # Win rate vs best checkpoint
        self.prev_best_bench = None  # Bench% when best_model was last promoted (for bench-gated promotion)

        # Best checkpoint for "current vs best" training
        # If no best exists, we train against MaxDamage until we beat it
        self.save_dir = Path("data/checkpoints")
        self.best_checkpoint_path = self.save_dir / "best_model.pt"

        # Rolling win rate tracker for real-time display
        self._rolling_results: deque[bool] = deque(maxlen=100)

        # Process management
        self.workers = []
        self.task_queues = []
        self.result_queue = None
        self.control_queues = []
        self._shutdown_requested = False

    def start_workers(self):
        """Start worker processes."""
        self.result_queue = mp.Queue()

        for i in range(self.num_workers):
            # Each worker gets its own server port
            port = self.server_ports[i % len(self.server_ports)]

            task_queue = mp.Queue()
            control_queue = mp.Queue()

            worker = mp.Process(
                target=worker_process,
                args=(
                    i,
                    port,
                    self.envs_per_worker,
                    task_queue,
                    self.result_queue,
                    control_queue,
                    str(self.device),
                    self.config.battle_format,
                ),
                daemon=True,
            )
            worker.start()

            self.workers.append(worker)
            self.task_queues.append(task_queue)
            self.control_queues.append(control_queue)

        print(f"Started {self.num_workers} workers")
        time.sleep(2)

    def stop_workers(self):
        """Stop all workers immediately."""
        print("Stopping workers...")

        # Terminate workers immediately - they handle SIGTERM with os._exit
        for w in self.workers:
            if w.is_alive():
                w.terminate()

        # Brief wait for clean exit
        for w in self.workers:
            w.join(timeout=2)
            if w.is_alive():
                w.kill()  # SIGKILL if still alive

        print("All workers stopped")

    def get_opponent_types(self) -> list[str]:
        """Get opponent types for training - 100% vs best.

        'best' means either:
        - MaxDamage (if no best checkpoint exists yet)
        - The best checkpoint (once we've beaten MaxDamage)
        """
        total_envs = self.num_workers * self.envs_per_worker
        return ["best"] * total_envs

    def has_best_checkpoint(self) -> bool:
        """Check if a best checkpoint exists."""
        return self.best_checkpoint_path.exists()

    def get_best_opponent_weights(self) -> dict | None:
        """Get weights for the 'best' opponent.

        Returns:
            - None if no best checkpoint (workers will use MaxDamage)
            - Checkpoint weights if best_model.pt exists
        """
        if not self.has_best_checkpoint():
            return None

        # Load best checkpoint weights
        checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return {k: v.cpu() for k, v in checkpoint["model_state_dict"].items()}
        return {k: v.cpu() for k, v in checkpoint.items()}

    def broadcast_tasks(self, opponent_types: list[str], opponent_weights: dict | None):
        """Send tasks to all workers."""
        model_weights = {k: v.cpu() for k, v in self.network.state_dict().items()}

        types_per_worker = len(opponent_types) // self.num_workers

        for i, q in enumerate(self.task_queues):
            start_idx = i * types_per_worker
            end_idx = start_idx + types_per_worker if i < self.num_workers - 1 else len(opponent_types)
            worker_types = opponent_types[start_idx:end_idx]

            task = {
                "model_weights": model_weights,
                "opponent_types": worker_types,
                "opponent_weights": opponent_weights,
            }
            q.put(task)

    def collect_results(self, timeout: float = 300.0) -> list[dict]:
        """Collect results from all workers."""
        results = []
        deadline = time.time() + timeout

        while len(results) < self.num_workers and time.time() < deadline:
            if self._shutdown_requested:
                break
            try:
                result = self.result_queue.get(timeout=1.0)
                results.append(result)
            except Empty:
                continue

        return results

    def run_promotion_evaluation(self, num_games: int = 100) -> tuple[float | None, list[dict]]:
        """Run evaluation games against the 'best' opponent for promotion decision.

        If no best checkpoint exists, evaluates against MaxDamage.
        If best checkpoint exists, evaluates against that checkpoint.

        Args:
            num_games: Number of evaluation games to run

        Returns:
            Tuple of (win_rate, experiences):
            - win_rate: Win rate against best (0.0 to 1.0), or None if failed
            - experiences: List of experience dicts from evaluation games
        """
        if self._shutdown_requested:
            return None, []

        # Send evaluation task to first worker only
        model_weights = {k: v.cpu() for k, v in self.network.state_dict().items()}

        # Get best opponent weights (None means use MaxDamage)
        opponent_weights = self.get_best_opponent_weights()

        # All games vs best
        eval_opponent_types = ["best"] * self.envs_per_worker

        task = {
            "model_weights": model_weights,
            "opponent_types": eval_opponent_types,
            "opponent_weights": opponent_weights,
        }

        wins = 0
        total = 0
        all_experiences = []

        # Run multiple batches to get num_games
        games_per_batch = self.envs_per_worker
        batches_needed = (num_games + games_per_batch - 1) // games_per_batch

        for _ in range(batches_needed):
            if self._shutdown_requested:
                break

            # Send task to first worker
            self.task_queues[0].put(task)

            # Wait for result
            try:
                result = self.result_queue.get(timeout=120.0)

                # Collect experiences from evaluation games
                all_experiences.extend(result.get("experiences", []))

                for won, opp_type in result.get("battle_results", []):
                    if won is not None:
                        total += 1
                        if won:
                            wins += 1

                    if total >= num_games:
                        break
            except Empty:
                continue

            if total >= num_games:
                break

        if total == 0:
            return None, all_experiences

        return wins / total, all_experiences

    def run_benchmark_evaluation(self, num_games: int = 50) -> tuple[float | None, list[dict]]:
        """Run evaluation games against MaxDamage and collect experiences.

        This provides the Bench% metric AND returns experiences for training.
        Runs games using one worker.

        Args:
            num_games: Number of evaluation games to run

        Returns:
            Tuple of (win_rate, experiences):
            - win_rate: Win rate against MaxDamage (0.0 to 1.0), or None if failed
            - experiences: List of experience dicts from benchmark games
        """
        if self._shutdown_requested:
            return None, []

        # Send evaluation task to first worker only
        model_weights = {k: v.cpu() for k, v in self.network.state_dict().items()}

        # All games vs MaxDamage
        eval_opponent_types = ["max_damage"] * self.envs_per_worker

        task = {
            "model_weights": model_weights,
            "opponent_types": eval_opponent_types,
            "opponent_weights": None,  # Not needed for MaxDamage
        }

        wins = 0
        total = 0
        all_experiences = []

        # Run multiple batches to get num_games
        games_per_batch = self.envs_per_worker
        batches_needed = (num_games + games_per_batch - 1) // games_per_batch

        for _ in range(batches_needed):
            if self._shutdown_requested:
                break

            # Send task to first worker
            self.task_queues[0].put(task)

            # Wait for result
            try:
                result = self.result_queue.get(timeout=120.0)

                # Collect experiences from benchmark games
                all_experiences.extend(result.get("experiences", []))

                for won, opp_type in result.get("battle_results", []):
                    if opp_type == "max_damage" and won is not None:
                        total += 1
                        if won:
                            wins += 1
                        # Update self-play manager's tracking
                        self.self_play_manager.update_after_game(None, won, "max_damage")

                    if total >= num_games:
                        break
            except Empty:
                continue

            if total >= num_games:
                break

        if total == 0:
            return None, all_experiences

        return wins / total, all_experiences

    def train(
        self,
        total_timesteps: int,
        save_dir: str = "data/checkpoints",
        log_dir: str = "runs",
    ):
        """Main training loop."""
        from torch.utils.tensorboard import SummaryWriter
        from showdown_bot.training.trainer import TrainingDisplay, MemoryMonitor

        save_path = Path(save_dir)
        memory_monitor = MemoryMonitor()
        save_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)

        def handle_signal(signum, frame):
            print("\nShutdown requested...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        total_envs = self.num_workers * self.envs_per_worker
        target_timesteps = self.total_timesteps + total_timesteps

        print(f"\n{'='*60}")
        print("Multi-Process Training")
        print(f"{'='*60}")
        print(f"Workers: {self.num_workers}")
        print(f"Envs per worker: {self.envs_per_worker}")
        print(f"Total parallel envs: {total_envs}")
        print(f"Server ports: {self.server_ports}")
        print(f"Starting from: {self.total_timesteps:,}")
        print(f"Target: {target_timesteps:,}")
        if self.has_best_checkpoint():
            print(f"Training: Current vs Best checkpoint")
        else:
            print(f"Training: Current vs MaxDamage (no best checkpoint yet)")
        print(f"{'='*60}\n")

        self.start_workers()

        # Create display
        display = TrainingDisplay(
            total_timesteps=target_timesteps,
            start_timesteps=self.total_timesteps,
        )
        display.initialize()

        last_checkpoint_timesteps = self.total_timesteps

        # Early checkpoint schedule: more frequent evaluations to avoid training too long on bad bot
        # Checkpoints at: 25k, 50k, 100k, 150k, 200k, then every 100k after
        early_checkpoints = [25000, 50000, 100000, 150000, 200000]

        def get_next_checkpoint(current_steps: int) -> int:
            """Get the next checkpoint timestep."""
            for cp in early_checkpoints:
                if current_steps < cp:
                    return cp
            # After early phase, checkpoint every 100k
            return ((current_steps // 100000) + 1) * 100000

        try:
            while self.total_timesteps < target_timesteps and not self._shutdown_requested:
                # Reset buffer
                self.buffer.reset()

                # Get opponent configuration - current vs best
                opponent_types = self.get_opponent_types()
                opponent_weights = self.get_best_opponent_weights()

                # Set model to eval for inference
                self.network.eval()

                # Broadcast tasks to workers
                t_start = time.perf_counter()
                self.broadcast_tasks(opponent_types, opponent_weights)

                # Collect results
                results = self.collect_results()
                if not results:
                    display.message("No results from workers, retrying...")
                    continue

                t_rollout = time.perf_counter() - t_start

                # Aggregate experiences and battle results
                total_steps = 0
                for result in results:
                    for exp in result["experiences"]:
                        if self.buffer.ptr < self.buffer.buffer_size:
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
                            total_steps += 1

                    # Update self-play manager with battle results
                    for won, opp_type in result["battle_results"]:
                        self.self_play_manager.update_after_game(None, won, opp_type)
                        self.total_episodes += 1
                        # Track rolling win rate
                        if won is not None:
                            self._rolling_results.append(won)

                self.total_timesteps += total_steps

                # Compute advantages
                if self.buffer.ptr > 0:
                    last_value = np.array([self.buffer.values[:self.buffer.ptr].mean()], dtype=np.float32)
                else:
                    last_value = np.zeros((1,), dtype=np.float32)
                last_done = np.zeros((1,), dtype=np.float32)
                self.buffer.compute_advantages(last_value, last_done)

                # PPO update
                self.network.train()
                progress = self.total_timesteps / target_timesteps
                self.ppo.update_learning_rate(progress)

                t_ppo_start = time.perf_counter()
                ppo_stats = self.ppo.update(self.buffer, self.config.batch_size)
                t_ppo = time.perf_counter() - t_ppo_start

                self.total_updates += 1

                # Note: Opponent pool disabled - using true self-play (current model vs itself)
                # Historical checkpoints are not used for opponents

                # Get stats for display
                # Use rolling win rate from training games (last 100)
                if self._rolling_results:
                    overall_win_rate = sum(self._rolling_results) / len(self._rolling_results)
                else:
                    overall_win_rate = 0.0

                # Use last benchmark result for bench_rate display
                bench_rate = self.last_bench_rate

                # Update display
                display.update(
                    timesteps=self.total_timesteps,
                    rollout_stats={
                        "battles": sum(len(r["battle_results"]) for r in results),
                        "wins": sum(sum(1 for won, _ in r["battle_results"] if won) for r in results),
                        "win_rate": overall_win_rate,
                        "avg_reward": 0.0,
                        "avg_length": 0.0,
                    },
                    ppo_stats=ppo_stats,
                    pool_size=self.self_play_manager.opponent_pool.size,
                    benchmark_win_rate=bench_rate,
                )

                # Log to TensorBoard
                if self.total_updates % 5 == 0:
                    writer.add_scalar("train/policy_loss", ppo_stats.policy_loss, self.total_timesteps)
                    writer.add_scalar("train/value_loss", ppo_stats.value_loss, self.total_timesteps)
                    writer.add_scalar("train/entropy", ppo_stats.entropy_loss, self.total_timesteps)
                    writer.add_scalar("train/explained_variance", ppo_stats.explained_variance, self.total_timesteps)
                    writer.add_scalar("train/clip_fraction", ppo_stats.clip_fraction, self.total_timesteps)
                    writer.add_scalar("train/approx_kl", ppo_stats.approx_kl, self.total_timesteps)
                    writer.add_scalar("rollout/win_rate", overall_win_rate, self.total_timesteps)
                    if bench_rate is not None:
                        writer.add_scalar("rollout/bench_win_rate", bench_rate, self.total_timesteps)

                # Save checkpoint and run evaluation periodically
                # Early training: 25k, 50k, 100k, 150k, 200k; then every 100k
                next_checkpoint = get_next_checkpoint(last_checkpoint_timesteps)
                if self.total_timesteps >= next_checkpoint:
                    self.save_checkpoint(save_path / "latest.pt")
                    last_checkpoint_timesteps = self.total_timesteps

                    # Run MaxDamage benchmark at every checkpoint
                    # Experiences are added to training buffer
                    print(f"\nRunning MaxDamage benchmark ({self.benchmark_games} games)...")
                    bench_rate_eval, bench_experiences = self.run_benchmark_evaluation(num_games=self.benchmark_games)
                    if bench_rate_eval is not None:
                        self.last_bench_rate = bench_rate_eval
                        bench_rate = bench_rate_eval
                        print(f"Bench%: {bench_rate:.1%} vs MaxDamage")
                        writer.add_scalar("eval/bench_win_rate", bench_rate, self.total_timesteps)
                        if bench_rate > self.best_bench_rate:
                            self.best_bench_rate = bench_rate

                    # Add benchmark experiences to training buffer (don't waste them)
                    for exp in bench_experiences:
                        if self.buffer.ptr < self.buffer.buffer_size:
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

                    # Evaluate current vs best for promotion
                    # If no best exists, this evaluates vs MaxDamage
                    print("Running promotion evaluation vs best (200 games)...")
                    eval_vs_best, promo_experiences = self.run_promotion_evaluation(num_games=200)

                    # Add promotion eval experiences to training buffer
                    for exp in promo_experiences:
                        if self.buffer.ptr < self.buffer.buffer_size:
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

                    if eval_vs_best is not None:
                        self.last_best_eval_rate = eval_vs_best
                        opponent_name = "Best" if self.has_best_checkpoint() else "MaxDamage"
                        print(f"vs {opponent_name}: {eval_vs_best:.1%}")
                        writer.add_scalar("eval/vs_best_win_rate", eval_vs_best, self.total_timesteps)

                        # Bench-gated promotion: must beat best AND maintain bench%
                        # bench_threshold = max(floor, prev_best_bench - tolerance)
                        bench_floor = 0.20  # Must be better than random
                        bench_tolerance = 0.08  # Allow ~1 std dev variance
                        prev_bench = self.prev_best_bench if self.prev_best_bench is not None else 0.0
                        bench_threshold = max(bench_floor, prev_bench - bench_tolerance)

                        # Check both conditions for promotion
                        beats_best = eval_vs_best >= 0.30
                        meets_bench = bench_rate_eval is not None and bench_rate_eval >= bench_threshold

                        if beats_best and meets_bench:
                            self.save_checkpoint(save_path / "best_model.pt")
                            self.prev_best_bench = bench_rate_eval  # Update for next promotion
                            print(f"Promoted! Current -> Best")
                            print(f"  vs Best: {eval_vs_best:.1%} >= 30%")
                            print(f"  Bench%: {bench_rate_eval:.1%} >= {bench_threshold:.1%} (floor={bench_floor:.0%}, prev={prev_bench:.1%})")
                            writer.add_scalar("eval/promotion", 1.0, self.total_timesteps)
                        elif beats_best and not meets_bench:
                            print(f"Promotion BLOCKED - bench% too low")
                            print(f"  vs Best: {eval_vs_best:.1%} >= 30% (passed)")
                            print(f"  Bench%: {bench_rate_eval:.1%} < {bench_threshold:.1%} (failed)")
                            writer.add_scalar("eval/promotion", 0.0, self.total_timesteps)
                        else:
                            writer.add_scalar("eval/promotion", 0.0, self.total_timesteps)

                # Check memory usage
                mem_status, mem_percent = memory_monitor.check_memory()
                if mem_status == "hard_limit":
                    print(f"\nMemory critical ({mem_percent:.1f}%), emergency shutdown!")
                    self.save_checkpoint(save_path / "latest.pt")
                    break
                elif mem_status == "soft_limit":
                    print(f"\nMemory high ({mem_percent:.1f}%), graceful shutdown...")
                    self.save_checkpoint(save_path / "latest.pt")
                    break

        finally:
            display.close()
            self.stop_workers()
            self.save_checkpoint(save_path / "latest.pt")
            self.self_play_manager.opponent_pool.save_metadata()
            writer.close()

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Total episodes: {self.total_episodes:,}")
        print(f"Best Bench rate: {self.best_bench_rate:.1%}")
        print(f"{'='*60}")

    def save_checkpoint(self, path: Path):
        """Save checkpoint compatible with main trainer."""
        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.ppo.optimizer.state_dict(),
            "stats": {
                "total_timesteps": self.total_timesteps,
                "total_episodes": self.total_episodes,
                "total_updates": self.total_updates,
                "best_win_rate": self.best_bench_rate,
                "last_bench_rate": self.last_bench_rate,
                "prev_best_bench": self.prev_best_bench,  # For bench-gated promotion
            },
            "self_play": {
                "agent_skill": self.self_play_manager.agent_skill,
                "games_vs_self_play": self.self_play_manager.games_vs_self_play,
                "games_vs_max_damage": self.self_play_manager.games_vs_max_damage,
                "games_vs_random": self.self_play_manager.games_vs_random,
                "results_self_play": list(self.self_play_manager._results_self_play),
                "results_max_damage": list(self.self_play_manager._results_max_damage),
                "results_random": list(self.self_play_manager._results_random),
                "rolling_results": list(self._rolling_results),
            },
            "config": {
                "num_workers": self.num_workers,
                "envs_per_worker": self.envs_per_worker,
            },
        }
        torch.save(checkpoint, path)
        print(f"Saved: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint (compatible with main trainer checkpoints)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.network.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        stats = checkpoint.get("stats", {})
        self.total_timesteps = stats.get("total_timesteps", 0)
        self.total_episodes = stats.get("total_episodes", 0)
        self.total_updates = stats.get("total_updates", 0)
        self.best_bench_rate = stats.get("best_win_rate", 0.0)
        self.last_bench_rate = stats.get("last_bench_rate")
        self.prev_best_bench = stats.get("prev_best_bench")  # For bench-gated promotion

        # Load self-play state
        sp_state = checkpoint.get("self_play", {})
        if sp_state:
            self.self_play_manager.agent_skill = sp_state.get("agent_skill", 1000.0)
            self.self_play_manager.games_vs_self_play = sp_state.get("games_vs_self_play", 0)
            self.self_play_manager.games_vs_max_damage = sp_state.get("games_vs_max_damage", 0)
            self.self_play_manager.games_vs_random = sp_state.get("games_vs_random", 0)

            # Restore rolling results
            if "results_max_damage" in sp_state:
                self.self_play_manager._results_max_damage = deque(
                    sp_state["results_max_damage"],
                    maxlen=self.self_play_manager._rolling_window,
                )
            if "results_self_play" in sp_state:
                self.self_play_manager._results_self_play = deque(
                    sp_state["results_self_play"],
                    maxlen=self.self_play_manager._rolling_window,
                )
            if "rolling_results" in sp_state:
                self._rolling_results = deque(
                    sp_state["rolling_results"],
                    maxlen=100,
                )

        print(f"Loaded: {path}")
        print(f"  Steps: {self.total_timesteps:,}")
        print(f"  Bench rate: {self.best_bench_rate:.1%}")


def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Multi-process training with self-play")
    parser.add_argument("--workers", "-w", type=int, default=2,
                        help="Number of worker processes (default: 2)")
    parser.add_argument("--envs-per-worker", "-e", type=int, default=6,
                        help="Environments per worker (default: 6)")
    parser.add_argument("--server-ports", type=int, nargs="+", default=[8000, 8001],
                        help="Server ports (one per worker recommended)")
    parser.add_argument("--timesteps", "-t", type=int, default=5_000_000,
                        help="Total timesteps to train (default: 5M)")
    parser.add_argument("--resume", "-r", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--save-dir", type=str, default="data/checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--self-play-start", action="store_true",
                        help="Start with self-play instead of MaxDamage (copies initial model as best)")
    parser.add_argument("--entropy-coef", type=float, default=0.05,
                        help="Entropy coefficient for exploration (default: 0.05)")
    parser.add_argument("--benchmark-games", type=int, default=50,
                        help="Number of MaxDamage benchmark games per checkpoint (default: 50)")

    args = parser.parse_args()

    # Validate workers vs ports
    if len(args.server_ports) < args.workers:
        print(f"Warning: {args.workers} workers but only {len(args.server_ports)} ports.")
        print("For best performance, use one port per worker.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = TrainingConfig()
    # Override entropy_coef if specified via CLI
    config.entropy_coef = args.entropy_coef
    print(f"Entropy coef: {config.entropy_coef}")
    print(f"Benchmark games: {args.benchmark_games}")

    trainer = MultiProcessTrainer(
        num_workers=args.workers,
        envs_per_worker=args.envs_per_worker,
        server_ports=args.server_ports,
        device=device,
        config=config,
        benchmark_games=args.benchmark_games,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # If --self-play-start is set and not resuming, create best_model.pt from initial weights
    # This makes training start as "current vs current" instead of "current vs MaxDamage"
    if args.self_play_start and not args.resume:
        best_path = Path(args.save_dir) / "best_model.pt"
        if not best_path.exists():
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": trainer.network.state_dict(),
            }, best_path)
            print(f"Self-play start: Created initial best_model.pt")
            print(f"Training will be: Current vs Current (not MaxDamage)")
        else:
            print(f"Warning: best_model.pt already exists, --self-play-start has no effect")

    trainer.train(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()

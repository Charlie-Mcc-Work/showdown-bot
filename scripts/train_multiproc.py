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
        """Player that collects experiences for this worker."""

        def __init__(self, model, state_encoder, dev, **kwargs):
            super().__init__(**kwargs)
            self.model = model
            self.state_encoder = state_encoder
            self.device = dev
            self.current_experiences = []
            self.prev_hp_fraction = None
            self.prev_opp_hp_fraction = None
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

            reward = calculate_reward(battle, self.prev_hp_fraction, self.prev_opp_hp_fraction)

            our_remaining = [p for p in battle.team.values() if not p.fainted]
            opp_remaining = [p for p in battle.opponent_team.values() if not p.fainted]
            self.prev_hp_fraction = sum(p.current_hp_fraction for p in our_remaining) / max(1, len(our_remaining))
            self.prev_opp_hp_fraction = sum(p.current_hp_fraction for p in opp_remaining) / max(1, len(opp_remaining))

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
                self.current_experiences[-1]["reward"] = final_reward
                self.current_experiences[-1]["done"] = True

            self.prev_hp_fraction = None
            self.prev_opp_hp_fraction = None

        def get_experiences(self):
            exps = self.current_experiences.copy()
            self.current_experiences = []
            return exps

        def reset(self):
            self.current_experiences = []
            self.prev_hp_fraction = None
            self.prev_opp_hp_fraction = None
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
            if opp_type == "max_damage":
                opponent = MaxDamagePlayer(
                    battle_format=battle_format,
                    max_concurrent_battles=1,
                    server_configuration=server_config,
                )
            elif opp_type == "self_play" and opponent_weights is not None:
                opponent = SelfPlayOpponent(
                    model=opponent_network,
                    state_encoder=encoder,
                    dev=device,
                    battle_format=battle_format,
                    max_concurrent_battles=1,
                    server_configuration=server_config,
                )
            else:
                # Fallback to max_damage if no self-play weights
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
    ):
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.server_ports = server_ports
        self.device = device
        self.config = config

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
        """Get opponent types based on curriculum."""
        self_play_ratio, max_damage_ratio, _ = self.self_play_manager.get_curriculum_ratios()

        types = []
        total_envs = self.num_workers * self.envs_per_worker

        # Distribute opponent types across all envs
        num_self_play = int(total_envs * self_play_ratio)
        num_max_damage = total_envs - num_self_play

        types.extend(["self_play"] * num_self_play)
        types.extend(["max_damage"] * num_max_damage)

        # Shuffle to distribute evenly
        import random
        random.shuffle(types)

        return types

    def get_self_play_opponent_weights(self) -> dict | None:
        """Get weights for self-play opponent from pool."""
        if self.self_play_manager.opponent_pool.size == 0:
            return None

        # Sample an opponent from the pool
        opponent_info = self.self_play_manager.opponent_pool.sample_opponent(
            strategy="diverse",
            current_skill=self.self_play_manager.agent_skill,
        )
        if opponent_info is None:
            return None

        # Load weights from checkpoint file
        checkpoint = torch.load(
            opponent_info.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        return checkpoint

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

    def train(
        self,
        total_timesteps: int,
        save_dir: str = "data/checkpoints",
        log_dir: str = "runs",
    ):
        """Main training loop."""
        from torch.utils.tensorboard import SummaryWriter
        from showdown_bot.training.trainer import TrainingDisplay

        save_path = Path(save_dir)
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
        print(f"Self-play: Enabled")
        print(f"{'='*60}\n")

        self.start_workers()

        # Create display
        display = TrainingDisplay(
            total_timesteps=target_timesteps,
            start_timesteps=self.total_timesteps,
        )
        display.initialize()

        last_checkpoint_timesteps = self.total_timesteps

        try:
            while self.total_timesteps < target_timesteps and not self._shutdown_requested:
                # Reset buffer
                self.buffer.reset()

                # Get opponent configuration from curriculum
                opponent_types = self.get_opponent_types()
                opponent_weights = self.get_self_play_opponent_weights()

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

                # Maybe add to opponent pool
                if self.self_play_manager.should_add_checkpoint(self.total_timesteps):
                    self.self_play_manager.add_checkpoint(
                        self.network,
                        self.total_timesteps,
                    )

                # Get stats for display
                sp_stats = self.self_play_manager.get_stats()
                bench_rate = sp_stats.get("rolling_win_rate_max_damage")
                win_rates = self.self_play_manager.get_rolling_win_rates()
                overall_win_rate = 0.0
                total_games = 0
                for opp_type, rate in win_rates.items():
                    if rate is not None:
                        overall_win_rate += rate
                        total_games += 1
                if total_games > 0:
                    overall_win_rate /= total_games

                # Update best bench rate
                if bench_rate is not None and bench_rate > self.best_bench_rate:
                    self.best_bench_rate = bench_rate

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
                    writer.add_scalar("rollout/win_rate", overall_win_rate, self.total_timesteps)
                    if bench_rate is not None:
                        writer.add_scalar("rollout/bench_win_rate", bench_rate, self.total_timesteps)

                # Save checkpoint periodically
                if self.total_timesteps - last_checkpoint_timesteps >= 50000:
                    self.save_checkpoint(save_path / "latest.pt")
                    last_checkpoint_timesteps = self.total_timesteps

                    # Save best model
                    if bench_rate is not None and bench_rate >= self.best_bench_rate:
                        self.save_checkpoint(save_path / "best_model.pt")

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
            },
            "self_play": {
                "agent_skill": self.self_play_manager.agent_skill,
                "games_vs_self_play": self.self_play_manager.games_vs_self_play,
                "games_vs_max_damage": self.self_play_manager.games_vs_max_damage,
                "games_vs_random": self.self_play_manager.games_vs_random,
                "results_self_play": list(self.self_play_manager._results_self_play),
                "results_max_damage": list(self.self_play_manager._results_max_damage),
                "results_random": list(self.self_play_manager._results_random),
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

    args = parser.parse_args()

    # Validate workers vs ports
    if len(args.server_ports) < args.workers:
        print(f"Warning: {args.workers} workers but only {len(args.server_ports)} ports.")
        print("For best performance, use one port per worker.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = TrainingConfig()

    trainer = MultiProcessTrainer(
        num_workers=args.workers,
        envs_per_worker=args.envs_per_worker,
        server_ports=args.server_ports,
        device=device,
        config=config,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()

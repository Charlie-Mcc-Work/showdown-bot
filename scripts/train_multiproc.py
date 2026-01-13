#!/usr/bin/env python3
"""Multi-process training script that bypasses Python's GIL.

Architecture:
- Main process: Owns the model, does PPO updates
- Worker processes: Run battles, collect experiences, return them to main

This allows true parallelism across CPU cores.

Usage:
    # Start servers first
    for i in {0..2}; do
        node ~/pokemon-showdown/pokemon-showdown start --no-security --port $((8000+i)) &
    done

    # Run multi-process training
    python scripts/train_multiproc.py --workers 3 --envs-per-worker 4 --server-ports 8000 8001 8002
"""

import argparse
import asyncio
import multiprocessing as mp
import os
import signal
import sys
import time
import warnings
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
    server_ports: list[int],
    num_envs: int,
    weight_queue: mp.Queue,
    experience_queue: mp.Queue,
    control_queue: mp.Queue,
    device_str: str,
):
    """Worker process that runs battles and collects experiences.

    This runs in a separate process, bypassing the GIL.
    """
    # Delay imports to avoid issues with multiprocessing
    import torch
    from torch.amp import autocast
    from poke_env.player import Player, RandomPlayer
    from poke_env.battle import AbstractBattle
    from poke_env.ps_client import ServerConfiguration

    from showdown_bot.models.network import PolicyValueNetwork
    from showdown_bot.environment.state_encoder import StateEncoder
    from showdown_bot.training.trainer import calculate_reward

    print(f"[Worker {worker_id}] Starting on ports {server_ports} with {num_envs} envs")

    device = torch.device(device_str)

    # Create network and encoder
    encoder = StateEncoder(device=device)
    network = PolicyValueNetwork.from_config().to(device)
    network.eval()

    # Server configs
    server_configs = [
        ServerConfiguration(
            websocket_url=f"ws://localhost:{port}/showdown/websocket",
            authentication_url="https://play.pokemonshowdown.com/action.php?",
        )
        for port in server_ports
    ]

    # Define the trainable player class inline to avoid import issues
    class WorkerTrainablePlayer(Player):
        """Player that collects experiences for this worker."""

        def __init__(self, model, state_encoder, device, **kwargs):
            super().__init__(**kwargs)
            self.model = model
            self.state_encoder = state_encoder
            self.device = device
            self.current_experiences = []
            self.prev_hp_fraction = None
            self.prev_opp_hp_fraction = None
            self._won = None

        def choose_move(self, battle: AbstractBattle):
            state = self.state_encoder.encode_battle(battle)

            # Prepare tensors
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

            self.prev_hp_fraction = sum(
                p.current_hp_fraction for p in battle.team.values() if not p.fainted
            ) / 6
            self.prev_opp_hp_fraction = sum(
                p.current_hp_fraction for p in battle.opponent_team.values() if not p.fainted
            ) / 6

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

    async def run_battles(weights_dict):
        """Run battles and return experiences."""
        network.load_state_dict(weights_dict)

        players = []
        opponents = []
        for i in range(num_envs):
            server_config = server_configs[i % len(server_configs)]

            player = WorkerTrainablePlayer(
                model=network,
                state_encoder=encoder,
                device=device,
                battle_format="gen9randombattle",
                max_concurrent_battles=1,
                server_configuration=server_config,
            )
            players.append(player)

            opponent = RandomPlayer(
                battle_format="gen9randombattle",
                max_concurrent_battles=1,
                server_configuration=server_config,
            )
            opponents.append(opponent)

        async def run_single(player, opponent):
            try:
                await asyncio.wait_for(
                    player.battle_against(opponent, n_battles=1),
                    timeout=120.0
                )
                return player.get_experiences(), player._won
            except Exception as e:
                print(f"[Worker {worker_id}] Battle error: {e}")
                return [], None

        tasks = [run_single(p, o) for p, o in zip(players, opponents)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_experiences = []
        wins = 0
        losses = 0

        for result in results:
            if isinstance(result, Exception):
                continue
            exps, won = result
            all_experiences.extend(exps)
            if won is True:
                wins += 1
            elif won is False:
                losses += 1

        # Cleanup
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
            "wins": wins,
            "losses": losses,
        }

    # Main worker loop
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while running:
        try:
            # Check control queue
            try:
                msg = control_queue.get_nowait()
                if msg == "stop":
                    break
            except Empty:
                pass

            # Wait for weights
            try:
                weights = weight_queue.get(timeout=1.0)
            except Empty:
                continue

            if weights is None:
                break

            # Run battles in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_battles(weights))
                experience_queue.put(result)
            finally:
                loop.close()

        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"[Worker {worker_id}] Shutdown")


class MultiProcessTrainer:
    """Coordinates multi-process training."""

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
        from showdown_bot.environment.state_encoder import StateEncoder
        from showdown_bot.training.ppo import PPO
        from showdown_bot.training.buffer import RolloutBuffer

        self.encoder = StateEncoder(device=device)
        self.network = PolicyValueNetwork.from_config().to(device)

        # Create buffer and PPO trainer
        # Use num_envs=1 since experiences are added one at a time
        # from the shared experience queue (same pattern as trainer.py)
        self.buffer = RolloutBuffer(
            buffer_size=config.rollout_steps,
            num_envs=1,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=device,
        )

        self.ppo = PPO.from_config(self.network, device=device, config=config)

        # Stats
        self.total_timesteps = 0
        self.total_episodes = 0
        self.total_wins = 0
        self.total_losses = 0

        # Process management
        self.workers = []
        self.weight_queues = []
        self.experience_queue = None
        self.control_queues = []
        self._shutdown_requested = False

    def start_workers(self):
        """Start worker processes."""
        self.experience_queue = mp.Queue()

        ports_per_worker = max(1, len(self.server_ports) // self.num_workers)

        for i in range(self.num_workers):
            start_idx = i * ports_per_worker
            end_idx = start_idx + ports_per_worker
            if i == self.num_workers - 1:
                worker_ports = self.server_ports[start_idx:]
            else:
                worker_ports = self.server_ports[start_idx:end_idx]

            if not worker_ports:
                worker_ports = [self.server_ports[i % len(self.server_ports)]]

            weight_queue = mp.Queue()
            control_queue = mp.Queue()

            worker = mp.Process(
                target=worker_process,
                args=(
                    i,
                    worker_ports,
                    self.envs_per_worker,
                    weight_queue,
                    self.experience_queue,
                    control_queue,
                    str(self.device),
                ),
            )
            worker.start()

            self.workers.append(worker)
            self.weight_queues.append(weight_queue)
            self.control_queues.append(control_queue)

        print(f"Started {self.num_workers} workers")
        time.sleep(2)  # Let workers initialize

    def stop_workers(self):
        """Stop all workers."""
        print("Stopping workers...")
        for q in self.control_queues:
            try:
                q.put("stop")
            except:
                pass

        for q in self.weight_queues:
            try:
                q.put(None)
            except:
                pass

        for w in self.workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()

        print("All workers stopped")

    def broadcast_weights(self):
        """Send current model weights to all workers."""
        weights = {k: v.cpu() for k, v in self.network.state_dict().items()}
        for q in self.weight_queues:
            q.put(weights)

    def collect_experiences(self, timeout: float = 300.0) -> list:
        """Collect experiences from all workers."""
        results = []
        deadline = time.time() + timeout

        while len(results) < self.num_workers and time.time() < deadline:
            if self._shutdown_requested:
                break
            try:
                result = self.experience_queue.get(timeout=1.0)
                results.append(result)
            except Empty:
                continue

        return results

    def train(self, total_timesteps: int, save_dir: str = "data/checkpoints", log_dir: str = "runs"):
        """Main training loop."""
        from torch.utils.tensorboard import SummaryWriter

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)

        def handle_signal(signum, frame):
            print("\nShutdown requested...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        total_envs = self.num_workers * self.envs_per_worker
        print(f"\n{'='*60}")
        print("Multi-Process Training")
        print(f"{'='*60}")
        print(f"Workers: {self.num_workers}")
        print(f"Envs per worker: {self.envs_per_worker}")
        print(f"Total parallel envs: {total_envs}")
        print(f"Server ports: {self.server_ports}")
        print(f"Target timesteps: {total_timesteps:,}")
        print(f"{'='*60}\n")

        self.start_workers()
        start_time = time.time()
        last_log_time = start_time
        last_save_steps = 0

        try:
            while self.total_timesteps < total_timesteps and not self._shutdown_requested:
                # Broadcast weights
                self.broadcast_weights()

                # Collect experiences
                results = self.collect_experiences()
                if not results:
                    print("No results, retrying...")
                    continue

                # Aggregate
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

                    self.total_timesteps += result["steps"]
                    self.total_wins += result["wins"]
                    self.total_losses += result["losses"]
                    self.total_episodes += result["wins"] + result["losses"]

                # PPO update when buffer is full (rollout_steps experiences collected)
                if self.buffer.ptr >= self.config.rollout_steps:
                    self.network.train()
                    ppo_stats = self.ppo.update(self.buffer, self.config.batch_size)
                    self.network.eval()
                    self.buffer.reset()

                    writer.add_scalar("train/policy_loss", ppo_stats.policy_loss, self.total_timesteps)
                    writer.add_scalar("train/value_loss", ppo_stats.value_loss, self.total_timesteps)
                    writer.add_scalar("train/entropy", ppo_stats.entropy_loss, self.total_timesteps)

                # Logging
                now = time.time()
                if now - last_log_time >= 10:
                    elapsed = now - start_time
                    steps_per_sec = self.total_timesteps / elapsed
                    win_rate = self.total_wins / max(self.total_episodes, 1)

                    print(
                        f"Steps: {self.total_timesteps:,} | "
                        f"Episodes: {self.total_episodes:,} | "
                        f"Win rate: {win_rate:.1%} | "
                        f"Speed: {steps_per_sec:.1f} steps/s"
                    )

                    writer.add_scalar("train/win_rate", win_rate, self.total_timesteps)
                    writer.add_scalar("train/steps_per_sec", steps_per_sec, self.total_timesteps)
                    last_log_time = now

                # Save periodically
                if self.total_timesteps - last_save_steps >= 50000:
                    self.save_checkpoint(save_path / "latest.pt")
                    last_save_steps = self.total_timesteps

        finally:
            self.stop_workers()
            self.save_checkpoint(save_path / "latest.pt")
            writer.close()

        print(f"\nTraining complete!")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Final win rate: {self.total_wins / max(self.total_episodes, 1):.1%}")

    def save_checkpoint(self, path: Path):
        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.ppo.optimizer.state_dict(),
            "stats": {
                "total_timesteps": self.total_timesteps,
                "total_episodes": self.total_episodes,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
            },
        }
        torch.save(checkpoint, path)
        print(f"Saved: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        stats = checkpoint.get("stats", {})
        self.total_timesteps = stats.get("total_timesteps", 0)
        self.total_episodes = stats.get("total_episodes", 0)
        self.total_wins = stats.get("total_wins", 0)
        self.total_losses = stats.get("total_losses", 0)
        print(f"Loaded: {path} (Steps: {self.total_timesteps:,})")


def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Multi-process training")
    parser.add_argument("--workers", "-w", type=int, default=3)
    parser.add_argument("--envs-per-worker", "-e", type=int, default=4)
    parser.add_argument("--server-ports", type=int, nargs="+", default=[8000, 8001, 8002])
    parser.add_argument("--timesteps", "-t", type=int, default=1_000_000)
    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="data/checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")

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

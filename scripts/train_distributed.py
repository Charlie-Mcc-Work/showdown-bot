#!/usr/bin/env python3
"""Distributed training script using PyTorch DDP for gradient sharing across workers."""

import argparse
import asyncio
import os
import sys
import warnings
from pathlib import Path

# Suppress ROCm/PyTorch experimental feature warnings
warnings.filterwarnings("ignore", message=".*hipBLASLt.*")
warnings.filterwarnings("ignore", message=".*Flash attention.*experimental.*")
warnings.filterwarnings("ignore", message=".*Memory Efficient attention.*experimental.*")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast

from showdown_bot.config import training_config
from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.training.trainer import Trainer
from showdown_bot.training.ppo import PPO, PPOStats
from showdown_bot.training.buffer import RolloutBuffer
import torch.nn.functional as F


class DistributedPPO(PPO):
    """PPO with gradient synchronization across distributed workers.

    All workers compute gradients independently, then we average them
    before the optimizer step so all models stay in sync.
    """

    def __init__(self, *args, world_size: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_size = world_size

    def _sync_gradients(self) -> None:
        """Average gradients across all workers."""
        if self.world_size <= 1 or not dist.is_initialized():
            return

        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size

    def update(self, buffer: RolloutBuffer, batch_size: int) -> PPOStats:
        """Perform a PPO update with distributed gradient synchronization."""
        # Handle empty buffer
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
                with autocast("cuda", enabled=self.use_amp):
                    _, log_probs, entropy, values = self.model.get_action_and_value(
                        player_pokemon=batch["player_pokemon"],
                        opponent_pokemon=batch["opponent_pokemon"],
                        player_active_idx=batch["player_active_idx"],
                        opponent_active_idx=batch["opponent_active_idx"],
                        field_state=batch["field_state"],
                        action_mask=batch["action_mask"],
                        action=batch["actions"],
                    )

                    log_ratio = log_probs - batch["old_log_probs"]
                    ratio = torch.exp(log_ratio)

                    advantages = batch["advantages"]
                    surrogate1 = ratio * advantages
                    surrogate2 = torch.clamp(
                        ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                    ) * advantages
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()

                    values_clipped = batch["old_values"] + torch.clamp(
                        values - batch["old_values"],
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    value_loss1 = F.mse_loss(values, batch["returns"])
                    value_loss2 = F.mse_loss(values_clipped, batch["returns"])
                    value_loss = torch.max(value_loss1, value_loss2)

                    entropy_loss = -entropy.mean()

                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss
                    )

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # DISTRIBUTED: Sync gradients across all workers before stepping
                self._sync_gradients()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

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

            if self.target_kl is not None and total_approx_kl / num_updates > self.target_kl:
                break

        # Compute explained variance
        with torch.no_grad():
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


def setup_distributed():
    """Initialize distributed training from environment variables set by torchrun."""
    if "RANK" not in os.environ:
        # Not running with torchrun, single process mode
        return 0, 1, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize process group
    # Use gloo backend - works for both CPU and single-GPU setups
    # nccl requires multiple GPUs (one per process)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= world_size:
        backend = "nccl"
        torch.cuda.set_device(local_rank)
    else:
        # Single GPU or CPU - use gloo for gradient sync
        backend = "gloo"
        if num_gpus > 0:
            # All workers share the single GPU
            torch.cuda.set_device(0)

    dist.init_process_group(backend=backend)

    return rank, world_size, True


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def find_latest_checkpoint(save_dir: str) -> Path | None:
    """Find the latest checkpoint in save_dir."""
    save_path = Path(save_dir)
    if not save_path.exists():
        return None

    # Check for best_model.pt first, then latest.pt
    best = save_path / "best_model.pt"
    if best.exists():
        return best

    latest = save_path / "latest.pt"
    if latest.exists():
        return latest

    # Find checkpoint with highest timestep
    checkpoints = list(save_path.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    def get_timestep(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    return max(checkpoints, key=get_timestep)


class DistributedTrainer(Trainer):
    """Extended trainer with distributed gradient sync support."""

    def __init__(self, *args, rank: int = 0, world_size: int = 1, **kwargs):
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        super().__init__(*args, **kwargs)

        # Replace standard PPO with distributed version that syncs gradients
        if world_size > 1:
            self.ppo = DistributedPPO(
                model=self.model,
                learning_rate=self.config.learning_rate,
                clip_epsilon=self.config.clip_epsilon,
                value_coef=self.config.value_coef,
                entropy_coef=self.config.entropy_coef,
                max_grad_norm=self.config.max_grad_norm,
                num_epochs=self.config.num_epochs,
                device=self.device,
                world_size=world_size,
            )

    def _save_checkpoint(self, filename: str = "latest.pt") -> None:
        """Only main process saves checkpoints."""
        if self.is_main:
            super()._save_checkpoint(filename)

    def _log_training(self, rollout_stats: dict, ppo_stats: dict) -> None:
        """Only main process logs."""
        if self.is_main:
            super()._log_training(rollout_stats, ppo_stats)


async def train(args: argparse.Namespace) -> None:
    """Main distributed training function."""
    rank, world_size, is_distributed = setup_distributed()
    is_main = rank == 0

    try:
        # Device setup
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if args.device == "auto":
            if num_gpus > 0:
                # All workers use GPU 0 if single GPU, else each gets their own
                gpu_id = rank if num_gpus >= world_size else 0
                device = torch.device("cuda", gpu_id)
                if is_main:
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                    if num_gpus < world_size:
                        print(f"  (Single GPU shared by {world_size} workers via gloo)")
            else:
                device = torch.device("cpu")
                if is_main:
                    print("Using CPU")
        else:
            device = torch.device(args.device)
            if is_main:
                print(f"Using {args.device.upper()}")

        if is_main:
            print(f"\nDistributed training: {world_size} workers")
            print(f"Each worker: {args.num_envs} environments")
            print(f"Total parallel battles: {world_size * args.num_envs}")

        # Create model
        if is_main:
            print("\nInitializing neural network...")
        model = PolicyValueNetwork.from_config().to(device)

        if is_main:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {num_params:,}")

        # Distribute server ports across workers
        if args.server_ports:
            # Each worker gets a subset of ports
            ports_per_worker = len(args.server_ports) // world_size
            start_idx = rank * ports_per_worker
            end_idx = start_idx + ports_per_worker
            worker_ports = args.server_ports[start_idx:end_idx]
            if not worker_ports:
                worker_ports = [args.server_ports[rank % len(args.server_ports)]]
        else:
            worker_ports = None

        if is_main and args.server_ports:
            print(f"Worker {rank} using ports: {worker_ports}")

        # Create trainer with distributed gradient sync
        trainer = DistributedTrainer(
            model=model,
            device=device,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            self_play_dir=args.self_play_dir,
            use_self_play=not args.no_self_play,
            num_envs=args.num_envs,
            server_ports=worker_ports,
            rank=rank,
            world_size=world_size,
        )

        # Load checkpoint (all workers load same checkpoint to start in sync)
        if args.resume:
            if args.resume == "auto":
                checkpoint_path = find_latest_checkpoint(args.save_dir)
                if checkpoint_path:
                    if is_main:
                        print(f"\nAuto-resuming from: {checkpoint_path}")
                    trainer.load_checkpoint(str(checkpoint_path))
                elif is_main:
                    print("\nNo checkpoint found, starting fresh training")
            else:
                trainer.load_checkpoint(args.resume)

        # Synchronize before starting training
        if is_distributed:
            dist.barrier()

        # Print startup info (only main worker)
        if is_main:
            print("\n" + "=" * 60)
            print("NOTE: Training requires Pokemon Showdown servers running locally.")
            if args.server_ports:
                print(f"Using {len(args.server_ports)} servers on ports: {args.server_ports}")
            print("=" * 60 + "\n")

        # Start training
        await trainer.train(
            total_timesteps=args.timesteps,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
        )

    finally:
        cleanup_distributed()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Distributed training for Pokemon Showdown RL bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timesteps", "-t", type=int,
        default=training_config.total_timesteps,
        help="Total timesteps to train for",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default=training_config.save_dir,
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--log-dir", type=str,
        default=training_config.log_dir,
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--self-play-dir", type=str,
        default="data/opponents",
        help="Directory for self-play opponent pool",
    )
    parser.add_argument(
        "--resume", "-r", type=str, nargs="?", const="auto", default=None,
        help="Resume from checkpoint (auto or path)",
    )
    parser.add_argument(
        "--eval-interval", type=int,
        default=training_config.eval_interval,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--save-interval", type=int,
        default=training_config.checkpoint_interval,
        help="Steps between saving checkpoints",
    )
    parser.add_argument(
        "--no-self-play", action="store_true",
        help="Disable self-play training",
    )
    parser.add_argument(
        "--num-envs", "-e", type=int,
        default=training_config.num_envs,
        help="Number of parallel environments PER WORKER",
    )
    parser.add_argument(
        "--server-ports", type=int, nargs="+", default=None,
        help="Pokemon Showdown server ports (distributed across workers)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training",
    )

    args = parser.parse_args()
    asyncio.run(train(args))


if __name__ == "__main__":
    main()

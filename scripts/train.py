#!/usr/bin/env python3
"""Main training script for the Pokemon Showdown RL bot."""

import argparse
import asyncio
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

from showdown_bot.config import training_config
from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.training.trainer import Trainer


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def find_latest_checkpoint(save_dir: str) -> Path | None:
    """Find the latest checkpoint in save_dir."""
    save_path = Path(save_dir)
    if not save_path.exists():
        return None

    # Check for latest.pt first
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


async def train(args: argparse.Namespace) -> None:
    """Main training function."""
    device = get_device()

    # Create model
    print("\nInitializing neural network...")
    model = PolicyValueNetwork.from_config().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        self_play_dir=args.self_play_dir,
        use_self_play=not args.no_self_play,
        num_envs=args.num_envs,
    )

    # Load checkpoint if provided
    if args.resume:
        if args.resume == "auto":
            # Find latest checkpoint
            checkpoint_path = find_latest_checkpoint(args.save_dir)
            if checkpoint_path:
                print(f"\nAuto-resuming from: {checkpoint_path}")
                trainer.load_checkpoint(str(checkpoint_path))
            else:
                print("\nNo checkpoint found, starting fresh training")
        else:
            # Use specified path
            trainer.load_checkpoint(args.resume)

    # Start training
    print("\n" + "=" * 60)
    print("NOTE: Training requires a Pokemon Showdown server running locally.")
    print("If not running, start it with:")
    print("  cd ~/pokemon-showdown && node pokemon-showdown start --no-security")
    print("=" * 60 + "\n")

    try:
        await trainer.train(
            total_timesteps=args.timesteps,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
        )
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nMake sure Pokemon Showdown is running locally:")
        print("  1. Clone: git clone https://github.com/smogon/pokemon-showdown.git")
        print("  2. cd pokemon-showdown && npm install")
        print("  3. node pokemon-showdown start --no-security")
        raise


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Train the Pokemon Showdown RL bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=training_config.total_timesteps,
        help="Total timesteps to train for",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=training_config.save_dir,
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=training_config.log_dir,
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--self-play-dir",
        type=str,
        default="data/opponents",
        help="Directory for self-play opponent pool",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Resume from checkpoint. Use without argument to auto-find latest, or specify path",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=training_config.eval_interval,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=training_config.checkpoint_interval,
        help="Steps between saving checkpoints",
    )
    parser.add_argument(
        "--no-self-play",
        action="store_true",
        help="Disable self-play training (train against random only)",
    )
    parser.add_argument(
        "--num-envs",
        "-e",
        type=int,
        default=training_config.num_envs,
        help="Number of parallel environments for faster training",
    )

    args = parser.parse_args()
    asyncio.run(train(args))


if __name__ == "__main__":
    main()

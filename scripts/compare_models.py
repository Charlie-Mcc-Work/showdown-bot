#!/usr/bin/env python3
"""Compare two models by having them battle each other."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from showdown_bot.config import training_config
from showdown_bot.environment.battle_env import NeuralNetworkPlayer
from showdown_bot.environment.state_encoder import StateEncoder
from showdown_bot.models.network import PolicyValueNetwork


def load_model(checkpoint_path: str, device: torch.device) -> PolicyValueNetwork:
    """Load a model from checkpoint."""
    model = PolicyValueNetwork.from_config().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint)

    model.eval()
    return model


async def compare_models(
    model1_path: str,
    model2_path: str,
    num_battles: int = 100,
    battle_format: str = "gen9randombattle",
) -> dict:
    """Have two models battle each other.

    Args:
        model1_path: Path to first model checkpoint
        model2_path: Path to second model checkpoint
        num_battles: Number of battles to run
        battle_format: Pokemon Showdown battle format

    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("Model vs Model Comparison")
    print("=" * 60)

    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load models
    print(f"\nModel 1: {model1_path}")
    model1 = load_model(model1_path, device)

    print(f"Model 2: {model2_path}")
    model2 = load_model(model2_path, device)

    # Create players
    player1 = NeuralNetworkPlayer(
        model=model1,
        state_encoder=StateEncoder(device=device),
        device=device,
        deterministic=True,
        battle_format=battle_format,
        max_concurrent_battles=1,
    )

    player2 = NeuralNetworkPlayer(
        model=model2,
        state_encoder=StateEncoder(device=device),
        device=device,
        deterministic=True,
        battle_format=battle_format,
        max_concurrent_battles=1,
    )

    print(f"\nRunning {num_battles} battles...")
    print("-" * 40)

    # Run battles
    await player1.battle_against(player2, n_battles=num_battles)

    # Calculate results
    model1_wins = player1.n_won_battles
    model2_wins = player2.n_won_battles
    model1_win_rate = model1_wins / num_battles
    model2_win_rate = model2_wins / num_battles

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Model 1 ({Path(model1_path).name}): {model1_wins} wins ({model1_win_rate:.1%})")
    print(f"Model 2 ({Path(model2_path).name}): {model2_wins} wins ({model2_win_rate:.1%})")
    print("-" * 40)

    if model1_wins > model2_wins:
        diff = model1_wins - model2_wins
        print(f"Model 1 wins by {diff} games!")
    elif model2_wins > model1_wins:
        diff = model2_wins - model1_wins
        print(f"Model 2 wins by {diff} games!")
    else:
        print("It's a tie!")

    print("=" * 60)

    return {
        "model1_path": model1_path,
        "model2_path": model2_path,
        "model1_wins": model1_wins,
        "model2_wins": model2_wins,
        "model1_win_rate": model1_win_rate,
        "model2_win_rate": model2_win_rate,
        "total_battles": num_battles,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare two trained models by having them battle"
    )
    parser.add_argument(
        "model1",
        type=str,
        help="Path to first model checkpoint",
    )
    parser.add_argument(
        "model2",
        type=str,
        help="Path to second model checkpoint",
    )
    parser.add_argument(
        "--battles",
        "-n",
        type=int,
        default=100,
        help="Number of battles to run (default: 100)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="gen9randombattle",
        help="Battle format (default: gen9randombattle)",
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.model1).exists():
        print(f"Error: Model 1 not found: {args.model1}")
        sys.exit(1)
    if not Path(args.model2).exists():
        print(f"Error: Model 2 not found: {args.model2}")
        sys.exit(1)

    asyncio.run(compare_models(
        args.model1,
        args.model2,
        num_battles=args.battles,
        battle_format=args.format,
    ))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluation script - test bot against various opponents."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from poke_env.player import RandomPlayer

from showdown_bot.config import training_config
from showdown_bot.environment.battle_env import NeuralNetworkPlayer, MaxDamagePlayer
from showdown_bot.environment.state_encoder import StateEncoder
from showdown_bot.models.network import PolicyValueNetwork


async def evaluate_model(
    checkpoint_path: str | None = None,
    num_battles: int = 100,
) -> dict[str, float]:
    """Evaluate the model against various baselines.

    Args:
        checkpoint_path: Path to model checkpoint (None for untrained)
        num_battles: Number of battles per opponent

    Returns:
        Dictionary of win rates against each opponent
    """
    print("=" * 60)
    print("Pokemon Showdown RL Bot - Evaluation")
    print("=" * 60)

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model = PolicyValueNetwork.from_config().to(device)
    if checkpoint_path:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("\nUsing untrained model (random initialization)")
    model.eval()

    # Create our player
    player = NeuralNetworkPlayer(
        model=model,
        state_encoder=StateEncoder(device=device),
        device=device,
        deterministic=True,
        battle_format=training_config.battle_format,
        max_concurrent_battles=1,
    )

    results = {}

    # Evaluate against Random
    print(f"\nEvaluating vs Random ({num_battles} battles)...")
    random_opponent = RandomPlayer(
        battle_format=training_config.battle_format,
        max_concurrent_battles=1,
    )
    await player.battle_against(random_opponent, n_battles=num_battles)
    results["vs_random"] = player.n_won_battles / num_battles
    print(f"  Win rate: {results['vs_random'] * 100:.1f}%")

    # Reset player stats
    player.reset_battles()

    # Evaluate against MaxDamage
    print(f"\nEvaluating vs MaxDamage ({num_battles} battles)...")
    maxdmg_opponent = MaxDamagePlayer(
        battle_format=training_config.battle_format,
        max_concurrent_battles=1,
    )
    await player.battle_against(maxdmg_opponent, n_battles=num_battles)
    results["vs_maxdamage"] = player.n_won_battles / num_battles
    print(f"  Win rate: {results['vs_maxdamage'] * 100:.1f}%")

    print("\n" + "=" * 60)
    print("Summary:")
    for opponent, win_rate in results.items():
        print(f"  {opponent}: {win_rate * 100:.1f}%")
    print("=" * 60)

    return results


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the Pokemon Showdown bot")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--battles",
        "-n",
        type=int,
        default=100,
        help="Number of battles per opponent",
    )
    args = parser.parse_args()

    asyncio.run(evaluate_model(args.checkpoint, args.battles))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactive play script - play against the trained bot."""

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


async def play_against_bot(checkpoint_path: str | None = None) -> None:
    """Play against the trained bot or a baseline."""
    print("=" * 60)
    print("Pokemon Showdown RL Bot - Interactive Play")
    print("=" * 60)

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if checkpoint_path:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        model = PolicyValueNetwork.from_config().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        player = NeuralNetworkPlayer(
            model=model,
            state_encoder=StateEncoder(device=device),
            device=device,
            deterministic=True,  # Use greedy action selection
            battle_format=training_config.battle_format,
            max_concurrent_battles=1,
        )
        print("Loaded trained model.")
    else:
        print("\nNo checkpoint provided, using MaxDamage heuristic player.")
        player = MaxDamagePlayer(
            battle_format=training_config.battle_format,
            max_concurrent_battles=1,
        )

    # Create a random opponent for testing
    opponent = RandomPlayer(
        battle_format=training_config.battle_format,
        max_concurrent_battles=1,
    )

    print(f"\nBattle format: {training_config.battle_format}")
    print("\nStarting battles...")

    # Run some test battles
    num_battles = 10
    await player.battle_against(opponent, n_battles=num_battles)

    print(f"\nResults after {num_battles} battles:")
    print(f"  Bot wins: {player.n_won_battles}")
    print(f"  Opponent wins: {opponent.n_won_battles}")
    print(f"  Win rate: {player.n_won_battles / num_battles * 100:.1f}%")


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Play against the Pokemon Showdown bot")
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
        default=10,
        help="Number of battles to play",
    )
    args = parser.parse_args()

    asyncio.run(play_against_bot(args.checkpoint))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Main training script for the Pokemon Showdown RL bot."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from poke_env.player import RandomPlayer
from tqdm import tqdm

from showdown_bot.config import training_config, server_config
from showdown_bot.environment.battle_env import NeuralNetworkPlayer, MaxDamagePlayer
from showdown_bot.environment.state_encoder import StateEncoder
from showdown_bot.models.network import PolicyValueNetwork


async def test_random_battles(
    player1: RandomPlayer,
    player2: RandomPlayer,
    num_battles: int = 10,
) -> tuple[int, int]:
    """Run test battles between two players.

    Returns:
        Tuple of (player1_wins, player2_wins)
    """
    await player1.battle_against(player2, n_battles=num_battles)

    p1_wins = player1.n_won_battles
    p2_wins = player2.n_won_battles

    return p1_wins, p2_wins


async def train_agent() -> None:
    """Main training loop."""
    print("=" * 60)
    print("Pokemon Showdown RL Bot - Training")
    print("=" * 60)

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create model
    print("\nInitializing neural network...")
    model = PolicyValueNetwork.from_config().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create state encoder
    state_encoder = StateEncoder(device=device)

    # Create players
    print("\nCreating players...")

    # Our trained agent
    agent = NeuralNetworkPlayer(
        model=model,
        state_encoder=state_encoder,
        device=device,
        battle_format=training_config.battle_format,
        max_concurrent_battles=1,
    )

    # Opponent (start with random, later use self-play)
    opponent = RandomPlayer(
        battle_format=training_config.battle_format,
        max_concurrent_battles=1,
    )

    print(f"\nBattle format: {training_config.battle_format}")
    print(f"Total training timesteps: {training_config.total_timesteps:,}")
    print("\n" + "=" * 60)
    print("NOTE: Training requires a Pokemon Showdown server running locally.")
    print("See: https://github.com/smogon/pokemon-showdown")
    print("=" * 60)

    # TODO: Implement full PPO training loop
    # For now, just test that battles can run
    print("\nRunning test battles to verify setup...")

    try:
        # Test with two random players first
        test_p1 = RandomPlayer(
            battle_format=training_config.battle_format,
            max_concurrent_battles=1,
        )
        test_p2 = RandomPlayer(
            battle_format=training_config.battle_format,
            max_concurrent_battles=1,
        )

        print("Testing random vs random (5 battles)...")
        p1_wins, p2_wins = await test_random_battles(test_p1, test_p2, num_battles=5)
        print(f"Results: Player 1 wins: {p1_wins}, Player 2 wins: {p2_wins}")

        print("\nSetup verified! Full training loop coming soon.")
        print("Next steps:")
        print("  1. Implement PPO algorithm")
        print("  2. Add experience buffer")
        print("  3. Implement self-play")
        print("  4. Add logging and checkpointing")

    except Exception as e:
        print(f"\nError during testing: {e}")
        print("\nMake sure Pokemon Showdown is running locally:")
        print("  1. Clone: git clone https://github.com/smogon/pokemon-showdown.git")
        print("  2. cd pokemon-showdown && npm install")
        print("  3. node pokemon-showdown start --no-security")
        raise


def main() -> None:
    """Entry point."""
    asyncio.run(train_agent())


if __name__ == "__main__":
    main()

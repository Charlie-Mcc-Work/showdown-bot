#!/usr/bin/env python3
"""Play against the trained bot in your browser."""

import argparse
import asyncio
import sys
import warnings
from pathlib import Path

# Suppress ROCm/PyTorch warnings
warnings.filterwarnings("ignore", message=".*hipBLASLt.*")
warnings.filterwarnings("ignore", message=".*Flash attention.*experimental.*")
warnings.filterwarnings("ignore", message=".*Memory Efficient attention.*experimental.*")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from poke_env import AccountConfiguration
from poke_env.player import Player

from showdown_bot.config import training_config
from showdown_bot.environment.state_encoder import StateEncoder
from showdown_bot.models.network import PolicyValueNetwork


class HumanChallengableBot(Player):
    """A bot that can be challenged by humans in the browser."""

    def __init__(
        self,
        model: PolicyValueNetwork,
        state_encoder: StateEncoder,
        device: torch.device,
        deterministic: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.state_encoder = state_encoder
        self.device = device
        self.deterministic = deterministic
        self.model.eval()

    def choose_move(self, battle):
        """Choose a move using the neural network."""
        state = self.state_encoder.encode_battle(battle)

        # Add batch dimension and move to device
        player_pokemon = state.player_pokemon.unsqueeze(0).to(self.device)
        opponent_pokemon = state.opponent_pokemon.unsqueeze(0).to(self.device)
        player_active_idx = torch.tensor([state.player_active_idx], device=self.device)
        opponent_active_idx = torch.tensor([state.opponent_active_idx], device=self.device)
        field_state = state.field_state.unsqueeze(0).to(self.device)
        action_mask = state.action_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.deterministic:
                # Greedy action selection
                policy, _ = self.model(
                    player_pokemon,
                    opponent_pokemon,
                    player_active_idx,
                    opponent_active_idx,
                    field_state,
                    action_mask,
                )
                action = policy.argmax(dim=-1).item()
            else:
                # Sample from policy
                action, _, _, _ = self.model.get_action_and_value(
                    player_pokemon,
                    opponent_pokemon,
                    player_active_idx,
                    opponent_active_idx,
                    field_state,
                    action_mask,
                )
                action = action.item()

        # Convert action to battle order
        order = self.state_encoder.action_to_battle_order(action, battle)
        if order:
            return order

        return self.choose_random_move(battle)


def find_best_checkpoint() -> Path | None:
    """Find the best model checkpoint."""
    checkpoints_dir = Path("data/checkpoints")

    # Check for best_model.pt first
    best = checkpoints_dir / "best_model.pt"
    if best.exists():
        return best

    # Fall back to latest.pt
    latest = checkpoints_dir / "latest.pt"
    if latest.exists():
        return latest

    return None


async def run_bot(
    checkpoint_path: str | None = None,
    bot_name: str = "TrainedBot",
    accept_all: bool = False,
    challenge_user: str | None = None,
) -> None:
    """Run the bot and accept challenges."""
    print("=" * 60)
    print("Pokemon Showdown RL Bot - Play Against Me!")
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

    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint()
        if checkpoint_path is None:
            print("\nNo checkpoint found! Train a model first with:")
            print("  python scripts/train.py -t 50000 --num-envs 4")
            return
        checkpoint_path = str(checkpoint_path)

    print(f"\nLoading model: {checkpoint_path}")

    # Load model
    model = PolicyValueNetwork.from_config().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both full checkpoints and model-only saves
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        stats = checkpoint.get("stats", {})
        print(f"  Timesteps trained: {stats.get('total_timesteps', 'unknown'):,}")
        print(f"  Best win rate: {stats.get('best_win_rate', 0) * 100:.1f}%")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create bot player
    bot = HumanChallengableBot(
        model=model,
        state_encoder=StateEncoder(device=device),
        device=device,
        deterministic=True,
        account_configuration=AccountConfiguration(bot_name, None),
        battle_format=training_config.battle_format,
        max_concurrent_battles=1,
    )

    print(f"\n{'=' * 60}")
    print(f"Bot is online as: {bot_name}")
    print(f"Battle format: {training_config.battle_format}")
    print(f"{'=' * 60}")
    print(f"\nTo play against the bot:")
    print(f"  1. Make sure the client HTTP server is running:")
    print(f"       cd ~/pokemon-showdown-client && npx http-server -p 8080 -c-1")
    print(f"  2. Open in your browser:")
    print(f"       http://localhost:8080/play.pokemonshowdown.com/testclient.html?~~localhost:8000")
    print(f"  3. Pick a username (any name works)")
    print(f"  4. Click 'Find a user' and search for: {bot_name}")
    print(f"  5. Click 'Challenge' and select '{training_config.battle_format}'")
    print(f"\nPress Ctrl+C to stop the bot")
    print(f"{'=' * 60}\n")

    try:
        if challenge_user:
            # Challenge a specific user
            print(f"Challenging {challenge_user}...")
            await bot.send_challenges(challenge_user, n_challenges=1)
        else:
            # Accept challenges continuously
            print("Waiting for challenges...")
            while True:
                try:
                    await bot.accept_challenges(None, 1)
                    print(f"\nBattle complete! Record: {bot.n_won_battles}W - {bot.n_finished_battles - bot.n_won_battles}L")
                    print("Waiting for next challenge...")
                except Exception as e:
                    if "connection" in str(e).lower():
                        print(f"Connection issue: {e}")
                        await asyncio.sleep(2)
                    else:
                        raise
    except KeyboardInterrupt:
        print(f"\n\nBot stopped.")
        print(f"Final record: {bot.n_won_battles}W - {bot.n_finished_battles - bot.n_won_battles}L")
        if bot.n_finished_battles > 0:
            print(f"Win rate: {bot.n_won_battles / bot.n_finished_battles * 100:.1f}%")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Run the trained bot for browser play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-finds best if not specified)",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="TrainedBot",
        help="Bot's username on Pokemon Showdown",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        default=None,
        help="Challenge a specific user instead of waiting for challenges",
    )
    args = parser.parse_args()

    asyncio.run(run_bot(
        checkpoint_path=args.checkpoint,
        bot_name=args.name,
        challenge_user=args.challenge,
    ))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Play against the bot as a human via terminal interface."""

import argparse
import asyncio
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*hipBLASLt.*")
warnings.filterwarnings("ignore", message=".*Flash attention.*experimental.*")
warnings.filterwarnings("ignore", message=".*Memory Efficient attention.*experimental.*")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from poke_env import AccountConfiguration
from poke_env.player import Player

from showdown_bot.config import training_config
from showdown_bot.environment.state_encoder import StateEncoder
from showdown_bot.models.network import PolicyValueNetwork


class HumanPlayer(Player):
    """A player controlled by human input in terminal."""

    def choose_move(self, battle):
        """Let the human choose a move."""
        print("\n" + "=" * 60)
        print(f"Turn {battle.turn}")
        print("=" * 60)

        # Show your active Pokemon
        active = battle.active_pokemon
        if active:
            hp_pct = active.current_hp_fraction * 100
            status = f" [{active.status.name}]" if active.status else ""
            print(f"\nYour active: {active.species} ({hp_pct:.0f}% HP){status}")
            if active.boosts:
                boosts = {k: v for k, v in active.boosts.items() if v != 0}
                if boosts:
                    print(f"  Boosts: {boosts}")

        # Show opponent's active Pokemon
        opp_active = battle.opponent_active_pokemon
        if opp_active:
            hp_pct = opp_active.current_hp_fraction * 100
            status = f" [{opp_active.status.name}]" if opp_active.status else ""
            print(f"Opponent: {opp_active.species} ({hp_pct:.0f}% HP){status}")

        # Show available moves
        print("\n--- MOVES ---")
        available_moves = list(battle.available_moves)
        for i, move in enumerate(available_moves):
            move_type = move.type.name if move.type else "???"
            pp = f"{move.current_pp}/{move.max_pp}" if move.current_pp is not None else "?/?"
            print(f"  {i + 1}. {move.id} (Type: {move_type}, PP: {pp}, Power: {move.base_power})")

        # Show available switches
        print("\n--- SWITCHES ---")
        available_switches = list(battle.available_switches)
        for i, pokemon in enumerate(available_switches):
            hp_pct = pokemon.current_hp_fraction * 100
            status = f" [{pokemon.status.name}]" if pokemon.status else ""
            print(f"  {i + 5}. {pokemon.species} ({hp_pct:.0f}% HP){status}")

        # Show team summary
        print("\n--- YOUR TEAM ---")
        for pokemon in battle.team.values():
            hp_pct = pokemon.current_hp_fraction * 100
            status = f" [{pokemon.status.name}]" if pokemon.status else ""
            active_marker = " (active)" if pokemon == active else ""
            fainted = " [FAINTED]" if pokemon.fainted else ""
            print(f"  {pokemon.species}: {hp_pct:.0f}% HP{status}{active_marker}{fainted}")

        # Get user input
        while True:
            try:
                print("\n" + "-" * 40)
                choice = input("Enter choice (1-4 for moves, 5-9 for switches): ").strip()

                if not choice:
                    continue

                idx = int(choice) - 1

                if 0 <= idx < len(available_moves):
                    return self.create_order(available_moves[idx])
                elif 4 <= idx < 4 + len(available_switches):
                    return self.create_order(available_switches[idx - 4])
                else:
                    print(f"Invalid choice. Enter 1-{len(available_moves)} for moves or 5-{4 + len(available_switches)} for switches.")
            except ValueError:
                print("Please enter a number.")
            except KeyboardInterrupt:
                print("\nForfeiting...")
                return self.choose_random_move(battle)


class BotPlayer(Player):
    """The trained bot player."""

    def __init__(self, model, state_encoder, device, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.state_encoder = state_encoder
        self.device = device
        self.model.eval()

    def choose_move(self, battle):
        state = self.state_encoder.encode_battle(battle)

        player_pokemon = state.player_pokemon.unsqueeze(0).to(self.device)
        opponent_pokemon = state.opponent_pokemon.unsqueeze(0).to(self.device)
        player_active_idx = torch.tensor([state.player_active_idx], device=self.device)
        opponent_active_idx = torch.tensor([state.opponent_active_idx], device=self.device)
        field_state = state.field_state.unsqueeze(0).to(self.device)
        action_mask = state.action_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, _ = self.model(
                player_pokemon, opponent_pokemon,
                player_active_idx, opponent_active_idx,
                field_state, action_mask,
            )
            action = policy.argmax(dim=-1).item()

        order = self.state_encoder.action_to_battle_order(action, battle)
        return order if order else self.choose_random_move(battle)


def find_best_checkpoint():
    checkpoints_dir = Path("data/checkpoints")
    for name in ["best_model.pt", "latest.pt"]:
        path = checkpoints_dir / name
        if path.exists():
            return path
    return None


async def main(checkpoint_path=None):
    print("=" * 60)
    print("Pokemon Showdown - Human vs Trained Bot")
    print("=" * 60)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Find checkpoint
    if not checkpoint_path:
        checkpoint_path = find_best_checkpoint()
        if not checkpoint_path:
            print("No checkpoint found! Train a model first.")
            return

    print(f"\nLoading bot model: {checkpoint_path}")

    # Load model
    model = PolicyValueNetwork.from_config().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        stats = checkpoint.get("stats", {})
        print(f"  Trained for: {stats.get('total_timesteps', '?'):,} steps")
        print(f"  Best win rate: {stats.get('best_win_rate', 0) * 100:.1f}%")
    else:
        model.load_state_dict(checkpoint)

    # Create players
    human = HumanPlayer(
        account_configuration=AccountConfiguration("Human", None),
        battle_format=training_config.battle_format,
        max_concurrent_battles=1,
    )

    bot = BotPlayer(
        model=model,
        state_encoder=StateEncoder(device=device),
        device=device,
        account_configuration=AccountConfiguration("TrainedBot", None),
        battle_format=training_config.battle_format,
        max_concurrent_battles=1,
    )

    print(f"\nBattle format: {training_config.battle_format}")
    print("\nStarting battle... (You are 'Human', bot is 'TrainedBot')")
    print("=" * 60)

    try:
        await human.battle_against(bot, n_battles=1)

        print("\n" + "=" * 60)
        if human.n_won_battles > 0:
            print("YOU WIN!")
        else:
            print("Bot wins!")
        print("=" * 60)

        # Play again?
        while True:
            again = input("\nPlay again? (y/n): ").strip().lower()
            if again == 'y':
                await human.battle_against(bot, n_battles=1)
                print("\n" + "=" * 60)
                print(f"Score: You {human.n_won_battles} - {bot.n_won_battles} Bot")
                print("=" * 60)
            elif again == 'n':
                break

    except KeyboardInterrupt:
        print("\n\nGoodbye!")

    print(f"\nFinal score: You {human.n_won_battles} - {bot.n_won_battles} Bot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against the trained bot in terminal")
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to checkpoint")
    args = parser.parse_args()
    asyncio.run(main(args.checkpoint))

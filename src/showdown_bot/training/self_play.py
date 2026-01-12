"""Self-play system for training against historical checkpoints."""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder
from poke_env.battle import AbstractBattle

from showdown_bot.config import training_config
from showdown_bot.environment.state_encoder import StateEncoder
from showdown_bot.models.network import PolicyValueNetwork


@dataclass
class OpponentInfo:
    """Information about an opponent in the pool."""

    checkpoint_path: Path
    skill_rating: float = 1000.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    timestep_added: int = 0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played


class HistoricalPlayer(Player):
    """A player that uses a historical checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path,
        model_class: type[PolicyValueNetwork],
        state_encoder: StateEncoder,
        device: torch.device,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.state_encoder = state_encoder
        self.device = device

        # Load model from checkpoint
        self.model = model_class.from_config().to(device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

    def _load_checkpoint(self, path: Path) -> None:
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Assume it's just the state dict
            self.model.load_state_dict(checkpoint)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose a move using the loaded model."""
        # Handle cleanup race condition - model may be deleted while websocket
        # still has pending messages
        if not hasattr(self, 'model') or self.model is None:
            return self.choose_random_move(battle)

        state = self.state_encoder.encode_battle(battle)

        player_pokemon = state.player_pokemon.unsqueeze(0).to(self.device)
        opponent_pokemon = state.opponent_pokemon.unsqueeze(0).to(self.device)
        player_active_idx = torch.tensor([state.player_active_idx], device=self.device)
        opponent_active_idx = torch.tensor([state.opponent_active_idx], device=self.device)
        field_state = state.field_state.unsqueeze(0).to(self.device)
        action_mask = state.action_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(
                player_pokemon,
                opponent_pokemon,
                player_active_idx,
                opponent_active_idx,
                field_state,
                action_mask,
            )
            action = logits.argmax(dim=-1).item()

        order = self.state_encoder.action_to_battle_order(action, battle)
        if order:
            return order

        return self.choose_random_move(battle)


class OpponentPool:
    """Pool of historical opponents for self-play training."""

    def __init__(
        self,
        pool_dir: Path,
        max_size: int = 10,
        device: torch.device | None = None,
    ):
        """Initialize opponent pool.

        Args:
            pool_dir: Directory to store opponent checkpoints
            max_size: Maximum number of opponents to keep
            device: Device for loading models
        """
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.device = device or torch.device("cpu")

        self.opponents: list[OpponentInfo] = []
        self.state_encoder = StateEncoder(device=self.device)

        # Load existing checkpoints
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        """Load existing checkpoints from the pool directory."""
        for checkpoint_path in sorted(self.pool_dir.glob("opponent_*.pt")):
            # Extract timestep from filename
            try:
                timestep = int(checkpoint_path.stem.split("_")[1])
            except (IndexError, ValueError):
                timestep = 0

            self.opponents.append(
                OpponentInfo(
                    checkpoint_path=checkpoint_path,
                    timestep_added=timestep,
                )
            )

    def add_opponent(
        self,
        model: PolicyValueNetwork,
        timestep: int,
        skill_rating: float = 1000.0,
    ) -> OpponentInfo:
        """Add a new opponent to the pool.

        Args:
            model: The model to save as an opponent
            timestep: Current training timestep
            elo_rating: Initial Elo rating

        Returns:
            The created OpponentInfo
        """
        # Save checkpoint
        checkpoint_path = self.pool_dir / f"opponent_{timestep}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        opponent_info = OpponentInfo(
            checkpoint_path=checkpoint_path,
            skill_rating=skill_rating,
            timestep_added=timestep,
        )
        self.opponents.append(opponent_info)

        # Remove oldest if over max size
        if len(self.opponents) > self.max_size:
            self._prune_pool()

        return opponent_info

    def _prune_pool(self) -> None:
        """Remove excess opponents, keeping diverse skill levels."""
        if len(self.opponents) <= self.max_size:
            return

        # Sort by skill rating
        sorted_opponents = sorted(self.opponents, key=lambda x: x.skill_rating)

        # Keep best, worst, and evenly distributed others
        to_keep = set()

        # Always keep the best and most recent
        to_keep.add(sorted_opponents[-1].checkpoint_path)
        to_keep.add(self.opponents[-1].checkpoint_path)

        # Keep evenly distributed by Elo
        step = len(sorted_opponents) / (self.max_size - 2)
        for i in range(self.max_size - 2):
            idx = int(i * step)
            to_keep.add(sorted_opponents[idx].checkpoint_path)

        # Remove opponents not in to_keep
        new_opponents = []
        for opponent in self.opponents:
            if opponent.checkpoint_path in to_keep:
                new_opponents.append(opponent)
            else:
                # Delete the checkpoint file
                opponent.checkpoint_path.unlink(missing_ok=True)

        self.opponents = new_opponents

    def sample_opponent(
        self,
        strategy: str = "uniform",
        current_skill: float = 1000.0,
    ) -> OpponentInfo | None:
        """Sample an opponent from the pool.

        Args:
            strategy: Sampling strategy ("uniform", "skill_matched", "prioritized")
            current_skill: Current agent's skill rating for matched sampling

        Returns:
            Selected opponent or None if pool is empty
        """
        if not self.opponents:
            return None

        if strategy == "uniform":
            return random.choice(self.opponents)

        elif strategy == "skill_matched":
            # Prefer opponents with similar skill
            weights = []
            for opp in self.opponents:
                skill_diff = abs(opp.skill_rating - current_skill)
                weight = 1.0 / (1.0 + skill_diff / 100.0)
                weights.append(weight)

            total = sum(weights)
            weights = [w / total for w in weights]
            return random.choices(self.opponents, weights=weights, k=1)[0]

        elif strategy == "prioritized":
            # Prioritize opponents we've played less against
            weights = []
            for opp in self.opponents:
                weight = 1.0 / (1.0 + opp.games_played)
                weights.append(weight)

            total = sum(weights)
            weights = [w / total for w in weights]
            return random.choices(self.opponents, weights=weights, k=1)[0]

        else:
            return random.choice(self.opponents)

    def create_player(
        self,
        opponent_info: OpponentInfo,
        battle_format: str,
    ) -> HistoricalPlayer:
        """Create a player from an opponent info.

        Args:
            opponent_info: The opponent to create a player for
            battle_format: Pokemon Showdown battle format

        Returns:
            A HistoricalPlayer instance
        """
        return HistoricalPlayer(
            checkpoint_path=opponent_info.checkpoint_path,
            model_class=PolicyValueNetwork,
            state_encoder=self.state_encoder,
            device=self.device,
            battle_format=battle_format,
            max_concurrent_battles=1,
        )

    def update_stats(
        self,
        opponent_info: OpponentInfo,
        won: bool,
        agent_skill: float,
    ) -> tuple[float, float]:
        """Update opponent statistics after a game.

        Args:
            opponent_info: The opponent that was played against
            won: Whether the agent won
            agent_skill: Current agent skill rating

        Returns:
            Tuple of (new_agent_skill, new_opponent_skill)
        """
        opponent_info.games_played += 1
        if won:
            opponent_info.losses += 1
        else:
            opponent_info.wins += 1

        # Update skill ratings (using Elo algorithm internally)
        new_agent_skill, new_opponent_skill = calculate_elo_update(
            agent_skill,
            opponent_info.skill_rating,
            won,
        )
        opponent_info.skill_rating = new_opponent_skill

        return new_agent_skill, new_opponent_skill

    @property
    def size(self) -> int:
        """Number of opponents in the pool."""
        return len(self.opponents)

    @property
    def average_skill(self) -> float:
        """Average skill rating in the pool."""
        if not self.opponents:
            return 1000.0
        return sum(o.skill_rating for o in self.opponents) / len(self.opponents)


def calculate_elo_update(
    player_elo: float,
    opponent_elo: float,
    player_won: bool,
    k_factor: float = 32.0,
) -> tuple[float, float]:
    """Calculate new Elo ratings after a game.

    Args:
        player_elo: Player's current Elo
        opponent_elo: Opponent's current Elo
        player_won: Whether the player won
        k_factor: Elo K-factor (higher = more volatile)

    Returns:
        Tuple of (new_player_elo, new_opponent_elo)
    """
    # Expected scores
    expected_player = 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400.0))
    expected_opponent = 1.0 - expected_player

    # Actual scores
    actual_player = 1.0 if player_won else 0.0
    actual_opponent = 1.0 - actual_player

    # New ratings
    new_player_elo = player_elo + k_factor * (actual_player - expected_player)
    new_opponent_elo = opponent_elo + k_factor * (actual_opponent - expected_opponent)

    return new_player_elo, new_opponent_elo


class SelfPlayManager:
    """Manages self-play training with opponent pool."""

    def __init__(
        self,
        pool_dir: Path | str,
        max_pool_size: int = 10,
        self_play_ratio: float = 0.8,
        checkpoint_interval: int = 50000,
        sampling_strategy: str = "skill_matched",
        device: torch.device | None = None,
    ):
        """Initialize self-play manager.

        Args:
            pool_dir: Directory for opponent checkpoints
            max_pool_size: Maximum opponents in pool
            self_play_ratio: Fraction of games against self-play opponents
            checkpoint_interval: Timesteps between adding to pool
            sampling_strategy: How to sample opponents
            device: Device for models
        """
        self.pool_dir = Path(pool_dir)
        self.self_play_ratio = self_play_ratio
        self.checkpoint_interval = checkpoint_interval
        self.sampling_strategy = sampling_strategy
        self.device = device or torch.device("cpu")

        self.opponent_pool = OpponentPool(
            pool_dir=self.pool_dir,
            max_size=max_pool_size,
            device=self.device,
        )

        self.agent_skill = 1000.0
        # Initialize to negative interval so first checkpoint triggers at timestep 0
        self.last_checkpoint_timestep = -checkpoint_interval
        self.games_vs_self_play = 0
        self.games_vs_random = 0

    def should_use_self_play(self) -> bool:
        """Decide whether to use self-play or random opponent."""
        if self.opponent_pool.size == 0:
            return False
        return random.random() < self.self_play_ratio

    def should_add_checkpoint(self, current_timestep: int) -> bool:
        """Check if it's time to add a new checkpoint to the pool."""
        return (
            current_timestep - self.last_checkpoint_timestep >= self.checkpoint_interval
        )

    def add_checkpoint(
        self,
        model: PolicyValueNetwork,
        timestep: int,
    ) -> OpponentInfo:
        """Add current model to opponent pool."""
        opponent_info = self.opponent_pool.add_opponent(
            model=model,
            timestep=timestep,
            skill_rating=self.agent_skill,
        )
        self.last_checkpoint_timestep = timestep
        return opponent_info

    def get_opponent(
        self,
        battle_format: str,
    ) -> tuple[Player, OpponentInfo | None]:
        """Get an opponent to play against.

        Args:
            battle_format: Pokemon Showdown battle format

        Returns:
            Tuple of (opponent_player, opponent_info or None for random)
        """
        from poke_env.player import RandomPlayer

        if self.should_use_self_play():
            opponent_info = self.opponent_pool.sample_opponent(
                strategy=self.sampling_strategy,
                current_skill=self.agent_skill,
            )
            if opponent_info:
                player = self.opponent_pool.create_player(opponent_info, battle_format)
                self.games_vs_self_play += 1
                return player, opponent_info

        # Fallback to random
        self.games_vs_random += 1
        return RandomPlayer(
            battle_format=battle_format,
            max_concurrent_battles=1,
        ), None

    def update_after_game(
        self,
        opponent_info: OpponentInfo | None,
        won: bool,
    ) -> None:
        """Update statistics after a game.

        Args:
            opponent_info: The opponent played (None for random)
            won: Whether the agent won
        """
        if opponent_info is not None:
            self.agent_skill, _ = self.opponent_pool.update_stats(
                opponent_info, won, self.agent_skill
            )

    def get_stats(self) -> dict[str, Any]:
        """Get self-play statistics."""
        total_games = self.games_vs_self_play + self.games_vs_random
        return {
            "agent_skill": self.agent_skill,
            "pool_size": self.opponent_pool.size,
            "pool_avg_skill": self.opponent_pool.average_skill,
            "games_vs_self_play": self.games_vs_self_play,
            "games_vs_random": self.games_vs_random,
            "self_play_ratio_actual": (
                self.games_vs_self_play / total_games if total_games > 0 else 0.0
            ),
        }

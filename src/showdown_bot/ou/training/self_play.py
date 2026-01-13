"""Self-play system for OU training against historical checkpoints.

This adapts the random battles self-play system for OU format:
- Uses OUPlayerNetwork instead of PolicyValueNetwork
- Handles team assignment for OU battles
- Supports skill-matched opponent sampling
"""

import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder
from poke_env.battle import AbstractBattle

from showdown_bot.ou.player.network import OUPlayerNetwork
from showdown_bot.ou.player.state_encoder import OUStateEncoder
from showdown_bot.ou.player.ou_player import OUTeambuilder, OUMaxDamagePlayer, _get_team_preview_order
from showdown_bot.ou.shared.data_loader import Team

logger = logging.getLogger(__name__)


@dataclass
class OUOpponentInfo:
    """Information about an OU opponent in the pool."""

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


class OUHistoricalPlayer(Player):
    """An OU player that uses a historical checkpoint."""

    def __init__(
        self,
        checkpoint_path: Path,
        state_encoder: OUStateEncoder,
        device: torch.device,
        teams: list[Team] | None = None,
        **kwargs: Any,
    ):
        """Initialize historical player.

        Args:
            checkpoint_path: Path to the checkpoint file
            state_encoder: State encoder for battles
            device: Device for inference
            teams: Teams to use (picks randomly)
            **kwargs: Arguments passed to poke-env Player
        """
        if teams:
            kwargs.setdefault("team", OUTeambuilder(teams))

        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.state_encoder = state_encoder
        self.device = device

        # Load model from checkpoint
        self.network = self._create_network()
        self._load_checkpoint(checkpoint_path)
        self.network.eval()

    def _create_network(self) -> OUPlayerNetwork:
        """Create network with same architecture as training."""
        return OUPlayerNetwork(
            pokemon_dim=self.state_encoder.POKEMON_FEATURES,
            field_dim=self.state_encoder.FIELD_FEATURES,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            num_actions=self.state_encoder.NUM_ACTIONS,
        ).to(self.device)

    def _load_checkpoint(self, path: Path) -> None:
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Assume it's just the state dict
            self.network.load_state_dict(checkpoint)

    def teampreview(self, battle: AbstractBattle) -> str:
        """Select lead Pokemon during team preview phase."""
        return _get_team_preview_order(battle)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose a move using the loaded model."""
        # Handle cleanup race condition
        if not hasattr(self, 'network') or self.network is None:
            logger.warning(
                f"OUHistoricalPlayer network is None during choose_move "
                f"(battle turn {battle.turn}) - using random move."
            )
            return self.choose_random_move(battle)

        # Encode state
        state = self.state_encoder.encode_battle(battle)
        state = state.to_device(self.device)

        # Get action from network (greedy)
        with torch.no_grad():
            self.network.eval()
            action, _ = self.network.get_action(state, deterministic=True)

        # Convert to battle order
        order = self.state_encoder.action_to_battle_order(action, battle)
        if order:
            return order

        return self.choose_random_move(battle)


class OUOpponentPool:
    """Pool of historical OU opponents for self-play training."""

    def __init__(
        self,
        pool_dir: Path | str,
        max_size: int = 10,
        device: torch.device | None = None,
        teams: list[Team] | None = None,
    ):
        """Initialize opponent pool.

        Args:
            pool_dir: Directory to store opponent checkpoints
            max_size: Maximum number of opponents to keep
            device: Device for loading models
            teams: Teams for opponents to use
        """
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.device = device or torch.device("cpu")
        self.teams = teams or []

        self.opponents: list[OUOpponentInfo] = []
        self.state_encoder = OUStateEncoder(device=self.device)

        # Load existing checkpoints
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        """Load existing checkpoints from the pool directory."""
        for checkpoint_path in sorted(self.pool_dir.glob("ou_opponent_*.pt")):
            # Extract timestep from filename
            try:
                timestep = int(checkpoint_path.stem.split("_")[2])
            except (IndexError, ValueError):
                timestep = 0

            self.opponents.append(
                OUOpponentInfo(
                    checkpoint_path=checkpoint_path,
                    timestep_added=timestep,
                )
            )
        if self.opponents:
            logger.info(f"Loaded {len(self.opponents)} existing OU opponents from pool")

    def add_opponent(
        self,
        network: OUPlayerNetwork,
        timestep: int,
        skill_rating: float = 1000.0,
    ) -> OUOpponentInfo:
        """Add a new opponent to the pool.

        Args:
            network: The network to save as an opponent
            timestep: Current training timestep
            skill_rating: Initial skill rating

        Returns:
            The created OUOpponentInfo
        """
        # Save checkpoint
        checkpoint_path = self.pool_dir / f"ou_opponent_{timestep}.pt"
        checkpoint = {
            "model_state_dict": network.state_dict(),
            "timestep": timestep,
            "skill_rating": skill_rating,
        }
        torch.save(checkpoint, checkpoint_path)

        opponent_info = OUOpponentInfo(
            checkpoint_path=checkpoint_path,
            skill_rating=skill_rating,
            timestep_added=timestep,
        )
        self.opponents.append(opponent_info)

        logger.info(f"Added OU opponent at timestep {timestep} (skill: {skill_rating:.0f})")

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

        # Keep evenly distributed by skill
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
                logger.debug(f"Pruned opponent: {opponent.checkpoint_path}")

        self.opponents = new_opponents

    def sample_opponent(
        self,
        strategy: str = "uniform",
        current_skill: float = 1000.0,
    ) -> OUOpponentInfo | None:
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
        opponent_info: OUOpponentInfo,
        battle_format: str = "gen9ou",
        server_configuration=None,
    ) -> OUHistoricalPlayer:
        """Create a player from an opponent info.

        Args:
            opponent_info: The opponent to create a player for
            battle_format: Pokemon Showdown battle format
            server_configuration: Optional server configuration

        Returns:
            An OUHistoricalPlayer instance
        """
        return OUHistoricalPlayer(
            checkpoint_path=opponent_info.checkpoint_path,
            state_encoder=self.state_encoder,
            device=self.device,
            teams=self.teams,
            battle_format=battle_format,
            max_concurrent_battles=1,
            server_configuration=server_configuration,
        )

    def update_stats(
        self,
        opponent_info: OUOpponentInfo,
        won: bool,
        agent_skill: float,
        k_factor: float = 32.0,
    ) -> tuple[float, float]:
        """Update opponent statistics after a game.

        Args:
            opponent_info: The opponent that was played against
            won: Whether the agent won
            agent_skill: Current agent skill rating
            k_factor: Elo K-factor

        Returns:
            Tuple of (new_agent_skill, new_opponent_skill)
        """
        opponent_info.games_played += 1
        if won:
            opponent_info.losses += 1
        else:
            opponent_info.wins += 1

        # Update skill ratings using Elo
        new_agent_skill, new_opponent_skill = calculate_elo_update(
            agent_skill,
            opponent_info.skill_rating,
            won,
            k_factor,
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


class OUSelfPlayManager:
    """Manages self-play training for OU with opponent pool."""

    def __init__(
        self,
        pool_dir: Path | str,
        teams: list[Team],
        max_pool_size: int = 10,
        self_play_ratio: float = 0.7,
        checkpoint_interval: int = 50000,
        sampling_strategy: str = "skill_matched",
        device: torch.device | None = None,
    ):
        """Initialize self-play manager.

        Args:
            pool_dir: Directory for opponent checkpoints
            teams: Teams for players to use
            max_pool_size: Maximum opponents in pool
            self_play_ratio: Fraction of games against self-play opponents
            checkpoint_interval: Timesteps between adding to pool
            sampling_strategy: How to sample opponents ("uniform", "skill_matched", "prioritized")
            device: Device for models
        """
        self.pool_dir = Path(pool_dir)
        self.teams = teams
        self.self_play_ratio = self_play_ratio
        self.checkpoint_interval = checkpoint_interval
        self.sampling_strategy = sampling_strategy
        self.device = device or torch.device("cpu")

        self.opponent_pool = OUOpponentPool(
            pool_dir=self.pool_dir,
            max_size=max_pool_size,
            device=self.device,
            teams=teams,
        )

        self.agent_skill = 1000.0
        # Initialize to negative interval so first checkpoint triggers
        self.last_checkpoint_timestep = -checkpoint_interval
        self.games_vs_self_play = 0
        self.games_vs_random = 0
        self.games_vs_maxdamage = 0

    def should_use_self_play(self) -> bool:
        """Decide whether to use self-play or baseline opponent."""
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
        network: OUPlayerNetwork,
        timestep: int,
    ) -> OUOpponentInfo:
        """Add current model to opponent pool."""
        opponent_info = self.opponent_pool.add_opponent(
            network=network,
            timestep=timestep,
            skill_rating=self.agent_skill,
        )
        self.last_checkpoint_timestep = timestep
        return opponent_info

    def get_opponent(
        self,
        battle_format: str = "gen9ou",
        server_configuration=None,
    ) -> tuple[Player, OUOpponentInfo | None, str]:
        """Get an opponent to play against.

        Args:
            battle_format: Pokemon Showdown battle format
            server_configuration: Optional server configuration

        Returns:
            Tuple of (opponent_player, opponent_info or None, opponent_type)
        """
        if self.should_use_self_play():
            opponent_info = self.opponent_pool.sample_opponent(
                strategy=self.sampling_strategy,
                current_skill=self.agent_skill,
            )
            if opponent_info:
                player = self.opponent_pool.create_player(
                    opponent_info, battle_format, server_configuration
                )
                self.games_vs_self_play += 1
                return player, opponent_info, "self_play"

        # Use baseline opponent (gen9ou requires teams, so we use OUMaxDamagePlayer)
        self.games_vs_maxdamage += 1
        return OUMaxDamagePlayer(
            teams=self.teams,
            battle_format=battle_format,
            max_concurrent_battles=1,
            server_configuration=server_configuration,
        ), None, "maxdamage"

    def update_after_game(
        self,
        opponent_info: OUOpponentInfo | None,
        won: bool,
        opponent_type: str = "unknown",
    ) -> None:
        """Update statistics after a game.

        Args:
            opponent_info: The opponent played (None for baseline)
            won: Whether the agent won
            opponent_type: Type of opponent for logging
        """
        if opponent_info is not None:
            self.agent_skill, _ = self.opponent_pool.update_stats(
                opponent_info, won, self.agent_skill
            )
        else:
            # Update skill rating against assumed baseline rating (maxdamage)
            baseline_skill = 900.0
            expected = 1.0 / (1.0 + 10 ** ((baseline_skill - self.agent_skill) / 400.0))
            actual = 1.0 if won else 0.0
            self.agent_skill += 16.0 * (actual - expected)  # Lower K-factor for baselines

    def get_stats(self) -> dict[str, Any]:
        """Get self-play statistics."""
        total_games = self.games_vs_self_play + self.games_vs_random + self.games_vs_maxdamage
        return {
            "agent_skill": self.agent_skill,
            "pool_size": self.opponent_pool.size,
            "pool_avg_skill": self.opponent_pool.average_skill,
            "games_vs_self_play": self.games_vs_self_play,
            "games_vs_random": self.games_vs_random,
            "games_vs_maxdamage": self.games_vs_maxdamage,
            "self_play_ratio_actual": (
                self.games_vs_self_play / total_games if total_games > 0 else 0.0
            ),
        }

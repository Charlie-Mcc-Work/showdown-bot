"""Self-play system for training against historical checkpoints."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder
from poke_env.battle import AbstractBattle
from poke_env.ps_client import ServerConfiguration

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

        # Mixed precision inference - disabled for now, overhead exceeds benefit for small models
        self.use_amp = False  # device.type == "cuda"

        # Load model from checkpoint
        self.model = model_class.from_config().to(device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

    def _load_checkpoint(self, path: Path) -> None:
        """Load model weights from checkpoint, migrating if needed."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Try direct load first, migrate if dimensions don't match
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                migrated = self._migrate_state_dict(state_dict)
                self.model.load_state_dict(migrated)
            else:
                raise

    def _migrate_state_dict(
        self, old_state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Migrate old checkpoint state dict to match current model dimensions."""
        new_state_dict = self.model.state_dict()
        migrated = {}

        for key, new_tensor in new_state_dict.items():
            if key not in old_state_dict:
                migrated[key] = new_tensor
                continue

            old_tensor = old_state_dict[key]

            if old_tensor.shape == new_tensor.shape:
                migrated[key] = old_tensor
            elif len(old_tensor.shape) == 2 and len(new_tensor.shape) == 2:
                old_out, old_in = old_tensor.shape
                new_out, new_in = new_tensor.shape

                if old_out == new_out and old_in < new_in:
                    # Zero-pad new input features
                    padded = torch.zeros_like(new_tensor)
                    padded[:, :old_in] = old_tensor
                    migrated[key] = padded
                else:
                    migrated[key] = new_tensor
            else:
                migrated[key] = new_tensor

        return migrated

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose a move using the loaded model."""
        # Handle cleanup race condition - model may be deleted while websocket
        # still has pending messages. This should be rare - log if it happens.
        if not hasattr(self, 'model') or self.model is None:
            import logging
            logging.getLogger(__name__).warning(
                f"HistoricalPlayer model is None during choose_move (battle turn {battle.turn}) "
                "- using random move. If this happens often, increase cleanup delay."
            )
            return self.choose_random_move(battle)

        state = self.state_encoder.encode_battle(battle)

        player_pokemon = state.player_pokemon.unsqueeze(0).to(self.device)
        opponent_pokemon = state.opponent_pokemon.unsqueeze(0).to(self.device)
        player_active_idx = torch.tensor([state.player_active_idx], device=self.device)
        opponent_active_idx = torch.tensor([state.opponent_active_idx], device=self.device)
        field_state = state.field_state.unsqueeze(0).to(self.device)
        action_mask = state.action_mask.unsqueeze(0).to(self.device)

        with torch.no_grad(), autocast("cuda", enabled=self.use_amp):
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
        self.metadata_path = self.pool_dir / "pool_metadata.json"

        self.opponents: list[OpponentInfo] = []
        self.state_encoder = StateEncoder(device=self.device)

        # Load existing checkpoints and their metadata
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        """Load existing checkpoints from the pool directory."""
        # Load metadata if it exists
        metadata = self._load_metadata()

        for checkpoint_path in sorted(self.pool_dir.glob("opponent_*.pt")):
            # Extract timestep from filename
            try:
                timestep = int(checkpoint_path.stem.split("_")[1])
            except (IndexError, ValueError):
                timestep = 0

            # Get saved skill rating from metadata, default to 1000
            filename = checkpoint_path.name
            saved_info = metadata.get(filename, {})
            skill_rating = saved_info.get("skill_rating", 1000.0)
            games_played = saved_info.get("games_played", 0)
            wins = saved_info.get("wins", 0)
            losses = saved_info.get("losses", 0)

            self.opponents.append(
                OpponentInfo(
                    checkpoint_path=checkpoint_path,
                    timestep_added=timestep,
                    skill_rating=skill_rating,
                    games_played=games_played,
                    wins=wins,
                    losses=losses,
                )
            )

        if metadata and self.opponents:
            avg_skill = sum(o.skill_rating for o in self.opponents) / len(self.opponents)
            print(f"Loaded {len(self.opponents)} opponents (avg skill: {avg_skill:.0f})")

    def _load_metadata(self) -> dict[str, Any]:
        """Load opponent metadata from JSON file."""
        if not self.metadata_path.exists():
            return {}
        try:
            with open(self.metadata_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def save_metadata(self) -> None:
        """Save opponent metadata to JSON file."""
        metadata = {}
        for opp in self.opponents:
            metadata[opp.checkpoint_path.name] = {
                "skill_rating": opp.skill_rating,
                "games_played": opp.games_played,
                "wins": opp.wins,
                "losses": opp.losses,
                "timestep_added": opp.timestep_added,
            }
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except OSError:
            pass  # Non-critical, will rebuild from defaults

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

        # Save metadata after adding
        self.save_metadata()

        return opponent_info

    def _prune_pool(self) -> None:
        """Remove excess opponents, keeping diverse skill levels and training history.

        Strategy for diversity:
        1. Keep oldest opponent (weak baseline for confidence)
        2. Keep newest opponent (most recent checkpoint)
        3. Keep best opponent (strongest for challenge)
        4. Keep worst opponent (easy wins for exploration)
        5. Fill remaining slots with skill-diverse selection
        """
        if len(self.opponents) <= self.max_size:
            return

        to_keep: set[Path] = set()

        # Sort by different criteria
        by_skill = sorted(self.opponents, key=lambda x: x.skill_rating)
        by_time = sorted(self.opponents, key=lambda x: x.timestep_added)

        # Priority keeps (guaranteed slots)
        priority_count = min(4, self.max_size)

        # 1. Keep oldest (weak baseline)
        if by_time:
            to_keep.add(by_time[0].checkpoint_path)

        # 2. Keep newest (most recent)
        if by_time:
            to_keep.add(by_time[-1].checkpoint_path)

        # 3. Keep best (strongest)
        if by_skill:
            to_keep.add(by_skill[-1].checkpoint_path)

        # 4. Keep worst (easy wins)
        if by_skill:
            to_keep.add(by_skill[0].checkpoint_path)

        # 5. Fill remaining slots with skill-diverse selection
        remaining_slots = self.max_size - len(to_keep)
        if remaining_slots > 0:
            # Get opponents not yet kept, spread evenly by skill
            available = [o for o in by_skill if o.checkpoint_path not in to_keep]
            if available:
                step = max(1, len(available) / remaining_slots)
                for i in range(remaining_slots):
                    idx = min(int(i * step), len(available) - 1)
                    to_keep.add(available[idx].checkpoint_path)

        # Remove opponents not in to_keep
        new_opponents = []
        for opponent in self.opponents:
            if opponent.checkpoint_path in to_keep:
                new_opponents.append(opponent)
            else:
                # Delete the checkpoint file
                opponent.checkpoint_path.unlink(missing_ok=True)

        self.opponents = new_opponents

        # Save metadata after pruning
        self.save_metadata()

    def sample_opponent(
        self,
        strategy: str = "uniform",
        current_skill: float = 1000.0,
    ) -> OpponentInfo | None:
        """Sample an opponent from the pool.

        Args:
            strategy: Sampling strategy:
                - "uniform": Random selection
                - "skill_matched": Prefer similar skill (can cause echo chamber)
                - "prioritized": Prefer less-played opponents
                - "challenge": Prefer stronger opponents (helps learning)
                - "diverse": Mix of strategies (recommended for training)
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

        elif strategy == "challenge":
            # Prefer stronger opponents - helps agent learn from better play
            weights = []
            for opp in self.opponents:
                # Higher weight for opponents stronger than us
                skill_diff = opp.skill_rating - current_skill
                # Sigmoid-like weighting: strong preference for stronger opponents
                weight = 1.0 + max(0, skill_diff / 200.0)
                weights.append(weight)

            total = sum(weights)
            weights = [w / total for w in weights]
            return random.choices(self.opponents, weights=weights, k=1)[0]

        elif strategy == "diverse":
            # Mix of strategies to avoid echo chamber:
            # - 40% uniform (ensures all opponents get played)
            # - 30% challenge (learn from stronger opponents)
            # - 30% skill_matched (practice against similar skill)
            roll = random.random()
            if roll < 0.4:
                return random.choice(self.opponents)
            elif roll < 0.7:
                return self.sample_opponent("challenge", current_skill)
            else:
                return self.sample_opponent("skill_matched", current_skill)

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
        server_configuration: ServerConfiguration | None = None,
    ) -> HistoricalPlayer:
        """Create a player from an opponent info.

        Args:
            opponent_info: The opponent to create a player for
            battle_format: Pokemon Showdown battle format
            server_configuration: Server to connect to (None for default)

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
            server_configuration=server_configuration,
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

        # Save metadata periodically (every 10 games)
        total_games = sum(o.games_played for o in self.opponents)
        if total_games % 10 == 0:
            self.save_metadata()

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
    """Manages self-play training with opponent pool and curriculum learning."""

    def __init__(
        self,
        pool_dir: Path | str,
        max_pool_size: int = 10,
        self_play_ratio: float = 0.8,
        checkpoint_interval: int = 50000,
        sampling_strategy: str = "diverse",
        device: torch.device | None = None,
        # Curriculum parameters
        curriculum_enabled: bool = True,
        curriculum_skill_min: float = 1000.0,
        curriculum_skill_max: float = 5000.0,
        curriculum_early_self_play: float = 0.3,
        curriculum_early_max_damage: float = 0.4,
        curriculum_late_self_play: float = 0.8,
        curriculum_late_max_damage: float = 0.15,
    ):
        """Initialize self-play manager.

        Args:
            pool_dir: Directory for opponent checkpoints
            max_pool_size: Maximum opponents in pool
            self_play_ratio: Fraction of games against self-play opponents (if curriculum disabled)
            checkpoint_interval: Timesteps between adding to pool
            sampling_strategy: How to sample opponents
            device: Device for models
            curriculum_enabled: Whether to use curriculum-based opponent selection
            curriculum_skill_min: Skill level at start of curriculum
            curriculum_skill_max: Skill level at end of curriculum
            curriculum_early_self_play: Self-play ratio at early stage
            curriculum_early_max_damage: MaxDamage ratio at early stage
            curriculum_late_self_play: Self-play ratio at late stage
            curriculum_late_max_damage: MaxDamage ratio at late stage
        """
        self.pool_dir = Path(pool_dir)
        self.self_play_ratio = self_play_ratio
        self.checkpoint_interval = checkpoint_interval
        self.sampling_strategy = sampling_strategy
        self.device = device or torch.device("cpu")

        # Curriculum settings
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_skill_min = curriculum_skill_min
        self.curriculum_skill_max = curriculum_skill_max
        self.curriculum_early_self_play = curriculum_early_self_play
        self.curriculum_early_max_damage = curriculum_early_max_damage
        self.curriculum_late_self_play = curriculum_late_self_play
        self.curriculum_late_max_damage = curriculum_late_max_damage

        self.opponent_pool = OpponentPool(
            pool_dir=self.pool_dir,
            max_size=max_pool_size,
            device=self.device,
        )

        self.agent_skill = 1000.0
        # Initialize to negative interval so first checkpoint triggers at timestep 0
        self.last_checkpoint_timestep = -checkpoint_interval
        self.games_vs_self_play = 0
        self.games_vs_max_damage = 0
        self.games_vs_random = 0

    def get_curriculum_ratios(self) -> tuple[float, float, float]:
        """Get current opponent ratios based on skill level.

        Returns:
            Tuple of (self_play_ratio, max_damage_ratio, random_ratio)
        """
        if not self.curriculum_enabled:
            # Use fixed ratio when curriculum is disabled
            return self.self_play_ratio, 0.0, 1.0 - self.self_play_ratio

        # Calculate progress through curriculum (0 to 1)
        skill_range = self.curriculum_skill_max - self.curriculum_skill_min
        if skill_range <= 0:
            progress = 1.0
        else:
            progress = (self.agent_skill - self.curriculum_skill_min) / skill_range
            progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]

        # Linear interpolation between early and late ratios
        self_play = (
            self.curriculum_early_self_play
            + progress * (self.curriculum_late_self_play - self.curriculum_early_self_play)
        )
        max_damage = (
            self.curriculum_early_max_damage
            + progress * (self.curriculum_late_max_damage - self.curriculum_early_max_damage)
        )
        random_ratio = 1.0 - self_play - max_damage

        return self_play, max_damage, random_ratio

    def select_opponent_type(self) -> str:
        """Select which type of opponent to use based on curriculum.

        Returns:
            One of: "self_play", "max_damage", "random"
        """
        self_play_ratio, max_damage_ratio, _ = self.get_curriculum_ratios()

        # If no self-play opponents available, redistribute to others
        if self.opponent_pool.size == 0:
            # Can't do self-play, split between max_damage and random
            total = max_damage_ratio + (1.0 - self_play_ratio - max_damage_ratio)
            if total > 0:
                adjusted_max_damage = max_damage_ratio / total
            else:
                adjusted_max_damage = 0.5
            if random.random() < adjusted_max_damage:
                return "max_damage"
            return "random"

        roll = random.random()
        if roll < self_play_ratio:
            return "self_play"
        elif roll < self_play_ratio + max_damage_ratio:
            return "max_damage"
        else:
            return "random"

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
        server_configuration: ServerConfiguration | None = None,
    ) -> tuple[Player, OpponentInfo | None]:
        """Get an opponent to play against using curriculum-based selection.

        Args:
            battle_format: Pokemon Showdown battle format
            server_configuration: Server to connect to (None for default)

        Returns:
            Tuple of (opponent_player, opponent_info or None for non-self-play)
        """
        from poke_env.player import RandomPlayer
        from showdown_bot.environment.battle_env import MaxDamagePlayer

        opponent_type = self.select_opponent_type()

        if opponent_type == "self_play":
            opponent_info = self.opponent_pool.sample_opponent(
                strategy=self.sampling_strategy,
                current_skill=self.agent_skill,
            )
            if opponent_info:
                player = self.opponent_pool.create_player(
                    opponent_info, battle_format, server_configuration
                )
                self.games_vs_self_play += 1
                return player, opponent_info
            # Fallback if sampling failed
            opponent_type = "max_damage"

        if opponent_type == "max_damage":
            self.games_vs_max_damage += 1
            return MaxDamagePlayer(
                battle_format=battle_format,
                max_concurrent_battles=1,
                server_configuration=server_configuration,
            ), None

        # Random
        self.games_vs_random += 1
        return RandomPlayer(
            battle_format=battle_format,
            max_concurrent_battles=1,
            server_configuration=server_configuration,
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
        total_games = self.games_vs_self_play + self.games_vs_max_damage + self.games_vs_random
        self_play_ratio, max_damage_ratio, random_ratio = self.get_curriculum_ratios()
        return {
            "agent_skill": self.agent_skill,
            "pool_size": self.opponent_pool.size,
            "pool_avg_skill": self.opponent_pool.average_skill,
            "games_vs_self_play": self.games_vs_self_play,
            "games_vs_max_damage": self.games_vs_max_damage,
            "games_vs_random": self.games_vs_random,
            "self_play_ratio_actual": (
                self.games_vs_self_play / total_games if total_games > 0 else 0.0
            ),
            "max_damage_ratio_actual": (
                self.games_vs_max_damage / total_games if total_games > 0 else 0.0
            ),
            # Current curriculum ratios (what the scheduler is targeting)
            "curriculum_self_play_target": self_play_ratio,
            "curriculum_max_damage_target": max_damage_ratio,
            "curriculum_random_target": random_ratio,
        }

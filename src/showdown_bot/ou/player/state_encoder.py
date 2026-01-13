"""State encoder for OU battles.

This encoder extends the random battles encoder with:
1. Full knowledge of our team (moves, items, EVs we chose)
2. Team preview information
3. Opponent prediction based on revealed info + usage stats
4. Tera type tracking and decision making
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from poke_env.battle import AbstractBattle, Field, SideCondition, Weather

from showdown_bot.ou.shared.embeddings import SharedEmbeddings
from showdown_bot.ou.shared.data_loader import PokemonDataLoader, UsageStatsLoader


# Pre-defined mappings for efficient lookup (avoid string operations)
WEATHER_TO_IDX = {
    Weather.SUNNYDAY: 0, Weather.RAINDANCE: 1, Weather.SANDSTORM: 2,
    Weather.HAIL: 3, Weather.SNOW: 3, Weather.DESOLATELAND: 0,
    Weather.PRIMORDIALSEA: 1, Weather.DELTASTREAM: 4,
}
TERRAIN_TO_IDX = {
    Field.ELECTRIC_TERRAIN: 0, Field.GRASSY_TERRAIN: 1,
    Field.MISTY_TERRAIN: 2, Field.PSYCHIC_TERRAIN: 3,
}
# Status mapping
STATUS_TO_IDX = {"brn": 0, "frz": 1, "par": 2, "psn": 3, "slp": 4, "tox": 5}

# Type mapping (cached for fast lookup)
TYPE_TO_IDX = {
    "normal": 1, "fire": 2, "water": 3, "electric": 4, "grass": 5, "ice": 6,
    "fighting": 7, "poison": 8, "ground": 9, "flying": 10, "psychic": 11,
    "bug": 12, "rock": 13, "ghost": 14, "dragon": 15, "dark": 16, "steel": 17,
    "fairy": 18,
}


@dataclass
class OUEncodedState:
    """Encoded state for an OU battle.

    Extends the random battles encoded state with OU-specific features.
    """

    # Our team (full knowledge)
    our_team: torch.Tensor  # (6, pokemon_features)
    our_active_idx: int

    # Opponent team (partial knowledge)
    opp_revealed: torch.Tensor  # (6, pokemon_features) - revealed Pokemon
    opp_revealed_mask: torch.Tensor  # (6,) - which slots are revealed
    opp_active_idx: int
    opp_predicted: torch.Tensor  # (6, pokemon_features) - predicted unrevealed

    # Field state
    field_state: torch.Tensor  # Weather, terrain, hazards, etc.

    # Action mask
    action_mask: torch.Tensor  # (num_actions,) - legal actions

    # Tera state
    our_tera_used: bool
    our_tera_type: int  # 0 if not terrastalized
    opp_tera_used: bool
    opp_tera_type: int

    # Turn information
    turn_number: int

    def to_device(self, device: torch.device) -> "OUEncodedState":
        """Move all tensors to device."""
        return OUEncodedState(
            our_team=self.our_team.to(device),
            our_active_idx=self.our_active_idx,
            opp_revealed=self.opp_revealed.to(device),
            opp_revealed_mask=self.opp_revealed_mask.to(device),
            opp_active_idx=self.opp_active_idx,
            opp_predicted=self.opp_predicted.to(device),
            field_state=self.field_state.to(device),
            action_mask=self.action_mask.to(device),
            our_tera_used=self.our_tera_used,
            our_tera_type=self.our_tera_type,
            opp_tera_used=self.opp_tera_used,
            opp_tera_type=self.opp_tera_type,
            turn_number=self.turn_number,
        )


class OUStateEncoder:
    """Encodes OU battle state for the neural network.

    Key differences from RandomBattles encoder:
    1. We know our full team from the start
    2. We track what opponent has revealed
    3. We predict unrevealed opponent Pokemon using usage stats
    4. We track Tera state
    """

    # Feature dimensions
    POKEMON_FEATURES = 128  # Per-Pokemon encoding
    FIELD_FEATURES = 64  # Field state encoding
    NUM_ACTIONS = 13  # 4 moves + 5 switches + 4 tera moves

    def __init__(
        self,
        shared_embeddings: SharedEmbeddings | None = None,
        data_loader: PokemonDataLoader | None = None,
        usage_loader: UsageStatsLoader | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the encoder.

        Args:
            shared_embeddings: Shared embedding layers
            data_loader: Pokemon data loader
            usage_loader: Usage stats for opponent prediction
            device: Device for tensors
        """
        self.embeddings = shared_embeddings
        self.data_loader = data_loader or PokemonDataLoader()
        self.usage_loader = usage_loader or UsageStatsLoader()
        self.device = device or torch.device("cpu")

        # Our team (set before battle)
        self.our_team_data: list[dict[str, Any]] = []

        # Cache species_to_id mapping for fast lookup (load once)
        self.data_loader.load()
        self._species_id_cache = self.data_loader.species_to_id

        # Pre-allocate reusable arrays to reduce allocations
        self._zeros_pokemon = np.zeros((6, self.POKEMON_FEATURES), dtype=np.float32)
        self._zeros_field = np.zeros(self.FIELD_FEATURES, dtype=np.float32)

    def set_our_team(self, team_data: list[dict[str, Any]]) -> None:
        """Set our team data before the battle.

        Args:
            team_data: List of dicts with species, moves, item, etc.
        """
        self.our_team_data = team_data

    def encode_battle(self, battle: AbstractBattle) -> OUEncodedState:
        """Encode the current battle state.

        Args:
            battle: The poke-env battle object

        Returns:
            Encoded state for the neural network
        """
        # Encode our team
        our_team = self._encode_our_team(battle)
        our_active_idx = self._get_active_index(battle, is_opponent=False)

        # Encode opponent's revealed Pokemon
        opp_revealed, opp_mask = self._encode_opponent_revealed(battle)
        opp_active_idx = self._get_active_index(battle, is_opponent=True)

        # Predict unrevealed opponent Pokemon
        opp_predicted = self._predict_opponent_unrevealed(battle, opp_mask)

        # Encode field state
        field_state = self._encode_field(battle)

        # Create action mask
        action_mask = self._create_action_mask(battle)

        # Tera state
        our_tera_used = battle.can_tera is False  # Can't tera means already used
        our_tera_type = 0  # TODO: Get actual tera type
        opp_tera_used = False  # TODO: Track opponent tera
        opp_tera_type = 0

        return OUEncodedState(
            our_team=our_team,
            our_active_idx=our_active_idx,
            opp_revealed=opp_revealed,
            opp_revealed_mask=opp_mask,
            opp_active_idx=opp_active_idx,
            opp_predicted=opp_predicted,
            field_state=field_state,
            action_mask=action_mask,
            our_tera_used=our_tera_used,
            our_tera_type=our_tera_type,
            opp_tera_used=opp_tera_used,
            opp_tera_type=opp_tera_type,
            turn_number=battle.turn,
        )

    def _encode_our_team(self, battle: AbstractBattle) -> torch.Tensor:
        """Encode our full team.

        Unlike random battles, we have full knowledge of our team.
        """
        # Pre-allocate numpy array
        team_encoding = np.zeros((6, self.POKEMON_FEATURES), dtype=np.float32)

        for i, pokemon in enumerate(battle.team.values()):
            if i >= 6:
                break
            team_encoding[i] = self._encode_pokemon(pokemon, full_knowledge=True)

        return torch.from_numpy(team_encoding)

    def _encode_opponent_revealed(
        self, battle: AbstractBattle
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode opponent's revealed Pokemon.

        Returns:
            (encodings, mask) where mask is True for revealed Pokemon
        """
        # Pre-allocate numpy arrays
        revealed_encoding = np.zeros((6, self.POKEMON_FEATURES), dtype=np.float32)
        mask = np.zeros(6, dtype=np.bool_)

        for i, pokemon in enumerate(battle.opponent_team.values()):
            if i >= 6:
                break
            revealed_encoding[i] = self._encode_pokemon(pokemon, full_knowledge=False)
            mask[i] = True

        return torch.from_numpy(revealed_encoding), torch.from_numpy(mask)

    def _predict_opponent_unrevealed(
        self,
        battle: AbstractBattle,
        revealed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict unrevealed opponent Pokemon using usage stats.

        NOTE: Currently returns zeros - prediction not yet implemented.
        For efficiency, this returns a pre-allocated tensor.
        """
        # Return zeros tensor directly - no need to allocate new array
        # TODO: Implement actual prediction when ready
        return torch.zeros(6, self.POKEMON_FEATURES)

    def _encode_pokemon(
        self,
        pokemon: Any,  # poke_env Pokemon object
        full_knowledge: bool = False,
    ) -> np.ndarray:
        """Encode a single Pokemon.

        Args:
            pokemon: The poke-env Pokemon object
            full_knowledge: If True, use full info (our Pokemon). If False,
                          only use revealed info (opponent Pokemon).

        Returns:
            (POKEMON_FEATURES,) numpy array
        """
        # Pre-allocate array for efficiency
        features = np.zeros(self.POKEMON_FEATURES, dtype=np.float32)
        idx = 0

        # Species ID (normalized) - use cached species_to_id dict directly
        species_key = pokemon.species.lower().replace(" ", "")
        species_id = self._species_id_cache.get(species_key, 0)
        features[idx] = species_id / 1000.0
        idx += 1

        # HP
        features[idx] = pokemon.current_hp_fraction
        idx += 1

        # Status (7 slots: BRN, FRZ, PAR, PSN, SLP, TOX, healthy)
        if pokemon.status:
            status_name = pokemon.status.name.lower()
            if status_name in STATUS_TO_IDX:
                features[idx + STATUS_TO_IDX[status_name]] = 1.0
        else:
            features[idx + 6] = 1.0  # Healthy
        idx += 7

        # Boosts (7 stats)
        boosts = pokemon.boosts
        features[idx] = boosts.get("atk", 0) / 6.0
        features[idx + 1] = boosts.get("def", 0) / 6.0
        features[idx + 2] = boosts.get("spa", 0) / 6.0
        features[idx + 3] = boosts.get("spd", 0) / 6.0
        features[idx + 4] = boosts.get("spe", 0) / 6.0
        features[idx + 5] = boosts.get("accuracy", 0) / 6.0
        features[idx + 6] = boosts.get("evasion", 0) / 6.0
        idx += 7

        # Types (use direct lookup from module-level dict)
        if pokemon.type_1:
            type1_name = pokemon.type_1.name.lower()
            features[idx] = TYPE_TO_IDX.get(type1_name, 0) / 18.0
        if pokemon.type_2:
            type2_name = pokemon.type_2.name.lower()
            features[idx + 1] = TYPE_TO_IDX.get(type2_name, 0) / 18.0
        # idx += 2 (remaining features are zeros - padding)

        return features

    def _get_active_index(
        self, battle: AbstractBattle, is_opponent: bool = False
    ) -> int:
        """Get index of active Pokemon in team list."""
        team = battle.opponent_team if is_opponent else battle.team
        active = battle.opponent_active_pokemon if is_opponent else battle.active_pokemon

        for i, pokemon in enumerate(team.values()):
            if pokemon == active:
                return i
        return 0

    def _encode_field(self, battle: AbstractBattle) -> torch.Tensor:
        """Encode field state (weather, terrain, hazards, etc.)."""
        # Pre-allocate numpy array for speed
        features = np.zeros(self.FIELD_FEATURES, dtype=np.float32)
        idx = 0

        # Weather (5 types + none = 6 slots)
        if battle.weather:
            for w in battle.weather:
                if w in WEATHER_TO_IDX:
                    features[idx + WEATHER_TO_IDX[w]] = 1.0
                    break
        else:
            features[idx + 5] = 1.0  # No weather
        idx += 6

        # Terrain (4 types + none = 5 slots)
        terrain_set = False
        if battle.fields:
            for f in battle.fields:
                if f in TERRAIN_TO_IDX:
                    features[idx + TERRAIN_TO_IDX[f]] = 1.0
                    terrain_set = True
                    break
        if not terrain_set:
            features[idx + 4] = 1.0  # No terrain
        idx += 5

        # Our side conditions (using enum lookups)
        side = battle.side_conditions
        features[idx] = 1.0 if SideCondition.SPIKES in side else 0.0
        features[idx + 1] = 1.0 if SideCondition.STEALTH_ROCK in side else 0.0
        features[idx + 2] = 1.0 if SideCondition.TOXIC_SPIKES in side else 0.0
        features[idx + 3] = 1.0 if SideCondition.STICKY_WEB in side else 0.0
        idx += 4

        # Opponent side hazards
        opp_side = battle.opponent_side_conditions
        features[idx] = 1.0 if SideCondition.SPIKES in opp_side else 0.0
        features[idx + 1] = 1.0 if SideCondition.STEALTH_ROCK in opp_side else 0.0
        features[idx + 2] = 1.0 if SideCondition.TOXIC_SPIKES in opp_side else 0.0
        features[idx + 3] = 1.0 if SideCondition.STICKY_WEB in opp_side else 0.0
        idx += 4

        # Screens (our side)
        features[idx] = 1.0 if SideCondition.REFLECT in side else 0.0
        features[idx + 1] = 1.0 if SideCondition.LIGHT_SCREEN in side else 0.0
        features[idx + 2] = 1.0 if SideCondition.AURORA_VEIL in side else 0.0
        idx += 3

        # Opponent screens
        features[idx] = 1.0 if SideCondition.REFLECT in opp_side else 0.0
        features[idx + 1] = 1.0 if SideCondition.LIGHT_SCREEN in opp_side else 0.0
        features[idx + 2] = 1.0 if SideCondition.AURORA_VEIL in opp_side else 0.0
        idx += 3

        # Trick room
        features[idx] = 1.0 if Field.TRICK_ROOM in battle.fields else 0.0
        idx += 1

        # Tailwind
        features[idx] = 1.0 if SideCondition.TAILWIND in side else 0.0
        features[idx + 1] = 1.0 if SideCondition.TAILWIND in opp_side else 0.0

        return torch.from_numpy(features)

    def _create_action_mask(self, battle: AbstractBattle) -> torch.Tensor:
        """Create mask for legal actions.

        Actions:
        - 0-3: Regular moves
        - 4-7: Tera moves (same moves but with Tera)
        - 8-12: Switch to Pokemon 0-4
        """
        # Use numpy for efficiency
        mask = np.zeros(self.NUM_ACTIONS, dtype=np.float32)

        # Handle forced switch
        if battle.force_switch:
            # Only switches are valid
            available_switches = battle.available_switches
            team_list = [p for p in battle.team.values() if p != battle.active_pokemon]
            for pokemon in available_switches:
                if pokemon in team_list:
                    idx = team_list.index(pokemon)
                    if idx < 5:
                        mask[8 + idx] = 1.0
        else:
            # Regular turn - moves are available
            num_moves = min(len(battle.available_moves), 4)
            mask[:num_moves] = 1.0

            # Tera moves available if we can tera
            if battle.can_tera:
                mask[4:4 + num_moves] = 1.0

            # Switches
            available_switches = battle.available_switches
            team_list = [p for p in battle.team.values() if p != battle.active_pokemon]
            for pokemon in available_switches:
                if pokemon in team_list:
                    idx = team_list.index(pokemon)
                    if idx < 5:
                        mask[8 + idx] = 1.0

        # Ensure at least one action is valid
        if mask.sum() == 0:
            mask[0] = 1.0

        return torch.from_numpy(mask)

    def action_to_battle_order(
        self,
        action: int,
        battle: AbstractBattle,
    ) -> "BattleOrder | None":
        """Convert action index to poke-env battle order.

        Action space (13 actions):
        - 0-3: Regular moves
        - 4-7: Tera moves (same move but with terastallization)
        - 8-12: Switch to Pokemon 0-4

        Args:
            action: Action index (0-12)
            battle: Current battle state

        Returns:
            BattleOrder or None if action is invalid
        """
        from poke_env.player.battle_order import SingleBattleOrder

        # During forced switch, only switch actions are valid
        if battle.force_switch:
            if action < 8:
                # Model tried to use a move during forced switch - invalid
                return None
            # Fall through to switch handling below

        if action < 4:
            # Regular move action
            available_moves = battle.available_moves
            if action < len(available_moves):
                move = available_moves[action]
                return SingleBattleOrder(order=move)

        elif action < 8:
            # Tera move action (same moves but with tera)
            move_idx = action - 4
            available_moves = battle.available_moves

            if move_idx < len(available_moves) and battle.can_tera:
                move = available_moves[move_idx]
                return SingleBattleOrder(order=move, terastallize=True)
            elif move_idx < len(available_moves):
                # Can't tera, fall back to regular move
                return SingleBattleOrder(order=available_moves[move_idx])

        else:
            # Switch action (8-12 -> switch to team slot 0-4)
            switch_idx = action - 8
            available_switches = battle.available_switches
            team_list = [p for p in battle.team.values() if p != battle.active_pokemon]

            for pokemon in available_switches:
                if pokemon in team_list:
                    idx = team_list.index(pokemon)
                    if idx == switch_idx:
                        return SingleBattleOrder(order=pokemon)

        return None

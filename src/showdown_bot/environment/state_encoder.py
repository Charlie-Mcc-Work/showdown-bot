"""State encoder for converting Pokemon Showdown battle state to tensors."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from poke_env.battle import (
    AbstractBattle,
    Effect,
    Field,
    Move,
    Pokemon,
    SideCondition,
    Status,
    Weather,
)
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type mappings
TYPES = [
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}
NUM_TYPES = len(TYPES)

# Status mappings
STATUSES = ["brn", "frz", "par", "psn", "slp", "tox"]
STATUS_TO_IDX = {s: i for i, s in enumerate(STATUSES)}
NUM_STATUSES = len(STATUSES)

# Weather mappings
WEATHERS = ["sunnyday", "raindance", "sandstorm", "snow", "hail"]
WEATHER_TO_IDX = {w: i for i, w in enumerate(WEATHERS)}
NUM_WEATHERS = len(WEATHERS)

# Terrain mappings
TERRAINS = ["electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"]
TERRAIN_TO_IDX = {t: i for i, t in enumerate(TERRAINS)}
NUM_TERRAINS = len(TERRAINS)

# Side conditions (hazards, screens, etc.)
SIDE_CONDITIONS = [
    "spikes",
    "stealthrock",
    "stickyweb",
    "toxicspikes",
    "reflect",
    "lightscreen",
    "auroraveil",
    "tailwind",
]
SIDE_CONDITION_TO_IDX = {s: i for i, s in enumerate(SIDE_CONDITIONS)}
NUM_SIDE_CONDITIONS = len(SIDE_CONDITIONS)

# Move categories
MOVE_CATEGORIES = ["physical", "special", "status"]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(MOVE_CATEGORIES)}


@dataclass
class EncodedState:
    """Container for encoded battle state tensors."""

    # Pokemon encodings (batch, num_pokemon, features)
    player_pokemon: torch.Tensor  # (6, pokemon_features)
    opponent_pokemon: torch.Tensor  # (6, pokemon_features)

    # Active pokemon indices
    player_active_idx: int
    opponent_active_idx: int

    # Field state
    field_state: torch.Tensor  # (field_features,)

    # Action mask for legal moves
    action_mask: torch.Tensor  # (num_actions,)

    def to_device(self, device: torch.device) -> "EncodedState":
        """Move all tensors to device."""
        return EncodedState(
            player_pokemon=self.player_pokemon.to(device),
            opponent_pokemon=self.opponent_pokemon.to(device),
            player_active_idx=self.player_active_idx,
            opponent_active_idx=self.opponent_active_idx,
            field_state=self.field_state.to(device),
            action_mask=self.action_mask.to(device),
        )


class StateEncoder:
    """Encodes Pokemon Showdown battle state into tensors for the neural network."""

    # Feature dimensions
    POKEMON_FEATURES = (
        1  # HP fraction
        + NUM_TYPES  # Type 1 one-hot
        + NUM_TYPES  # Type 2 one-hot
        + NUM_STATUSES + 1  # Status one-hot (+ none)
        + 7  # Stat boosts (atk, def, spa, spd, spe, acc, eva)
        + 1  # Is active
        + 1  # Is fainted
        + 1  # Is revealed (for opponent)
        + 4 * (NUM_TYPES + 3 + 1 + 1)  # 4 moves: type, category, power norm, pp fraction
    )

    FIELD_FEATURES = (
        NUM_WEATHERS + 1  # Weather one-hot (+ none)
        + NUM_TERRAINS + 1  # Terrain one-hot (+ none)
        + NUM_SIDE_CONDITIONS  # Player side conditions
        + NUM_SIDE_CONDITIONS  # Opponent side conditions
        + 1  # Trick room active
        + 1  # Turn count (normalized)
    )

    NUM_ACTIONS = 9  # 4 moves + 5 switches (in random battles, max 5 switches possible)

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")

    def encode_battle(self, battle: AbstractBattle) -> EncodedState:
        """Encode a battle state into tensors."""
        # Encode player team
        player_pokemon = self._encode_team(
            list(battle.team.values()),
            battle.active_pokemon,
            revealed=True,
        )

        # Encode opponent team
        opponent_pokemon = self._encode_team(
            list(battle.opponent_team.values()),
            battle.opponent_active_pokemon,
            revealed=False,
        )

        # Find active indices
        player_active_idx = self._get_active_index(
            list(battle.team.values()), battle.active_pokemon
        )
        opponent_active_idx = self._get_active_index(
            list(battle.opponent_team.values()), battle.opponent_active_pokemon
        )

        # Encode field state
        field_state = self._encode_field(battle)

        # Create action mask
        action_mask = self._create_action_mask(battle)

        return EncodedState(
            player_pokemon=torch.tensor(player_pokemon, dtype=torch.float32),
            opponent_pokemon=torch.tensor(opponent_pokemon, dtype=torch.float32),
            player_active_idx=player_active_idx,
            opponent_active_idx=opponent_active_idx,
            field_state=torch.tensor(field_state, dtype=torch.float32),
            action_mask=torch.tensor(action_mask, dtype=torch.float32),
        )

    def _encode_team(
        self,
        team: list[Pokemon],
        active: Pokemon | None,
        revealed: bool,
    ) -> "NDArray[np.float32]":
        """Encode a team of Pokemon."""
        encoded = np.zeros((6, self.POKEMON_FEATURES), dtype=np.float32)

        for i, pokemon in enumerate(team[:6]):
            encoded[i] = self._encode_pokemon(pokemon, pokemon == active, revealed)

        return encoded

    def _encode_pokemon(
        self,
        pokemon: Pokemon,
        is_active: bool,
        is_revealed: bool,
    ) -> "NDArray[np.float32]":
        """Encode a single Pokemon's state."""
        features: list[float] = []

        # HP fraction (0-1)
        hp_fraction = pokemon.current_hp_fraction if pokemon.current_hp else 0.0
        features.append(hp_fraction)

        # Type 1 one-hot
        type1_onehot = [0.0] * NUM_TYPES
        if pokemon.type_1:
            type_name = pokemon.type_1.name.lower()
            if type_name in TYPE_TO_IDX:
                type1_onehot[TYPE_TO_IDX[type_name]] = 1.0
        features.extend(type1_onehot)

        # Type 2 one-hot
        type2_onehot = [0.0] * NUM_TYPES
        if pokemon.type_2:
            type_name = pokemon.type_2.name.lower()
            if type_name in TYPE_TO_IDX:
                type2_onehot[TYPE_TO_IDX[type_name]] = 1.0
        features.extend(type2_onehot)

        # Status one-hot (+ none)
        status_onehot = [0.0] * (NUM_STATUSES + 1)
        if pokemon.status:
            status_name = pokemon.status.name.lower()
            if status_name in STATUS_TO_IDX:
                status_onehot[STATUS_TO_IDX[status_name]] = 1.0
        else:
            status_onehot[-1] = 1.0  # No status
        features.extend(status_onehot)

        # Stat boosts (-6 to +6, normalized to -1 to 1)
        boost_order = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
        for stat in boost_order:
            boost = pokemon.boosts.get(stat, 0)
            features.append(boost / 6.0)

        # Is active
        features.append(1.0 if is_active else 0.0)

        # Is fainted
        features.append(1.0 if pokemon.fainted else 0.0)

        # Is revealed (for opponent pokemon)
        features.append(1.0 if is_revealed else 0.0)

        # Encode moves (4 slots)
        moves = list(pokemon.moves.values())[:4]
        for i in range(4):
            if i < len(moves):
                features.extend(self._encode_move(moves[i]))
            else:
                # Empty move slot
                features.extend([0.0] * (NUM_TYPES + 3 + 1 + 1))

        return np.array(features, dtype=np.float32)

    def _encode_move(self, move: Move) -> list[float]:
        """Encode a single move."""
        features: list[float] = []

        # Type one-hot
        type_onehot = [0.0] * NUM_TYPES
        if move.type:
            type_name = move.type.name.lower()
            if type_name in TYPE_TO_IDX:
                type_onehot[TYPE_TO_IDX[type_name]] = 1.0
        features.extend(type_onehot)

        # Category one-hot
        category_onehot = [0.0] * 3
        if move.category:
            cat_name = move.category.name.lower()
            if cat_name in CATEGORY_TO_IDX:
                category_onehot[CATEGORY_TO_IDX[cat_name]] = 1.0
        features.extend(category_onehot)

        # Base power (normalized, max ~250)
        power = move.base_power / 250.0 if move.base_power else 0.0
        features.append(min(power, 1.0))

        # PP fraction
        if move.max_pp and move.max_pp > 0:
            pp_fraction = move.current_pp / move.max_pp if move.current_pp else 0.0
        else:
            pp_fraction = 1.0
        features.append(pp_fraction)

        return features

    def _encode_field(self, battle: AbstractBattle) -> "NDArray[np.float32]":
        """Encode field conditions."""
        features: list[float] = []

        # Weather one-hot (+ none)
        weather_onehot = [0.0] * (NUM_WEATHERS + 1)
        if battle.weather:
            for w in battle.weather:
                weather_name = w.name.lower()
                if weather_name in WEATHER_TO_IDX:
                    weather_onehot[WEATHER_TO_IDX[weather_name]] = 1.0
                    break
        else:
            weather_onehot[-1] = 1.0  # No weather
        features.extend(weather_onehot)

        # Terrain one-hot (+ none)
        terrain_onehot = [0.0] * (NUM_TERRAINS + 1)
        if battle.fields:
            for f in battle.fields:
                field_name = f.name.lower()
                if field_name in TERRAIN_TO_IDX:
                    terrain_onehot[TERRAIN_TO_IDX[field_name]] = 1.0
                    break
        else:
            terrain_onehot[-1] = 1.0  # No terrain
        features.extend(terrain_onehot)

        # Player side conditions
        player_conditions = [0.0] * NUM_SIDE_CONDITIONS
        for condition in battle.side_conditions:
            cond_name = condition.name.lower()
            if cond_name in SIDE_CONDITION_TO_IDX:
                player_conditions[SIDE_CONDITION_TO_IDX[cond_name]] = 1.0
        features.extend(player_conditions)

        # Opponent side conditions
        opponent_conditions = [0.0] * NUM_SIDE_CONDITIONS
        for condition in battle.opponent_side_conditions:
            cond_name = condition.name.lower()
            if cond_name in SIDE_CONDITION_TO_IDX:
                opponent_conditions[SIDE_CONDITION_TO_IDX[cond_name]] = 1.0
        features.extend(opponent_conditions)

        # Trick room
        trick_room = 0.0
        if Field.TRICK_ROOM in battle.fields:
            trick_room = 1.0
        features.append(trick_room)

        # Turn count (normalized, assume max 100 turns)
        turn_normalized = min(battle.turn / 100.0, 1.0)
        features.append(turn_normalized)

        return np.array(features, dtype=np.float32)

    def _get_active_index(
        self,
        team: list[Pokemon],
        active: Pokemon | None,
    ) -> int:
        """Get the index of the active Pokemon in the team list."""
        if active is None:
            return 0
        for i, pokemon in enumerate(team):
            if pokemon == active:
                return i
        return 0

    def _create_action_mask(self, battle: AbstractBattle) -> "NDArray[np.float32]":
        """Create a mask for legal actions.

        Actions:
        - 0-3: Moves
        - 4-8: Switch to team slot 0-4 (excluding active)
        """
        mask = np.zeros(self.NUM_ACTIONS, dtype=np.float32)

        # During a forced switch (Pokemon fainted), only switches are valid
        if not battle.force_switch:
            # Check available moves
            available_moves = battle.available_moves
            for i, move in enumerate(available_moves[:4]):
                mask[i] = 1.0

        # Check available switches
        available_switches = battle.available_switches
        team_list = [p for p in battle.team.values() if p != battle.active_pokemon]
        for pokemon in available_switches:
            if pokemon in team_list:
                idx = team_list.index(pokemon)
                if idx < 5:  # Max 5 switch targets
                    mask[4 + idx] = 1.0

        # If no legal actions, enable first move as last resort
        # This lets action_to_battle_order return None, triggering choose_random_move
        # which has proper handling for all edge cases including forced switches
        if mask.sum() == 0:
            mask[0] = 1.0

        return mask

    def action_to_battle_order(
        self,
        action: int,
        battle: AbstractBattle,
    ) -> BattleOrder | None:
        """Convert action index to poke-env battle order.

        Returns None if action is invalid.
        """
        # During forced switch, only switch actions are valid
        if battle.force_switch:
            if action < 4:
                # Model tried to use a move during forced switch - invalid
                return None
            # Fall through to switch handling below

        if action < 4:
            # Move action
            available_moves = battle.available_moves
            if action < len(available_moves):
                move = available_moves[action]
                return SingleBattleOrder(move)
        else:
            # Switch action
            switch_idx = action - 4
            available_switches = battle.available_switches
            team_list = [p for p in battle.team.values() if p != battle.active_pokemon]

            for pokemon in available_switches:
                if pokemon in team_list:
                    idx = team_list.index(pokemon)
                    if idx == switch_idx:
                        return SingleBattleOrder(pokemon)

        return None

    @property
    def state_dim(self) -> int:
        """Total flattened state dimension."""
        return (
            6 * self.POKEMON_FEATURES  # Player team
            + 6 * self.POKEMON_FEATURES  # Opponent team
            + self.FIELD_FEATURES
        )

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

# Volatile status effects (battle-turn-only, not persistent status)
VOLATILE_STATUSES = [
    "leechseed",
    "substitute",
    "confusion",
    "curse",
    "taunt",
    "encore",
    "disable",
    "yawn",
    "perishsong",
    "aquaring",
    "ingrain",
    "focusenergy",
]
NUM_VOLATILE_STATUSES = len(VOLATILE_STATUSES)

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

    # Move features per slot
    MOVE_FEATURES = (
        NUM_TYPES  # Type one-hot
        + 3  # Category one-hot (physical, special, status)
        + 1  # Power normalized
        + 1  # PP fraction
        + 1  # Is revealed (for opponent modeling)
    )

    # Feature dimensions
    POKEMON_FEATURES = (
        1  # HP fraction
        + NUM_TYPES  # Type 1 one-hot
        + NUM_TYPES  # Type 2 one-hot
        + NUM_STATUSES + 1  # Status one-hot (+ none)
        + 7  # Stat boosts (atk, def, spa, spd, spe, acc, eva)
        + 1  # Speed stat (normalized, for turn order prediction)
        + NUM_VOLATILE_STATUSES  # Volatile status flags
        + 1  # Is active
        + 1  # Is fainted
        + 1  # Is revealed (for opponent)
        # Opponent modeling features
        + 1  # Num moves revealed (0-1 normalized)
        + 1  # Could have more moves (1 if <4 revealed)
        + 1  # Ability revealed
        + 1  # Item revealed
        + 1  # Item consumed/knocked off
        + 4 * MOVE_FEATURES  # 4 moves with reveal flag
    )

    FIELD_FEATURES = (
        NUM_WEATHERS + 1  # Weather one-hot (+ none)
        + NUM_TERRAINS + 1  # Terrain one-hot (+ none)
        + NUM_SIDE_CONDITIONS  # Player side conditions
        + NUM_SIDE_CONDITIONS  # Opponent side conditions
        + 1  # Trick room active
        + 1  # Turn count (normalized)
        # Opponent modeling
        + 1  # Num opponent Pokemon revealed (0-1 normalized)
        + 1  # Num opponent Pokemon unrevealed (0-1 normalized)
        # Type effectiveness (for active moves vs opponent active)
        + 4  # Type effectiveness multiplier for each move slot (normalized 0-1)
        # Damage estimates (like MaxDamage calculates)
        + 4  # Estimated damage for each move slot (normalized 0-1)
    )

    NUM_ACTIONS = 13  # 4 moves + 5 switches + 4 tera moves

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
        """Encode a single Pokemon's state.

        For opponent Pokemon, we track what information has been revealed to us:
        - Which moves have been used/revealed
        - Whether ability/item have been revealed
        - How many unknowns remain
        """
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

        # Speed stat (normalized, for turn order prediction)
        # Base speed ranges roughly 5-180, so normalize by 200
        base_speed = pokemon.base_stats.get("spe", 100) if pokemon.base_stats else 100
        features.append(min(base_speed / 200.0, 1.0))

        # Volatile status flags
        effects = pokemon.effects if hasattr(pokemon, "effects") else {}
        for volatile in VOLATILE_STATUSES:
            # Check if the volatile effect is present
            has_volatile = any(
                e.name.lower() == volatile for e in effects
            ) if effects else False
            features.append(1.0 if has_volatile else 0.0)

        # Is active
        features.append(1.0 if is_active else 0.0)

        # Is fainted
        features.append(1.0 if pokemon.fainted else 0.0)

        # Is revealed (for opponent pokemon)
        features.append(1.0 if is_revealed else 0.0)

        # Opponent modeling features
        moves = list(pokemon.moves.values())[:4]
        num_moves_revealed = len(moves)

        # Num moves revealed (normalized 0-1, max 4 moves)
        features.append(num_moves_revealed / 4.0)

        # Could have more moves (1 if <4 revealed, signals uncertainty)
        features.append(1.0 if num_moves_revealed < 4 else 0.0)

        # Ability revealed (poke-env sets ability when it's been revealed in battle)
        ability_revealed = pokemon.ability is not None
        features.append(1.0 if ability_revealed else 0.0)

        # Item revealed (poke-env sets item when it's been revealed)
        item_revealed = pokemon.item is not None and pokemon.item != ""
        features.append(1.0 if item_revealed else 0.0)

        # Item consumed/knocked off (item was revealed but is now gone)
        # poke-env uses empty string for consumed/knocked off items
        item_consumed = pokemon.item == ""
        features.append(1.0 if item_consumed else 0.0)

        # Encode moves (4 slots) with reveal flags
        # For our own Pokemon, all moves are revealed (is_revealed=True)
        # For opponent Pokemon, moves in pokemon.moves have been revealed
        for i in range(4):
            if i < len(moves):
                # Move is revealed (it's in the moves dict because we've seen it)
                features.extend(self._encode_move(moves[i], is_revealed=True))
            else:
                # Empty/unknown move slot - all zeros including is_revealed=0
                features.extend([0.0] * self.MOVE_FEATURES)

        return np.array(features, dtype=np.float32)

    def _encode_move(self, move: Move, is_revealed: bool = True) -> list[float]:
        """Encode a single move.

        Args:
            move: The move to encode
            is_revealed: Whether this move has been revealed to us (for opponent modeling)
        """
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

        # Is revealed flag (for opponent modeling)
        features.append(1.0 if is_revealed else 0.0)

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

        # Opponent modeling - how many opponent Pokemon have we seen?
        num_opponent_revealed = len(battle.opponent_team)
        # In random battles, opponent has 6 Pokemon
        max_opponent_pokemon = 6

        # Num opponent Pokemon revealed (normalized 0-1)
        features.append(num_opponent_revealed / max_opponent_pokemon)

        # Num opponent Pokemon unrevealed (normalized 0-1)
        # This signals how much uncertainty remains about opponent's team
        num_unrevealed = max_opponent_pokemon - num_opponent_revealed
        features.append(num_unrevealed / max_opponent_pokemon)

        # Type effectiveness for each move slot against opponent active Pokemon
        # This helps the network understand which moves are super effective
        opponent_active = battle.opponent_active_pokemon
        available_moves = battle.available_moves if battle.available_moves else []
        for i in range(4):
            if i < len(available_moves) and opponent_active is not None:
                move = available_moves[i]
                # damage_multiplier returns 0.25, 0.5, 1, 2, or 4
                # Normalize: 0.25->0.0625, 0.5->0.125, 1->0.25, 2->0.5, 4->1.0
                multiplier = opponent_active.damage_multiplier(move)
                features.append(multiplier / 4.0)
            else:
                # No move in this slot or no opponent active - neutral effectiveness
                features.append(0.25)  # 1.0 / 4.0 = neutral

        # Damage estimates for each move (like MaxDamage calculates)
        # This gives the model the same information MaxDamage uses
        user_active = battle.active_pokemon
        for i in range(4):
            if i < len(available_moves) and opponent_active is not None and user_active is not None:
                move = available_moves[i]
                damage_estimate = self._estimate_damage(move, user_active, opponent_active)
                # Normalize: assume max reasonable damage is ~200 base power equivalent
                # after all multipliers, cap at 1.0
                features.append(min(damage_estimate / 200.0, 1.0))
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _estimate_damage(self, move: Move, attacker: Pokemon, defender: Pokemon) -> float:
        """Estimate damage for a move, similar to MaxDamage calculation.

        Returns a normalized damage value (not actual HP damage, but relative strength).
        """
        # Status moves do no damage
        if move.category.name.lower() == "status":
            return 0.0

        base_power = move.base_power or 0
        if base_power == 0:
            return 0.0

        # Type effectiveness (0.25, 0.5, 1, 2, or 4)
        type_mult = defender.damage_multiplier(move)

        # STAB (Same Type Attack Bonus) - 1.5x if move type matches attacker's type
        stab = 1.0
        if move.type:
            if move.type == attacker.type_1 or move.type == attacker.type_2:
                stab = 1.5

        # Get stats - use stats if available, fall back to base_stats
        # For opponent Pokemon, we may only have base_stats
        def get_stat(pokemon: Pokemon, stat: str, default: int = 100) -> int:
            if pokemon.stats:
                val = pokemon.stats.get(stat)
                if val is not None:
                    return val
            if pokemon.base_stats:
                val = pokemon.base_stats.get(stat)
                if val is not None:
                    return val
            return default

        # Attack vs Defense ratio
        # For physical moves: Attack vs Defense
        # For special moves: Special Attack vs Special Defense
        if move.category.name.lower() == "physical":
            atk_stat = get_stat(attacker, "atk")
            def_stat = get_stat(defender, "def")
            atk_boost_key = "atk"
            def_boost_key = "def"
        else:  # special
            atk_stat = get_stat(attacker, "spa")
            def_stat = get_stat(defender, "spd")
            atk_boost_key = "spa"
            def_boost_key = "spd"

        # Include stat boosts
        atk_boost = attacker.boosts.get(atk_boost_key, 0)
        def_boost = defender.boosts.get(def_boost_key, 0)

        # Boost multipliers: +1 = 1.5x, +2 = 2x, etc. (simplified)
        atk_mult = max(2, 2 + atk_boost) / 2 if atk_boost >= 0 else 2 / max(2, 2 - atk_boost)
        def_mult = max(2, 2 + def_boost) / 2 if def_boost >= 0 else 2 / max(2, 2 - def_boost)

        atk_effective = atk_stat * atk_mult
        def_effective = def_stat * def_mult

        # Avoid division by zero
        if def_effective == 0:
            def_effective = 1

        # Simplified damage formula (proportional, not exact Pokemon formula)
        damage = base_power * stab * type_mult * (atk_effective / def_effective)

        return damage

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
        - 9-12: Tera moves (same moves but with terastallization)
        """
        mask = np.zeros(self.NUM_ACTIONS, dtype=np.float32)

        # During a forced switch (Pokemon fainted), only switches are valid
        if not battle.force_switch:
            # Check available moves
            available_moves = battle.available_moves
            for i, move in enumerate(available_moves[:4]):
                mask[i] = 1.0

            # Tera moves available if we can still tera
            if battle.can_tera:
                for i, move in enumerate(available_moves[:4]):
                    mask[9 + i] = 1.0

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

        Actions:
        - 0-3: Regular moves
        - 4-8: Switch to team slot 0-4
        - 9-12: Tera moves (same moves but with terastallization)

        Returns None if action is invalid.
        """
        # During forced switch, only switch actions are valid
        if battle.force_switch:
            if action < 4 or action >= 9:
                # Model tried to use a move during forced switch - invalid
                return None
            # Fall through to switch handling below

        if action < 4:
            # Regular move action
            available_moves = battle.available_moves
            if action < len(available_moves):
                move = available_moves[action]
                return SingleBattleOrder(move)
        elif action < 9:
            # Switch action
            switch_idx = action - 4
            available_switches = battle.available_switches
            team_list = [p for p in battle.team.values() if p != battle.active_pokemon]

            for pokemon in available_switches:
                if pokemon in team_list:
                    idx = team_list.index(pokemon)
                    if idx == switch_idx:
                        return SingleBattleOrder(pokemon)
        else:
            # Tera move action (9-12)
            move_idx = action - 9
            available_moves = battle.available_moves
            if move_idx < len(available_moves) and battle.can_tera:
                move = available_moves[move_idx]
                return SingleBattleOrder(order=move, terastallize=True)
            elif move_idx < len(available_moves):
                # Can't tera anymore, fall back to regular move
                move = available_moves[move_idx]
                return SingleBattleOrder(move)

        return None

    @property
    def state_dim(self) -> int:
        """Total flattened state dimension."""
        return (
            6 * self.POKEMON_FEATURES  # Player team
            + 6 * self.POKEMON_FEATURES  # Opponent team
            + self.FIELD_FEATURES
        )

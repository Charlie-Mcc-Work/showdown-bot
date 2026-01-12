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
from poke_env.battle import AbstractBattle

from showdown_bot.ou.shared.embeddings import SharedEmbeddings
from showdown_bot.ou.shared.data_loader import PokemonDataLoader, UsageStatsLoader


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
        team_encoding = torch.zeros(6, self.POKEMON_FEATURES)

        for i, pokemon in enumerate(battle.team.values()):
            if i >= 6:
                break
            team_encoding[i] = self._encode_pokemon(pokemon, full_knowledge=True)

        return team_encoding

    def _encode_opponent_revealed(
        self, battle: AbstractBattle
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode opponent's revealed Pokemon.

        Returns:
            (encodings, mask) where mask is True for revealed Pokemon
        """
        revealed_encoding = torch.zeros(6, self.POKEMON_FEATURES)
        mask = torch.zeros(6, dtype=torch.bool)

        for i, pokemon in enumerate(battle.opponent_team.values()):
            if i >= 6:
                break
            revealed_encoding[i] = self._encode_pokemon(pokemon, full_knowledge=False)
            mask[i] = True

        return revealed_encoding, mask

    def _predict_opponent_unrevealed(
        self,
        battle: AbstractBattle,
        revealed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict unrevealed opponent Pokemon using usage stats.

        Uses:
        - Revealed teammates to predict common partners
        - Overall usage stats for the tier
        """
        predicted = torch.zeros(6, self.POKEMON_FEATURES)

        # TODO: Implement actual prediction based on:
        # 1. Revealed Pokemon's common teammates
        # 2. Team archetype detection
        # 3. Usage rate priors

        return predicted

    def _encode_pokemon(
        self,
        pokemon: Any,  # poke_env Pokemon object
        full_knowledge: bool = False,
    ) -> torch.Tensor:
        """Encode a single Pokemon.

        Args:
            pokemon: The poke-env Pokemon object
            full_knowledge: If True, use full info (our Pokemon). If False,
                          only use revealed info (opponent Pokemon).

        Returns:
            (POKEMON_FEATURES,) tensor
        """
        features = []

        # Species (one-hot or embedding lookup)
        species_id = self.data_loader.get_species_id(pokemon.species)
        # For now, just use a normalized ID as placeholder
        features.append(species_id / 1000.0)

        # HP
        features.append(pokemon.current_hp_fraction)

        # Status
        status_vec = [0.0] * 7  # BRN, FRZ, PAR, PSN, SLP, TOX, healthy
        if pokemon.status:
            status_map = {"brn": 0, "frz": 1, "par": 2, "psn": 3, "slp": 4, "tox": 5}
            if pokemon.status.name.lower() in status_map:
                status_vec[status_map[pokemon.status.name.lower()]] = 1.0
        else:
            status_vec[6] = 1.0
        features.extend(status_vec)

        # Boosts
        boost_order = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
        for stat in boost_order:
            boost = pokemon.boosts.get(stat, 0)
            features.append(boost / 6.0)

        # Types
        type1_id = self.data_loader.get_type_id(str(pokemon.type_1))
        type2_id = self.data_loader.get_type_id(str(pokemon.type_2)) if pokemon.type_2 else 0
        features.append(type1_id / 18.0)
        features.append(type2_id / 18.0)

        # Pad or truncate to POKEMON_FEATURES
        while len(features) < self.POKEMON_FEATURES:
            features.append(0.0)

        return torch.tensor(features[:self.POKEMON_FEATURES], dtype=torch.float32)

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
        features = []

        # Weather (5 types + none)
        weather_vec = [0.0] * 6
        if battle.weather:
            weather_map = {
                "sunnyday": 0, "raindance": 1, "sandstorm": 2,
                "hail": 3, "snow": 3, "desolateland": 0,
                "primordialsea": 1, "deltastream": 4,
            }
            for w in battle.weather:
                w_lower = str(w).lower()
                if w_lower in weather_map:
                    weather_vec[weather_map[w_lower]] = 1.0
        else:
            weather_vec[5] = 1.0
        features.extend(weather_vec)

        # Terrain (5 types + none)
        terrain_vec = [0.0] * 5
        if battle.fields:
            terrain_map = {
                "electricterrain": 0, "grassyterrain": 1,
                "mistyterrain": 2, "psychicterrain": 3,
            }
            for f in battle.fields:
                f_lower = str(f).lower()
                if f_lower in terrain_map:
                    terrain_vec[terrain_map[f_lower]] = 1.0
        else:
            terrain_vec[4] = 1.0
        features.extend(terrain_vec)

        # Our side hazards
        side = battle.side_conditions
        features.append(1.0 if "spikes" in str(side).lower() else 0.0)
        features.append(1.0 if "stealthrock" in str(side).lower() else 0.0)
        features.append(1.0 if "toxicspikes" in str(side).lower() else 0.0)
        features.append(1.0 if "stickyweb" in str(side).lower() else 0.0)

        # Opponent side hazards
        opp_side = battle.opponent_side_conditions
        features.append(1.0 if "spikes" in str(opp_side).lower() else 0.0)
        features.append(1.0 if "stealthrock" in str(opp_side).lower() else 0.0)
        features.append(1.0 if "toxicspikes" in str(opp_side).lower() else 0.0)
        features.append(1.0 if "stickyweb" in str(opp_side).lower() else 0.0)

        # Screens
        features.append(1.0 if "reflect" in str(side).lower() else 0.0)
        features.append(1.0 if "lightscreen" in str(side).lower() else 0.0)
        features.append(1.0 if "auroraveil" in str(side).lower() else 0.0)

        # Opponent screens
        features.append(1.0 if "reflect" in str(opp_side).lower() else 0.0)
        features.append(1.0 if "lightscreen" in str(opp_side).lower() else 0.0)
        features.append(1.0 if "auroraveil" in str(opp_side).lower() else 0.0)

        # Trick room
        features.append(1.0 if battle.fields and "trickroom" in str(battle.fields).lower() else 0.0)

        # Tailwind
        features.append(1.0 if "tailwind" in str(side).lower() else 0.0)
        features.append(1.0 if "tailwind" in str(opp_side).lower() else 0.0)

        # Pad to FIELD_FEATURES
        while len(features) < self.FIELD_FEATURES:
            features.append(0.0)

        return torch.tensor(features[:self.FIELD_FEATURES], dtype=torch.float32)

    def _create_action_mask(self, battle: AbstractBattle) -> torch.Tensor:
        """Create mask for legal actions.

        Actions:
        - 0-3: Regular moves
        - 4-7: Tera moves (same moves but with Tera)
        - 8-12: Switch to Pokemon 0-4
        """
        mask = torch.zeros(self.NUM_ACTIONS, dtype=torch.float32)

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
            for i, move in enumerate(battle.available_moves[:4]):
                mask[i] = 1.0

                # Tera moves available if we can tera
                if battle.can_tera:
                    mask[4 + i] = 1.0

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

        return mask

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
        from poke_env.player.battle_order import BattleOrder

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
                return BattleOrder(move)

        elif action < 8:
            # Tera move action (same moves but with tera)
            move_idx = action - 4
            available_moves = battle.available_moves

            if move_idx < len(available_moves) and battle.can_tera:
                move = available_moves[move_idx]
                return BattleOrder(move, terastallize=True)
            elif move_idx < len(available_moves):
                # Can't tera, fall back to regular move
                return BattleOrder(available_moves[move_idx])

        else:
            # Switch action (8-12 -> switch to team slot 0-4)
            switch_idx = action - 8
            available_switches = battle.available_switches
            team_list = [p for p in battle.team.values() if p != battle.active_pokemon]

            for pokemon in available_switches:
                if pokemon in team_list:
                    idx = team_list.index(pokemon)
                    if idx == switch_idx:
                        return BattleOrder(pokemon)

        return None

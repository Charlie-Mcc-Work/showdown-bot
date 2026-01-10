"""Comprehensive tests for the state encoder."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, PropertyMock

from showdown_bot.environment.state_encoder import (
    StateEncoder,
    EncodedState,
    TYPE_TO_IDX,
    STATUS_TO_IDX,
    WEATHER_TO_IDX,
    TERRAIN_TO_IDX,
    SIDE_CONDITION_TO_IDX,
    NUM_TYPES,
    NUM_STATUSES,
)


class TestStateEncoderConstants:
    """Test encoder constants and mappings."""

    def test_type_mapping_completeness(self):
        """Verify all 18 Pokemon types are mapped."""
        assert len(TYPE_TO_IDX) == 18
        expected_types = [
            "normal", "fire", "water", "electric", "grass", "ice",
            "fighting", "poison", "ground", "flying", "psychic", "bug",
            "rock", "ghost", "dragon", "dark", "steel", "fairy"
        ]
        for t in expected_types:
            assert t in TYPE_TO_IDX

    def test_status_mapping_completeness(self):
        """Verify all status conditions are mapped."""
        expected_statuses = ["brn", "frz", "par", "psn", "slp", "tox"]
        for s in expected_statuses:
            assert s in STATUS_TO_IDX

    def test_weather_mapping(self):
        """Verify weather conditions are mapped."""
        expected_weathers = ["sunnyday", "raindance", "sandstorm", "snow", "hail"]
        for w in expected_weathers:
            assert w in WEATHER_TO_IDX

    def test_terrain_mapping(self):
        """Verify terrain conditions are mapped."""
        expected_terrains = ["electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"]
        for t in expected_terrains:
            assert t in TERRAIN_TO_IDX


class TestStateEncoderDimensions:
    """Test encoder output dimensions."""

    def test_pokemon_features_dimension(self):
        """Verify Pokemon feature dimension calculation."""
        encoder = StateEncoder()
        # HP + Type1 + Type2 + Status + Boosts + IsActive + IsFainted + IsRevealed + 4*MoveFeatures
        expected = (
            1  # HP
            + NUM_TYPES  # Type 1
            + NUM_TYPES  # Type 2
            + NUM_STATUSES + 1  # Status + none
            + 7  # Boosts
            + 1  # Is active
            + 1  # Is fainted
            + 1  # Is revealed
            + 4 * (NUM_TYPES + 3 + 1 + 1)  # 4 moves
        )
        assert StateEncoder.POKEMON_FEATURES == expected

    def test_field_features_dimension(self):
        """Verify field feature dimension calculation."""
        # Weather + Terrain + Player hazards + Opponent hazards + Trick room + Turn
        expected = (
            len(WEATHER_TO_IDX) + 1  # Weather + none
            + len(TERRAIN_TO_IDX) + 1  # Terrain + none
            + len(SIDE_CONDITION_TO_IDX)  # Player side
            + len(SIDE_CONDITION_TO_IDX)  # Opponent side
            + 1  # Trick room
            + 1  # Turn
        )
        assert StateEncoder.FIELD_FEATURES == expected

    def test_num_actions(self):
        """Verify action space size."""
        assert StateEncoder.NUM_ACTIONS == 9  # 4 moves + 5 switches


class TestStateEncoderInit:
    """Test encoder initialization."""

    def test_default_device(self):
        """Test default device is CPU."""
        encoder = StateEncoder()
        assert encoder.device == torch.device("cpu")

    def test_custom_device(self):
        """Test custom device setting."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = StateEncoder(device=device)
        assert encoder.device == device

    def test_state_dim_property(self):
        """Test total state dimension calculation."""
        encoder = StateEncoder()
        expected = (
            6 * StateEncoder.POKEMON_FEATURES  # Player team
            + 6 * StateEncoder.POKEMON_FEATURES  # Opponent team
            + StateEncoder.FIELD_FEATURES
        )
        assert encoder.state_dim == expected


class TestPokemonEncoding:
    """Test individual Pokemon encoding."""

    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    @pytest.fixture
    def mock_pokemon(self):
        """Create a mock Pokemon with typical attributes."""
        pokemon = MagicMock()
        pokemon.current_hp_fraction = 0.75
        pokemon.type_1 = MagicMock()
        pokemon.type_1.name = "FIRE"
        pokemon.type_2 = MagicMock()
        pokemon.type_2.name = "FLYING"
        pokemon.status = None
        pokemon.boosts = {"atk": 1, "def": 0, "spa": -1, "spd": 0, "spe": 2, "accuracy": 0, "evasion": 0}
        pokemon.fainted = False

        # Mock moves
        move1 = MagicMock()
        move1.type = MagicMock()
        move1.type.name = "FIRE"
        move1.category = MagicMock()
        move1.category.name = "SPECIAL"
        move1.base_power = 90
        move1.current_pp = 10
        move1.max_pp = 15

        move2 = MagicMock()
        move2.type = MagicMock()
        move2.type.name = "FLYING"
        move2.category = MagicMock()
        move2.category.name = "PHYSICAL"
        move2.base_power = 120
        move2.current_pp = 5
        move2.max_pp = 5

        pokemon.moves = {"flamethrower": move1, "bravebird": move2}

        return pokemon

    def test_encode_pokemon_hp(self, encoder, mock_pokemon):
        """Test HP encoding."""
        encoded = encoder._encode_pokemon(mock_pokemon, is_active=True, is_revealed=True)
        assert encoded[0] == pytest.approx(0.75)

    def test_encode_pokemon_types(self, encoder, mock_pokemon):
        """Test type encoding."""
        encoded = encoder._encode_pokemon(mock_pokemon, is_active=True, is_revealed=True)

        # Type 1 (fire) should be at index TYPE_TO_IDX["fire"] + 1 (after HP)
        fire_idx = 1 + TYPE_TO_IDX["fire"]
        assert encoded[fire_idx] == 1.0

        # Type 2 (flying) should be at index TYPE_TO_IDX["flying"] + 1 + NUM_TYPES
        flying_idx = 1 + NUM_TYPES + TYPE_TO_IDX["flying"]
        assert encoded[flying_idx] == 1.0

    def test_encode_pokemon_status(self, encoder, mock_pokemon):
        """Test status encoding when no status."""
        encoded = encoder._encode_pokemon(mock_pokemon, is_active=True, is_revealed=True)

        # Status starts after HP + Type1 + Type2
        status_start = 1 + NUM_TYPES + NUM_TYPES
        # Last status index should be 1.0 (no status)
        assert encoded[status_start + NUM_STATUSES] == 1.0

    def test_encode_pokemon_with_status(self, encoder, mock_pokemon):
        """Test status encoding with burn."""
        mock_pokemon.status = MagicMock()
        mock_pokemon.status.name = "BRN"

        encoded = encoder._encode_pokemon(mock_pokemon, is_active=True, is_revealed=True)

        status_start = 1 + NUM_TYPES + NUM_TYPES
        brn_idx = status_start + STATUS_TO_IDX["brn"]
        assert encoded[brn_idx] == 1.0

    def test_encode_pokemon_boosts(self, encoder, mock_pokemon):
        """Test stat boost encoding."""
        encoded = encoder._encode_pokemon(mock_pokemon, is_active=True, is_revealed=True)

        # Boosts start after HP + Type1 + Type2 + Status
        boost_start = 1 + NUM_TYPES + NUM_TYPES + NUM_STATUSES + 1

        # atk boost of +1 should be 1/6
        assert encoded[boost_start] == pytest.approx(1/6)
        # spa boost of -1 should be -1/6
        assert encoded[boost_start + 2] == pytest.approx(-1/6)
        # spe boost of +2 should be 2/6
        assert encoded[boost_start + 4] == pytest.approx(2/6)

    def test_encode_pokemon_is_active(self, encoder, mock_pokemon):
        """Test is_active flag encoding."""
        encoded_active = encoder._encode_pokemon(mock_pokemon, is_active=True, is_revealed=True)
        encoded_inactive = encoder._encode_pokemon(mock_pokemon, is_active=False, is_revealed=True)

        # is_active is after boosts
        is_active_idx = 1 + NUM_TYPES + NUM_TYPES + NUM_STATUSES + 1 + 7
        assert encoded_active[is_active_idx] == 1.0
        assert encoded_inactive[is_active_idx] == 0.0

    def test_encode_pokemon_output_shape(self, encoder, mock_pokemon):
        """Test output shape matches expected features."""
        encoded = encoder._encode_pokemon(mock_pokemon, is_active=True, is_revealed=True)
        assert len(encoded) == StateEncoder.POKEMON_FEATURES


class TestMoveEncoding:
    """Test move encoding."""

    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    @pytest.fixture
    def mock_move(self):
        move = MagicMock()
        move.type = MagicMock()
        move.type.name = "ELECTRIC"
        move.category = MagicMock()
        move.category.name = "SPECIAL"
        move.base_power = 90
        move.current_pp = 10
        move.max_pp = 15
        return move

    def test_encode_move_type(self, encoder, mock_move):
        """Test move type encoding."""
        features = encoder._encode_move(mock_move)

        electric_idx = TYPE_TO_IDX["electric"]
        assert features[electric_idx] == 1.0

    def test_encode_move_category(self, encoder, mock_move):
        """Test move category encoding."""
        features = encoder._encode_move(mock_move)

        # Category starts after type
        category_start = NUM_TYPES
        # Special is index 1
        assert features[category_start + 1] == 1.0

    def test_encode_move_power(self, encoder, mock_move):
        """Test move power encoding."""
        features = encoder._encode_move(mock_move)

        power_idx = NUM_TYPES + 3
        assert features[power_idx] == pytest.approx(90 / 250)

    def test_encode_move_pp(self, encoder, mock_move):
        """Test move PP encoding."""
        features = encoder._encode_move(mock_move)

        pp_idx = NUM_TYPES + 3 + 1
        assert features[pp_idx] == pytest.approx(10 / 15)

    def test_encode_move_feature_length(self, encoder, mock_move):
        """Test move feature length."""
        features = encoder._encode_move(mock_move)
        expected_length = NUM_TYPES + 3 + 1 + 1  # type + category + power + pp
        assert len(features) == expected_length


class TestFieldEncoding:
    """Test field state encoding."""

    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    @pytest.fixture
    def mock_battle(self):
        battle = MagicMock()
        battle.weather = {}
        battle.fields = {}
        battle.side_conditions = {}
        battle.opponent_side_conditions = {}
        battle.turn = 10
        return battle

    def test_encode_field_no_conditions(self, encoder, mock_battle):
        """Test field encoding with no active conditions."""
        encoded = encoder._encode_field(mock_battle)
        assert len(encoded) == StateEncoder.FIELD_FEATURES

    def test_encode_field_turn_normalization(self, encoder, mock_battle):
        """Test turn count normalization."""
        mock_battle.turn = 50

        encoded = encoder._encode_field(mock_battle)

        # Turn is last feature
        assert encoded[-1] == pytest.approx(0.5)

    def test_encode_field_turn_capped(self, encoder, mock_battle):
        """Test turn count is capped at 1.0."""
        mock_battle.turn = 150

        encoded = encoder._encode_field(mock_battle)

        assert encoded[-1] == pytest.approx(1.0)


class TestActionMask:
    """Test action mask creation."""

    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    @pytest.fixture
    def mock_battle(self):
        battle = MagicMock()

        # Mock force_switch (required for mask logic)
        battle.force_switch = False

        # Mock available moves
        move1 = MagicMock()
        move2 = MagicMock()
        battle.available_moves = [move1, move2]

        # Mock team and switches
        active = MagicMock()
        poke1 = MagicMock()
        poke2 = MagicMock()
        poke3 = MagicMock()

        battle.active_pokemon = active
        battle.team = {"active": active, "poke1": poke1, "poke2": poke2, "poke3": poke3}
        battle.available_switches = [poke1, poke2]

        return battle

    def test_action_mask_moves(self, encoder, mock_battle):
        """Test action mask has correct move slots."""
        mask = encoder._create_action_mask(mock_battle)

        # First 2 moves should be available
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 0.0
        assert mask[3] == 0.0

    def test_action_mask_switches(self, encoder, mock_battle):
        """Test action mask has correct switch slots."""
        mask = encoder._create_action_mask(mock_battle)

        # 2 switches should be available
        assert mask[4] == 1.0
        assert mask[5] == 1.0
        assert mask[6] == 0.0

    def test_action_mask_shape(self, encoder, mock_battle):
        """Test action mask shape."""
        mask = encoder._create_action_mask(mock_battle)
        assert len(mask) == StateEncoder.NUM_ACTIONS


class TestActionToBattleOrder:
    """Test action to battle order conversion."""

    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    @pytest.fixture
    def mock_battle(self):
        battle = MagicMock()

        # Mock force_switch (required for action_to_battle_order logic)
        battle.force_switch = False

        move1 = MagicMock()
        move2 = MagicMock()
        battle.available_moves = [move1, move2]

        active = MagicMock()
        poke1 = MagicMock()
        poke2 = MagicMock()

        battle.active_pokemon = active
        battle.team = {"active": active, "poke1": poke1, "poke2": poke2}
        battle.available_switches = [poke1]

        return battle

    def test_action_to_move(self, encoder, mock_battle):
        """Test converting move action to order."""
        result = encoder.action_to_battle_order(0, mock_battle)

        # Should return a SingleBattleOrder for the first move
        assert result is not None
        assert result.order == mock_battle.available_moves[0]

    def test_action_to_switch(self, encoder, mock_battle):
        """Test converting switch action to order."""
        result = encoder.action_to_battle_order(4, mock_battle)

        # Should return a SingleBattleOrder for the switch target
        assert result is not None
        # The switch target should be poke1 (first in available_switches)
        assert result.order == mock_battle.available_switches[0]

    def test_invalid_move_action(self, encoder, mock_battle):
        """Test invalid move index returns None."""
        result = encoder.action_to_battle_order(3, mock_battle)  # Only 2 moves available
        assert result is None

    def test_invalid_switch_action(self, encoder, mock_battle):
        """Test invalid switch index returns None."""
        result = encoder.action_to_battle_order(5, mock_battle)  # Only 1 switch available
        assert result is None


class TestEncodedState:
    """Test EncodedState dataclass."""

    def test_to_device(self):
        """Test moving encoded state to device."""
        state = EncodedState(
            player_pokemon=torch.zeros(6, 10),
            opponent_pokemon=torch.zeros(6, 10),
            player_active_idx=0,
            opponent_active_idx=0,
            field_state=torch.zeros(20),
            action_mask=torch.zeros(9),
        )

        device = torch.device("cpu")
        moved = state.to_device(device)

        assert moved.player_pokemon.device == device
        assert moved.opponent_pokemon.device == device
        assert moved.field_state.device == device
        assert moved.action_mask.device == device

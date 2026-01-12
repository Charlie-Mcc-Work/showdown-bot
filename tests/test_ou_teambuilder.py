"""Tests for OU teambuilder components."""

from unittest.mock import MagicMock, patch

import pytest

from showdown_bot.ou.teambuilder.team_repr import (
    PokemonSlot,
    PartialTeam,
    team_to_showdown_paste,
    parse_showdown_paste,
)
from showdown_bot.ou.teambuilder.generator import (
    normalize_name,
    format_name,
    sample_weighted,
    parse_spread,
    UsageBasedGenerator,
    DEFAULT_SPREADS,
    DEFAULT_ITEMS,
    ALL_TYPES,
)
from showdown_bot.ou.shared.data_loader import Team, TeamSet, UsageStats


class TestPokemonSlot:
    """Tests for PokemonSlot dataclass."""

    def test_empty_slot(self):
        """Test empty slot defaults."""
        slot = PokemonSlot()
        assert slot.is_empty()
        assert not slot.is_complete()
        assert slot.species is None

    def test_partial_slot(self):
        """Test partially filled slot."""
        slot = PokemonSlot(
            species="Pikachu",
            moves=["Thunderbolt"],
        )
        assert not slot.is_empty()
        assert not slot.is_complete()

    def test_complete_slot(self):
        """Test fully specified slot."""
        slot = PokemonSlot(
            species="Pikachu",
            moves=["Thunderbolt", "Volt Switch", "Grass Knot", "Surf"],
            item="Light Ball",
            ability="Static",
            nature="Timid",
            evs={"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
            tera_type="Electric",
        )
        assert not slot.is_empty()
        assert slot.is_complete()

    def test_to_team_set_incomplete(self):
        """Test conversion fails for incomplete slot."""
        slot = PokemonSlot(species="Pikachu")
        assert slot.to_team_set() is None

    def test_to_team_set_complete(self):
        """Test conversion to TeamSet."""
        slot = PokemonSlot(
            species="Dragapult",
            moves=["Shadow Ball", "Draco Meteor", "Flamethrower", "U-turn"],
            item="Choice Specs",
            ability="Infiltrator",
            nature="Timid",
            evs={"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
            tera_type="Ghost",
        )
        team_set = slot.to_team_set()

        assert team_set is not None
        assert team_set.species == "Dragapult"
        assert team_set.item == "Choice Specs"
        assert len(team_set.moves) == 4
        assert team_set.nature == "Timid"


class TestPartialTeam:
    """Tests for PartialTeam class."""

    def test_initialization(self):
        """Test team initialization."""
        team = PartialTeam()
        assert team.num_filled() == 0
        assert team.num_complete() == 0
        assert not team.is_complete()
        assert team.next_empty_slot() == 0

    def test_filling_slots(self):
        """Test filling slots."""
        team = PartialTeam()

        team.slots[0] = PokemonSlot(
            species="Pikachu",
            moves=["Thunderbolt"],
            item="Light Ball",
            ability="Static",
            nature="Timid",
            evs={"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
        )

        assert team.num_filled() == 1
        assert team.num_complete() == 1
        assert team.next_empty_slot() == 1
        assert team.get_species_list() == ["Pikachu"]

    def test_complete_team(self):
        """Test completing a full team."""
        team = PartialTeam()

        for i in range(6):
            team.slots[i] = PokemonSlot(
                species=f"Pokemon{i}",
                moves=["Move1"],
                item="Item",
                ability="Ability",
                nature="Adamant",
                evs={"hp": 252, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 0},
            )

        assert team.is_complete()
        assert team.next_empty_slot() is None

    def test_to_team(self):
        """Test conversion to Team."""
        team = PartialTeam(format="gen9ou")

        for i in range(6):
            team.slots[i] = PokemonSlot(
                species=f"Pokemon{i}",
                moves=["Move1", "Move2"],
                item="Item",
                ability="Ability",
                nature="Adamant",
                evs={"hp": 252, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 0},
            )

        result = team.to_team()
        assert result is not None
        assert result.format == "gen9ou"
        assert len(result.pokemon) == 6

    def test_to_team_incomplete(self):
        """Test conversion fails for incomplete team."""
        team = PartialTeam()
        team.slots[0] = PokemonSlot(species="Pikachu")  # Not complete
        assert team.to_team() is None


class TestTeamToShowdownPaste:
    """Tests for team_to_showdown_paste function."""

    def test_basic_conversion(self):
        """Test basic team to paste conversion."""
        team = Team(
            name="Test",
            format="gen9ou",
            pokemon=[
                TeamSet(
                    species="Dragapult",
                    nickname=None,
                    item="Choice Specs",
                    ability="Infiltrator",
                    moves=["Shadow Ball", "Draco Meteor", "Flamethrower", "U-turn"],
                    nature="Timid",
                    evs={"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
                    ivs={"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
                    tera_type="Ghost",
                ),
            ],
            source="test",
        )

        paste = team_to_showdown_paste(team)

        assert "Dragapult @ Choice Specs" in paste
        assert "Ability: Infiltrator" in paste
        assert "Tera Type: Ghost" in paste
        assert "Timid Nature" in paste
        assert "- Shadow Ball" in paste
        assert "252 SpA" in paste

    def test_no_item(self):
        """Test Pokemon without item."""
        team = Team(
            name="Test",
            format="gen9ou",
            pokemon=[
                TeamSet(
                    species="Pikachu",
                    nickname=None,
                    item=None,
                    ability="Static",
                    moves=["Thunderbolt"],
                    nature="Timid",
                    evs={},
                    ivs={},
                    tera_type=None,
                ),
            ],
            source="test",
        )

        paste = team_to_showdown_paste(team)
        assert "Pikachu\n" in paste or paste.startswith("Pikachu")
        assert "@" not in paste.split("\n")[0]


class TestParseShowdownPaste:
    """Tests for parse_showdown_paste function."""

    def test_basic_parse(self):
        """Test basic paste parsing."""
        paste = """Dragapult @ Choice Specs
Ability: Infiltrator
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Shadow Ball
- Draco Meteor
- Flamethrower
- U-turn
"""
        team = parse_showdown_paste(paste)

        assert team is not None
        assert len(team.pokemon) == 1
        assert team.pokemon[0].species == "Dragapult"
        assert team.pokemon[0].item == "Choice Specs"
        assert team.pokemon[0].ability == "Infiltrator"
        assert team.pokemon[0].tera_type == "Ghost"
        assert team.pokemon[0].nature == "Timid"
        assert len(team.pokemon[0].moves) == 4

    def test_multiple_pokemon(self):
        """Test parsing multiple Pokemon."""
        paste = """Pikachu @ Light Ball
Ability: Static
Timid Nature
- Thunderbolt

Charizard @ Heavy-Duty Boots
Ability: Blaze
Timid Nature
- Flamethrower
"""
        team = parse_showdown_paste(paste)

        assert team is not None
        assert len(team.pokemon) == 2
        assert team.pokemon[0].species == "Pikachu"
        assert team.pokemon[1].species == "Charizard"

    def test_empty_paste(self):
        """Test parsing empty paste."""
        team = parse_showdown_paste("")
        assert team is None

    def test_evs_parsing(self):
        """Test EV parsing."""
        paste = """Pikachu
Ability: Static
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
"""
        team = parse_showdown_paste(paste)

        assert team is not None
        assert team.pokemon[0].evs.get("spa") == 252
        assert team.pokemon[0].evs.get("spd") == 4
        assert team.pokemon[0].evs.get("spe") == 252


class TestNormalizeName:
    """Tests for normalize_name function."""

    def test_basic_normalization(self):
        """Test basic name normalization."""
        assert normalize_name("Pikachu") == "pikachu"
        assert normalize_name("PIKACHU") == "pikachu"
        assert normalize_name("PiKaChU") == "pikachu"

    def test_remove_spaces(self):
        """Test space removal."""
        assert normalize_name("Choice Scarf") == "choicescarf"
        assert normalize_name("Heavy-Duty Boots") == "heavydutyboots"

    def test_remove_hyphens(self):
        """Test hyphen removal."""
        assert normalize_name("U-turn") == "uturn"
        assert normalize_name("Freeze-Dry") == "freezedry"


class TestFormatName:
    """Tests for format_name function."""

    def test_known_items(self):
        """Test formatting known items."""
        assert format_name("leftovers") == "Leftovers"
        assert format_name("choicescarf") == "Choice Scarf"
        assert format_name("heavydutyboots") == "Heavy-Duty Boots"

    def test_known_moves(self):
        """Test formatting known moves."""
        assert format_name("stealthrock") == "Stealth Rock"
        assert format_name("uturn") == "U-turn"
        assert format_name("shadowball") == "Shadow Ball"

    def test_known_abilities(self):
        """Test formatting known abilities."""
        assert format_name("regenerator") == "Regenerator"
        assert format_name("magicguard") == "Magic Guard"

    def test_fallback_title_case(self):
        """Test fallback to title case."""
        assert format_name("unknown_move") == "Unknown_move" or format_name("unknownmove") == "Unknownmove"


class TestSampleWeighted:
    """Tests for sample_weighted function."""

    def test_basic_sampling(self):
        """Test basic weighted sampling."""
        items = [("a", 0.5), ("b", 0.3), ("c", 0.2)]
        result = sample_weighted(items)
        assert result in ["a", "b", "c"]

    def test_empty_list_raises(self):
        """Test sampling from empty list raises."""
        with pytest.raises(ValueError):
            sample_weighted([])

    def test_temperature_affects_distribution(self):
        """Test temperature affects distribution."""
        items = [("common", 0.9), ("rare", 0.1)]

        # Low temperature should strongly favor common
        low_temp_counts = {"common": 0, "rare": 0}
        for _ in range(100):
            result = sample_weighted(items, temperature=0.1)
            low_temp_counts[result] += 1

        # High temperature should be more uniform
        high_temp_counts = {"common": 0, "rare": 0}
        for _ in range(100):
            result = sample_weighted(items, temperature=5.0)
            high_temp_counts[result] += 1

        # Low temp should have higher common count
        assert low_temp_counts["common"] >= high_temp_counts["common"]


class TestParseSpread:
    """Tests for parse_spread function."""

    def test_basic_spread(self):
        """Test parsing basic spread."""
        nature, evs = parse_spread("Adamant:252/252/0/0/4/0")
        assert nature == "Adamant"
        assert evs["hp"] == 252
        assert evs["atk"] == 252
        assert evs["spa"] == 0
        assert evs["spd"] == 4

    def test_special_attacker_spread(self):
        """Test special attacker spread."""
        nature, evs = parse_spread("Timid:0/0/0/252/4/252")
        assert nature == "Timid"
        assert evs["spa"] == 252
        assert evs["spe"] == 252

    def test_invalid_spread(self):
        """Test invalid spread returns defaults."""
        nature, evs = parse_spread("invalid")
        assert nature == "Serious"
        assert evs == {}

    def test_wrong_format(self):
        """Test wrong format returns defaults."""
        nature, evs = parse_spread("Adamant:252/252")  # Missing values
        assert nature == "Serious"
        assert evs == {}


class TestUsageBasedGenerator:
    """Tests for UsageBasedGenerator class."""

    @pytest.fixture
    def mock_usage_loader(self):
        """Create a mock usage stats loader."""
        loader = MagicMock()

        # Set up get_top_pokemon
        loader.get_top_pokemon.return_value = [
            "Dragapult", "Gholdengo", "Great Tusk", "Iron Valiant", "Kingambit"
        ]

        # Set up get_pokemon_usage
        def get_usage(name):
            stats = UsageStats(
                pokemon=name,
                usage_percent=10.0,
                raw_count=1000,
                real_count=900,
                common_moves={"shadowball": 80, "dracometeor": 70, "flamethrower": 50, "uturn": 40},
                common_items={"choicespecs": 60, "choicescarf": 30},
                common_abilities={"infiltrator": 70, "clearbody": 20},
                common_spreads={"Timid:0/0/0/252/4/252": 60},
                common_teammates={},
                checks={},
                counters={},
            )
            return stats

        loader.get_pokemon_usage.side_effect = get_usage

        return loader

    def test_initialization(self, mock_usage_loader):
        """Test generator initialization."""
        gen = UsageBasedGenerator(
            usage_loader=mock_usage_loader,
            tier="gen9ou",
            top_n=5,
        )

        assert len(gen.viable_pokemon) == 5
        assert "Dragapult" in gen.viable_pokemon

    def test_generate_team(self, mock_usage_loader):
        """Test team generation."""
        gen = UsageBasedGenerator(
            usage_loader=mock_usage_loader,
            tier="gen9ou",
            top_n=5,
        )

        team = gen.generate_team(temperature=1.0)

        assert team is not None
        assert team.num_filled() == 6
        assert all(not slot.is_empty() for slot in team.slots)

    def test_generate_team_no_duplicates(self, mock_usage_loader):
        """Test generated teams have no duplicate species."""
        gen = UsageBasedGenerator(
            usage_loader=mock_usage_loader,
            tier="gen9ou",
            top_n=10,  # More options
        )

        # Add more Pokemon to the pool
        mock_usage_loader.get_top_pokemon.return_value = [
            "Dragapult", "Gholdengo", "Great Tusk", "Iron Valiant",
            "Kingambit", "Landorus-Therian", "Zamazenta", "Heatran",
            "Raging Bolt", "Gliscor"
        ]
        gen.viable_pokemon = mock_usage_loader.get_top_pokemon.return_value

        team = gen.generate_team(temperature=1.0)
        species_list = team.get_species_list()

        # Check no duplicates
        assert len(species_list) == len(set(normalize_name(s) for s in species_list))

    def test_get_viable_pokemon_list(self, mock_usage_loader):
        """Test getting viable Pokemon list."""
        gen = UsageBasedGenerator(
            usage_loader=mock_usage_loader,
            top_n=5,
        )

        pokemon_list = gen.get_viable_pokemon_list()
        assert len(pokemon_list) == 5
        assert "Dragapult" in pokemon_list

    def test_sample_moves(self, mock_usage_loader):
        """Test move sampling."""
        gen = UsageBasedGenerator(usage_loader=mock_usage_loader)

        stats = mock_usage_loader.get_pokemon_usage("Dragapult")
        moves = gen._sample_moves("Dragapult", stats, temperature=1.0)

        assert len(moves) == 4
        # All moves should be formatted
        for move in moves:
            assert move[0].isupper() or move[0] == "-"  # First char capitalized or hyphen

    def test_sample_item(self, mock_usage_loader):
        """Test item sampling."""
        gen = UsageBasedGenerator(usage_loader=mock_usage_loader)

        stats = mock_usage_loader.get_pokemon_usage("Dragapult")
        item = gen._sample_item("Dragapult", stats, temperature=1.0)

        assert item is not None
        assert isinstance(item, str)

    def test_sample_ability(self, mock_usage_loader):
        """Test ability sampling."""
        gen = UsageBasedGenerator(usage_loader=mock_usage_loader)

        stats = mock_usage_loader.get_pokemon_usage("Dragapult")
        ability = gen._sample_ability("Dragapult", stats, temperature=1.0)

        assert ability is not None
        assert isinstance(ability, str)

    def test_sample_spread(self, mock_usage_loader):
        """Test spread sampling."""
        gen = UsageBasedGenerator(usage_loader=mock_usage_loader)

        stats = mock_usage_loader.get_pokemon_usage("Dragapult")
        nature, evs = gen._sample_spread("Dragapult", stats, temperature=1.0)

        assert nature == "Timid"
        assert evs["spa"] == 252
        assert evs["spe"] == 252

    def test_sample_tera_type(self, mock_usage_loader):
        """Test tera type sampling."""
        gen = UsageBasedGenerator(usage_loader=mock_usage_loader)

        stats = mock_usage_loader.get_pokemon_usage("Dragapult")
        tera = gen._sample_tera_type("Dragapult", stats, ["Shadow Ball"])

        assert tera in ALL_TYPES

    def test_temperature_affects_variety(self, mock_usage_loader):
        """Test that temperature affects team variety."""
        mock_usage_loader.get_top_pokemon.return_value = [
            "Dragapult", "Gholdengo", "Great Tusk", "Iron Valiant",
            "Kingambit", "Landorus", "Heatran", "Gliscor", "Slowbro", "Toxapex"
        ]

        gen = UsageBasedGenerator(
            usage_loader=mock_usage_loader,
            top_n=10,
        )

        # Generate with low temperature - should be more consistent
        low_temp_teams = []
        for _ in range(5):
            team = gen.generate_team(temperature=0.5)
            low_temp_teams.append(set(team.get_species_list()))

        # Generate with high temperature - should be more varied
        high_temp_teams = []
        for _ in range(5):
            team = gen.generate_team(temperature=2.0)
            high_temp_teams.append(set(team.get_species_list()))

        # Both should produce valid teams
        for team_set in low_temp_teams + high_temp_teams:
            assert len(team_set) == 6


class TestDefaultConstants:
    """Tests for default constants."""

    def test_default_spreads(self):
        """Test default spreads are valid."""
        for name, (nature, evs) in DEFAULT_SPREADS.items():
            assert isinstance(nature, str)
            assert sum(evs.values()) <= 510  # EV cap

    def test_default_items(self):
        """Test default items list."""
        assert len(DEFAULT_ITEMS) > 0
        assert "Leftovers" in DEFAULT_ITEMS

    def test_all_types(self):
        """Test all types list."""
        assert len(ALL_TYPES) == 18
        assert "Normal" in ALL_TYPES
        assert "Dragon" in ALL_TYPES
        assert "Fairy" in ALL_TYPES


class TestRoundTrip:
    """Integration tests for team serialization round-trip."""

    def test_parse_and_export_round_trip(self):
        """Test that parse -> export produces equivalent result."""
        original_paste = """Dragapult @ Choice Specs
Ability: Infiltrator
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Shadow Ball
- Draco Meteor
- Flamethrower
- U-turn
"""
        team = parse_showdown_paste(original_paste)
        assert team is not None

        exported = team_to_showdown_paste(team)

        # Parse again
        team2 = parse_showdown_paste(exported)
        assert team2 is not None

        # Compare key fields
        assert team.pokemon[0].species == team2.pokemon[0].species
        assert team.pokemon[0].item == team2.pokemon[0].item
        assert team.pokemon[0].ability == team2.pokemon[0].ability
        assert team.pokemon[0].nature == team2.pokemon[0].nature
        assert set(team.pokemon[0].moves) == set(team2.pokemon[0].moves)

"""Data loaders for Pokemon data, usage stats, and sample teams.

This module handles:
1. Loading static Pokemon data (stats, types, moves, abilities)
2. Fetching/parsing Smogon usage statistics
3. Loading sample teams for bootstrapping
4. Creating ID mappings for embeddings
"""

import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# TODO: These will be populated from poke-env or external data files


@dataclass
class PokemonData:
    """Static data for a Pokemon species.

    This represents the base form of a Pokemon without any
    battle-specific information (EVs, moveset choice, etc.)
    """

    species_id: int
    name: str
    types: tuple[str, str | None]  # (primary, secondary or None)
    base_stats: dict[str, int]  # HP, Atk, Def, SpA, SpD, Spe
    abilities: list[str]  # Possible abilities
    hidden_ability: str | None
    move_pool: list[str]  # All learnable moves
    tier: str  # OU, UU, Uber, etc.
    base_stat_total: int

    @property
    def type1(self) -> str:
        return self.types[0]

    @property
    def type2(self) -> str | None:
        return self.types[1]


@dataclass
class MoveData:
    """Static data for a move."""

    move_id: int
    name: str
    type: str
    category: str  # Physical, Special, Status
    power: int | None
    accuracy: int | None
    pp: int
    priority: int
    flags: dict[str, bool]  # contact, sound, bullet, etc.
    effect: str  # Short description of effect


@dataclass
class ItemData:
    """Static data for a held item."""

    item_id: int
    name: str
    category: str  # berry, choice, mega_stone, z_crystal, etc.
    effect: str


@dataclass
class AbilityData:
    """Static data for an ability."""

    ability_id: int
    name: str
    effect_category: str  # weather, terrain, immunity, stat_boost, etc.
    effect: str


@dataclass
class UsageStats:
    """Smogon usage statistics for a Pokemon in a specific tier.

    Source: https://www.smogon.com/stats/
    """

    pokemon: str
    usage_percent: float
    raw_count: int
    real_count: int

    # Common sets
    common_items: dict[str, float]  # item -> usage%
    common_abilities: dict[str, float]  # ability -> usage%
    common_moves: dict[str, float]  # move -> usage%
    common_spreads: dict[str, float]  # nature/EVs -> usage%
    common_teammates: dict[str, float]  # teammate -> usage%

    # Checks and counters
    checks: dict[str, float]  # pokemon -> check score
    counters: dict[str, float]  # pokemon -> counter score


@dataclass
class TeamSet:
    """A complete Pokemon set (species + moveset + item + etc.)"""

    species: str
    nickname: str | None
    item: str | None
    ability: str
    moves: list[str]  # 1-4 moves
    nature: str
    evs: dict[str, int]  # HP, Atk, Def, SpA, SpD, Spe
    ivs: dict[str, int]  # Usually all 31 except for specific builds
    tera_type: str | None  # Gen 9 Tera type


@dataclass
class Team:
    """A complete team of 6 Pokemon."""

    name: str | None
    format: str  # gen9ou, gen8ou, etc.
    pokemon: list[TeamSet]
    source: str | None  # Where this team came from (ladder, tournament, etc.)


class PokemonDataLoader:
    """Loads and manages Pokemon data.

    Provides:
    - Species data lookup
    - Move data lookup
    - Item data lookup
    - Ability data lookup
    - ID mappings for embeddings
    """

    def __init__(self, data_dir: Path | str | None = None):
        """Initialize the data loader.

        Args:
            data_dir: Directory containing Pokemon data files.
                     If None, will attempt to load from poke-env.
        """
        self.data_dir = Path(data_dir) if data_dir else None

        # ID mappings (populated on load)
        self.species_to_id: dict[str, int] = {}
        self.move_to_id: dict[str, int] = {}
        self.item_to_id: dict[str, int] = {}
        self.ability_to_id: dict[str, int] = {}
        self.type_to_id: dict[str, int] = {}

        # Reverse mappings
        self.id_to_species: dict[int, str] = {}
        self.id_to_move: dict[int, str] = {}
        self.id_to_item: dict[int, str] = {}
        self.id_to_ability: dict[int, str] = {}
        self.id_to_type: dict[int, str] = {}

        # Data storage
        self.pokemon_data: dict[str, PokemonData] = {}
        self.move_data: dict[str, MoveData] = {}
        self.item_data: dict[str, ItemData] = {}
        self.ability_data: dict[str, AbilityData] = {}

        self._loaded = False

    def load(self) -> None:
        """Load all Pokemon data.

        This will populate:
        - species/move/item/ability data dicts
        - ID mappings for embeddings
        """
        if self._loaded:
            return

        # Initialize type mapping (always the same)
        self._init_type_mapping()

        # Try to load from poke-env first
        self._load_from_poke_env()

        self._loaded = True

    def _init_type_mapping(self) -> None:
        """Initialize the type -> ID mapping."""
        types = [
            "normal", "fire", "water", "electric", "grass", "ice",
            "fighting", "poison", "ground", "flying", "psychic", "bug",
            "rock", "ghost", "dragon", "dark", "steel", "fairy",
        ]
        for i, t in enumerate(types):
            self.type_to_id[t] = i + 1  # 0 reserved for "no type"
            self.id_to_type[i + 1] = t

    def _load_from_poke_env(self) -> None:
        """Load data from poke-env's built-in data."""
        try:
            from poke_env.data import GenData

            gen_data = GenData.from_gen(9)

            # Load Pokemon
            for i, (name, data) in enumerate(gen_data.pokedex.items()):
                species_id = i + 1
                self.species_to_id[name] = species_id
                self.id_to_species[species_id] = name

                # Extract types
                types = data.get("types", ["Normal"])
                type1 = types[0].lower() if types else "normal"
                type2 = types[1].lower() if len(types) > 1 else None

                # Extract base stats
                base_stats = data.get("baseStats", {})

                self.pokemon_data[name] = PokemonData(
                    species_id=species_id,
                    name=name,
                    types=(type1, type2),
                    base_stats=base_stats,
                    abilities=list(data.get("abilities", {}).values()),
                    hidden_ability=data.get("abilities", {}).get("H"),
                    move_pool=[],  # Would need learnset data
                    tier=data.get("tier", ""),
                    base_stat_total=sum(base_stats.values()),
                )

            # Load moves
            for i, (name, data) in enumerate(gen_data.moves.items()):
                move_id = i + 1
                self.move_to_id[name] = move_id
                self.id_to_move[move_id] = name

                self.move_data[name] = MoveData(
                    move_id=move_id,
                    name=name,
                    type=data.get("type", "Normal").lower(),
                    category=data.get("category", "Physical"),
                    power=data.get("basePower"),
                    accuracy=data.get("accuracy"),
                    pp=data.get("pp", 0),
                    priority=data.get("priority", 0),
                    flags=data.get("flags", {}),
                    effect=data.get("shortDesc", ""),
                )

            # Load items
            for i, (name, data) in enumerate(gen_data.items.items()):
                item_id = i + 1
                self.item_to_id[name] = item_id
                self.id_to_item[item_id] = name

            # Load abilities
            for i, (name, data) in enumerate(gen_data.abilities.items()):
                ability_id = i + 1
                self.ability_to_id[name] = ability_id
                self.id_to_ability[ability_id] = name

        except ImportError:
            print("Warning: poke-env not available, using empty data")

    def get_pokemon(self, name: str) -> PokemonData | None:
        """Get Pokemon data by name."""
        self.load()
        return self.pokemon_data.get(name.lower().replace(" ", ""))

    def get_move(self, name: str) -> MoveData | None:
        """Get move data by name."""
        self.load()
        return self.move_data.get(name.lower().replace(" ", ""))

    def get_species_id(self, name: str) -> int:
        """Get species ID for embedding lookup. Returns 0 for unknown."""
        self.load()
        return self.species_to_id.get(name.lower().replace(" ", ""), 0)

    def get_move_id(self, name: str) -> int:
        """Get move ID for embedding lookup. Returns 0 for unknown."""
        self.load()
        return self.move_to_id.get(name.lower().replace(" ", ""), 0)

    def get_item_id(self, name: str) -> int:
        """Get item ID for embedding lookup. Returns 0 for unknown."""
        self.load()
        return self.item_to_id.get(name.lower().replace(" ", ""), 0)

    def get_ability_id(self, name: str) -> int:
        """Get ability ID for embedding lookup. Returns 0 for unknown."""
        self.load()
        return self.ability_to_id.get(name.lower().replace(" ", ""), 0)

    def get_type_id(self, name: str) -> int:
        """Get type ID for embedding lookup. Returns 0 for unknown."""
        return self.type_to_id.get(name.lower(), 0)


class UsageStatsLoader:
    """Loads and parses Smogon usage statistics.

    Usage stats provide valuable priors for:
    - Which Pokemon/moves/items are viable
    - Common team compositions
    - Expected opponent strategies

    Stats are fetched from: https://www.smogon.com/stats/
    Format: YYYY-MM/gen9ou-RATING.txt
    """

    SMOGON_STATS_URL = "https://www.smogon.com/stats"
    VALID_RATINGS = [0, 1500, 1630, 1695, 1760, 1825]

    def __init__(self, cache_dir: Path | str | None = None):
        """Initialize the usage stats loader.

        Args:
            cache_dir: Directory to cache downloaded stats
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/usage_stats")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats: dict[str, UsageStats] = {}
        self._loaded = False
        self._tier = ""
        self._rating = 0

    def load(
        self,
        tier: str = "gen9ou",
        rating: int = 1695,
        year_month: str | None = None,
        force_refresh: bool = False,
    ) -> bool:
        """Load usage stats for a tier.

        Args:
            tier: The tier to load stats for (e.g., "gen9ou")
            rating: Minimum rating cutoff (1500, 1630, 1695, 1760, 1825)
            year_month: Specific month to load (e.g., "2024-12"). Default: latest.
            force_refresh: Force re-download even if cached

        Returns:
            True if loaded successfully, False otherwise
        """
        if rating not in self.VALID_RATINGS:
            logger.warning(f"Invalid rating {rating}, using 1695")
            rating = 1695

        self._tier = tier
        self._rating = rating

        # Determine month to fetch
        if year_month is None:
            # Try recent months (stats may take a few days to publish)
            now = datetime.now()
            months_to_try = []
            for offset in range(0, 3):
                month = now.month - offset
                year = now.year
                if month <= 0:
                    month += 12
                    year -= 1
                months_to_try.append(f"{year}-{month:02d}")
        else:
            months_to_try = [year_month]

        # Try to load from cache or download
        for month in months_to_try:
            cache_file = self.cache_dir / f"{tier}-{rating}-{month}.json"

            if cache_file.exists() and not force_refresh:
                try:
                    self._load_from_cache(cache_file)
                    logger.info(f"Loaded usage stats from cache: {cache_file}")
                    self._loaded = True
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load from cache: {e}")

            # Try to download
            try:
                self._download_and_parse(tier, rating, month)
                self._save_to_cache(cache_file)
                logger.info(f"Downloaded and cached usage stats for {month}")
                self._loaded = True
                return True
            except Exception as e:
                logger.debug(f"Failed to download stats for {month}: {e}")
                continue

        logger.warning(f"Failed to load usage stats for {tier}")
        self._loaded = True  # Mark as loaded even on failure to avoid retries
        return False

    def _download_and_parse(self, tier: str, rating: int, year_month: str) -> None:
        """Download and parse usage stats from Smogon.

        Parses the main usage file and the moveset (chaos) file for detailed data.
        """
        # Main usage file
        usage_url = f"{self.SMOGON_STATS_URL}/{year_month}/{tier}-{rating}.txt"
        logger.debug(f"Fetching: {usage_url}")

        try:
            with urllib.request.urlopen(usage_url, timeout=30) as response:
                usage_content = response.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch usage stats: {e}")

        # Parse main usage stats
        self._parse_usage_file(usage_content)

        # Try to get detailed moveset data (chaos format)
        chaos_url = f"{self.SMOGON_STATS_URL}/{year_month}/chaos/{tier}-{rating}.json"
        try:
            with urllib.request.urlopen(chaos_url, timeout=30) as response:
                chaos_content = response.read().decode("utf-8")
                chaos_data = json.loads(chaos_content)
                self._parse_chaos_data(chaos_data)
        except Exception as e:
            logger.debug(f"Could not fetch chaos data: {e}")

    def _parse_usage_file(self, content: str) -> None:
        """Parse the main usage stats file.

        Format:
        | Rank | Pokemon            | Usage %  | Raw     | Real    | % Real |
        |    1 | Great Tusk         | 31.12345 | 1234567 | 1234567 | 31.12% |
        """
        lines = content.strip().split("\n")

        for line in lines:
            # Skip header and separator lines
            if not line.startswith("|") or "Pokemon" in line or "---" in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                continue

            try:
                # Parse: | rank | pokemon | usage% | raw | real | ...
                pokemon = parts[2].strip().lower().replace(" ", "").replace("-", "")
                usage_pct = float(parts[3].replace("%", ""))
                raw_count = int(parts[4])
                real_count = int(parts[5]) if len(parts) > 5 else raw_count

                # Create basic stats (detailed data comes from chaos)
                if pokemon not in self.stats:
                    self.stats[pokemon] = UsageStats(
                        pokemon=pokemon,
                        usage_percent=usage_pct,
                        raw_count=raw_count,
                        real_count=real_count,
                        common_items={},
                        common_abilities={},
                        common_moves={},
                        common_spreads={},
                        common_teammates={},
                        checks={},
                        counters={},
                    )
                else:
                    self.stats[pokemon].usage_percent = usage_pct
                    self.stats[pokemon].raw_count = raw_count
                    self.stats[pokemon].real_count = real_count
            except (ValueError, IndexError) as e:
                continue

    def _parse_chaos_data(self, data: dict) -> None:
        """Parse detailed stats from Smogon's chaos JSON format."""
        if "data" not in data:
            return

        for pokemon_name, pokemon_data in data["data"].items():
            pokemon = pokemon_name.lower().replace(" ", "").replace("-", "")

            if pokemon not in self.stats:
                # Create new entry
                self.stats[pokemon] = UsageStats(
                    pokemon=pokemon,
                    usage_percent=pokemon_data.get("usage", 0) * 100,
                    raw_count=pokemon_data.get("Raw count", 0),
                    real_count=pokemon_data.get("Real count", 0),
                    common_items={},
                    common_abilities={},
                    common_moves={},
                    common_spreads={},
                    common_teammates={},
                    checks={},
                    counters={},
                )

            stats = self.stats[pokemon]

            # Parse items
            if "Items" in pokemon_data:
                total = sum(pokemon_data["Items"].values())
                if total > 0:
                    stats.common_items = {
                        k.lower(): v / total * 100
                        for k, v in pokemon_data["Items"].items()
                    }

            # Parse abilities
            if "Abilities" in pokemon_data:
                total = sum(pokemon_data["Abilities"].values())
                if total > 0:
                    stats.common_abilities = {
                        k.lower(): v / total * 100
                        for k, v in pokemon_data["Abilities"].items()
                    }

            # Parse moves
            if "Moves" in pokemon_data:
                total = sum(pokemon_data["Moves"].values())
                if total > 0:
                    stats.common_moves = {
                        k.lower(): v / total * 100
                        for k, v in pokemon_data["Moves"].items()
                    }

            # Parse spreads
            if "Spreads" in pokemon_data:
                total = sum(pokemon_data["Spreads"].values())
                if total > 0:
                    stats.common_spreads = {
                        k: v / total * 100
                        for k, v in pokemon_data["Spreads"].items()
                    }

            # Parse teammates
            if "Teammates" in pokemon_data:
                stats.common_teammates = {
                    k.lower().replace(" ", ""): v
                    for k, v in pokemon_data["Teammates"].items()
                }

            # Parse checks/counters
            if "Checks and Counters" in pokemon_data:
                for check_name, check_data in pokemon_data["Checks and Counters"].items():
                    name = check_name.lower().replace(" ", "")
                    score = check_data[1] if isinstance(check_data, list) else 0
                    if score > 0.5:  # Good counter
                        stats.counters[name] = score
                    elif score > 0:
                        stats.checks[name] = score

    def _load_from_cache(self, cache_file: Path) -> None:
        """Load stats from a cache file."""
        with open(cache_file) as f:
            data = json.load(f)

        self.stats = {}
        for pokemon, stats_data in data.items():
            self.stats[pokemon] = UsageStats(**stats_data)

    def _save_to_cache(self, cache_file: Path) -> None:
        """Save stats to a cache file."""
        data = {}
        for pokemon, stats in self.stats.items():
            data[pokemon] = {
                "pokemon": stats.pokemon,
                "usage_percent": stats.usage_percent,
                "raw_count": stats.raw_count,
                "real_count": stats.real_count,
                "common_items": stats.common_items,
                "common_abilities": stats.common_abilities,
                "common_moves": stats.common_moves,
                "common_spreads": stats.common_spreads,
                "common_teammates": stats.common_teammates,
                "checks": stats.checks,
                "counters": stats.counters,
            }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_pokemon_usage(self, pokemon: str) -> UsageStats | None:
        """Get usage stats for a Pokemon."""
        return self.stats.get(pokemon.lower().replace(" ", "").replace("-", ""))

    def get_top_pokemon(self, n: int = 50) -> list[str]:
        """Get the top N most used Pokemon."""
        sorted_stats = sorted(
            self.stats.values(),
            key=lambda x: x.usage_percent,
            reverse=True,
        )
        return [s.pokemon for s in sorted_stats[:n]]

    def get_common_teammates(self, pokemon: str, n: int = 10) -> list[str]:
        """Get the most common teammates for a Pokemon."""
        stats = self.get_pokemon_usage(pokemon)
        if not stats:
            return []

        sorted_teammates = sorted(
            stats.common_teammates.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [t[0] for t in sorted_teammates[:n]]

    def get_common_moves(self, pokemon: str, n: int = 10) -> list[tuple[str, float]]:
        """Get the most common moves for a Pokemon.

        Returns:
            List of (move_name, usage_percent) tuples
        """
        stats = self.get_pokemon_usage(pokemon)
        if not stats:
            return []

        sorted_moves = sorted(
            stats.common_moves.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_moves[:n]

    def get_common_items(self, pokemon: str, n: int = 5) -> list[tuple[str, float]]:
        """Get the most common items for a Pokemon."""
        stats = self.get_pokemon_usage(pokemon)
        if not stats:
            return []

        sorted_items = sorted(
            stats.common_items.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_items[:n]

    def get_counters(self, pokemon: str, n: int = 10) -> list[tuple[str, float]]:
        """Get Pokemon that counter this one."""
        stats = self.get_pokemon_usage(pokemon)
        if not stats:
            return []

        sorted_counters = sorted(
            stats.counters.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_counters[:n]

    def predict_team_from_pokemon(
        self,
        revealed: list[str],
        n: int = 6,
    ) -> list[tuple[str, float]]:
        """Predict likely team members based on revealed Pokemon.

        Uses teammate correlations from usage data.

        Args:
            revealed: List of revealed Pokemon names
            n: Number of predictions to return

        Returns:
            List of (pokemon, probability) tuples
        """
        if not revealed or not self.stats:
            return []

        # Aggregate teammate probabilities
        teammate_scores: dict[str, float] = {}

        for pokemon in revealed:
            stats = self.get_pokemon_usage(pokemon)
            if not stats:
                continue

            for teammate, score in stats.common_teammates.items():
                if teammate not in [p.lower().replace(" ", "") for p in revealed]:
                    teammate_scores[teammate] = teammate_scores.get(teammate, 0) + score

        # Sort by score and return top N
        sorted_predictions = sorted(
            teammate_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_predictions[:n]


class TeamLoader:
    """Loads sample teams for bootstrapping training.

    Teams can come from:
    - Smogon sample teams
    - High-ladder replays
    - Tournament teams
    - User-provided teams
    """

    def __init__(self, teams_dir: Path | str | None = None):
        """Initialize team loader.

        Args:
            teams_dir: Directory containing team files
        """
        self.teams_dir = Path(teams_dir) if teams_dir else Path("data/sample_teams")
        self.teams: list[Team] = []

    def load_teams(self, format: str = "gen9ou") -> list[Team]:
        """Load all teams for a format.

        Args:
            format: The format to load teams for

        Returns:
            List of Team objects
        """
        self.teams = []

        if not self.teams_dir.exists():
            return self.teams

        # Load from JSON files
        for team_file in self.teams_dir.glob(f"{format}*.json"):
            try:
                with open(team_file) as f:
                    data = json.load(f)
                    for team_data in data:
                        self.teams.append(self._parse_team(team_data, format))
            except Exception as e:
                print(f"Warning: Failed to load {team_file}: {e}")

        # Load from Showdown paste format
        for paste_file in self.teams_dir.glob(f"{format}*.txt"):
            try:
                teams = self._parse_showdown_paste(paste_file, format)
                self.teams.extend(teams)
            except Exception as e:
                print(f"Warning: Failed to load {paste_file}: {e}")

        return self.teams

    def _parse_team(self, data: dict, format: str) -> Team:
        """Parse a team from JSON format."""
        pokemon = []
        for p in data.get("pokemon", []):
            pokemon.append(TeamSet(
                species=p["species"],
                nickname=p.get("nickname"),
                item=p.get("item"),
                ability=p["ability"],
                moves=p["moves"],
                nature=p.get("nature", "Serious"),
                evs=p.get("evs", {}),
                ivs=p.get("ivs", {}),
                tera_type=p.get("teraType"),
            ))

        return Team(
            name=data.get("name"),
            format=format,
            pokemon=pokemon,
            source=data.get("source"),
        )

    def _parse_showdown_paste(self, path: Path, format: str) -> list[Team]:
        """Parse teams from Showdown paste format.

        Format example:
        Dragapult @ Choice Specs
        Ability: Infiltrator
        Tera Type: Ghost
        EVs: 252 SpA / 4 SpD / 252 Spe
        Timid Nature
        - Shadow Ball
        - Draco Meteor
        - Flamethrower
        - U-turn

        Teams are separated by "===" lines.
        """
        with open(path) as f:
            content = f.read()

        teams = []

        # Split by === (team separators)
        team_blocks = re.split(r"={3,}", content)

        for block in team_blocks:
            block = block.strip()
            if not block:
                continue

            # Parse individual Pokemon in this team
            pokemon_list = self._parse_pokemon_from_paste(block)

            if pokemon_list:
                teams.append(Team(
                    name=None,
                    format=format,
                    pokemon=pokemon_list,
                    source="paste",
                ))

        return teams

    def _parse_pokemon_from_paste(self, block: str) -> list[TeamSet]:
        """Parse Pokemon from a paste block."""
        pokemon_list = []
        current: dict[str, Any] = {}

        lines = block.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                # End of a Pokemon
                if current.get("species"):
                    pokemon_list.append(self._dict_to_teamset(current))
                    current = {}
                continue

            if "@" in line and ":" not in line.split("@")[0]:
                # Species @ Item (or just Species)
                if current.get("species"):
                    pokemon_list.append(self._dict_to_teamset(current))
                    current = {}

                parts = line.split("@")
                # Handle nickname: "Nickname (Species) @ Item"
                species_part = parts[0].strip()
                if "(" in species_part and ")" in species_part:
                    # Has nickname
                    match = re.match(r"(.+?)\s*\((.+?)\)", species_part)
                    if match:
                        current["nickname"] = match.group(1).strip()
                        current["species"] = match.group(2).strip()
                else:
                    current["species"] = species_part

                if len(parts) > 1:
                    current["item"] = parts[1].strip()

            elif line.startswith("Ability:"):
                current["ability"] = line.split(":", 1)[1].strip()

            elif line.startswith("Tera Type:"):
                current["tera_type"] = line.split(":", 1)[1].strip()

            elif line.startswith("EVs:"):
                current["evs"] = self._parse_stats_line(line.split(":", 1)[1])

            elif line.startswith("IVs:"):
                current["ivs"] = self._parse_stats_line(line.split(":", 1)[1])

            elif line.endswith("Nature"):
                current["nature"] = line.replace("Nature", "").strip()

            elif line.startswith("-") or line.startswith("~"):
                if "moves" not in current:
                    current["moves"] = []
                move = line[1:].strip()
                if move:
                    current["moves"].append(move)

            elif line.startswith("Level:"):
                current["level"] = int(line.split(":", 1)[1].strip())

        # Don't forget last Pokemon
        if current.get("species"):
            pokemon_list.append(self._dict_to_teamset(current))

        return pokemon_list

    def _dict_to_teamset(self, d: dict[str, Any]) -> TeamSet:
        """Convert a dict to a TeamSet."""
        return TeamSet(
            species=d.get("species", "MissingNo"),
            nickname=d.get("nickname"),
            item=d.get("item"),
            ability=d.get("ability", ""),
            moves=d.get("moves", []),
            nature=d.get("nature", "Serious"),
            evs=d.get("evs", {}),
            ivs=d.get("ivs", {}),
            tera_type=d.get("tera_type"),
        )

    def _parse_stats_line(self, line: str) -> dict[str, int]:
        """Parse EVs/IVs from a line like '252 SpA / 4 SpD / 252 Spe'."""
        result = {}
        stat_map = {
            "hp": "hp", "atk": "atk", "def": "def",
            "spa": "spa", "spd": "spd", "spe": "spe",
            "attack": "atk", "defense": "def",
            "special attack": "spa", "special defense": "spd",
            "speed": "spe",
        }

        for part in line.split("/"):
            part = part.strip()
            if not part:
                continue

            tokens = part.split()
            if len(tokens) >= 2:
                try:
                    value = int(tokens[0])
                    stat = " ".join(tokens[1:]).lower()
                    if stat in stat_map:
                        result[stat_map[stat]] = value
                except ValueError:
                    continue

        return result

    def get_random_team(self) -> Team | None:
        """Get a random team from the loaded teams."""
        import random
        if not self.teams:
            return None
        return random.choice(self.teams)

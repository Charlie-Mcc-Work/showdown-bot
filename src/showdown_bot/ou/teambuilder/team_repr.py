"""Team representation for the teambuilder.

This module handles:
1. Converting teams to/from tensor representations
2. Team validity checking
3. Team serialization (to Showdown paste format)
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from showdown_bot.ou.shared.data_loader import Team, TeamSet, PokemonDataLoader


@dataclass
class PokemonSlot:
    """Represents a single Pokemon slot in a team being built.

    During autoregressive generation, slots are filled one at a time.
    Empty slots have all fields as None.
    """

    species: str | None = None
    moves: list[str] | None = None  # Up to 4 moves
    item: str | None = None
    ability: str | None = None
    nature: str | None = None
    evs: dict[str, int] | None = None  # HP, Atk, Def, SpA, SpD, Spe
    tera_type: str | None = None

    def is_complete(self) -> bool:
        """Check if this slot is fully specified."""
        return all([
            self.species is not None,
            self.moves is not None and len(self.moves) >= 1,
            self.ability is not None,
            self.nature is not None,
            self.evs is not None,
        ])

    def is_empty(self) -> bool:
        """Check if this slot is empty."""
        return self.species is None

    def to_team_set(self) -> TeamSet | None:
        """Convert to a TeamSet if complete."""
        if not self.is_complete():
            return None

        return TeamSet(
            species=self.species,
            nickname=None,
            item=self.item,
            ability=self.ability,
            moves=self.moves,
            nature=self.nature,
            evs=self.evs,
            ivs={"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
            tera_type=self.tera_type,
        )


@dataclass
class PartialTeam:
    """A team in the process of being built.

    Tracks which slots are filled and provides context for generation.
    """

    slots: list[PokemonSlot]
    format: str = "gen9ou"

    def __init__(self, format: str = "gen9ou"):
        self.slots = [PokemonSlot() for _ in range(6)]
        self.format = format

    def num_filled(self) -> int:
        """Number of filled slots."""
        return sum(1 for s in self.slots if not s.is_empty())

    def num_complete(self) -> int:
        """Number of complete slots."""
        return sum(1 for s in self.slots if s.is_complete())

    def is_complete(self) -> bool:
        """Check if team is fully built."""
        return self.num_complete() == 6

    def next_empty_slot(self) -> int | None:
        """Get index of next empty slot, or None if full."""
        for i, slot in enumerate(self.slots):
            if slot.is_empty():
                return i
        return None

    def to_team(self) -> Team | None:
        """Convert to a complete Team if all slots are filled."""
        if not self.is_complete():
            return None

        pokemon = [s.to_team_set() for s in self.slots]
        if any(p is None for p in pokemon):
            return None

        return Team(
            name=None,
            format=self.format,
            pokemon=pokemon,
            source="generated",
        )

    def get_species_list(self) -> list[str]:
        """Get list of species already on the team."""
        return [s.species for s in self.slots if s.species is not None]

    def get_type_coverage(self) -> set[str]:
        """Get types that the team can hit super-effectively.

        TODO: Implement based on move types and type chart.
        """
        return set()

    def get_type_weaknesses(self) -> dict[str, int]:
        """Get types the team is weak to, with weakness count.

        TODO: Implement based on Pokemon types.
        """
        return {}


class TeamRepresentation(nn.Module):
    """Neural network representation of a team.

    Encodes a partial or complete team into tensors for the generator/evaluator.
    """

    def __init__(
        self,
        data_loader: PokemonDataLoader,
        pokemon_dim: int = 256,
        team_dim: int = 512,
    ):
        super().__init__()
        self.data_loader = data_loader
        self.pokemon_dim = pokemon_dim
        self.team_dim = team_dim

        # Placeholder for encoder - will use shared encoders
        self.pokemon_encoder: nn.Module | None = None
        self.team_encoder: nn.Module | None = None

    def set_encoders(
        self,
        pokemon_encoder: nn.Module,
        team_encoder: nn.Module,
    ) -> None:
        """Set the shared encoders."""
        self.pokemon_encoder = pokemon_encoder
        self.team_encoder = team_encoder

    def encode_partial_team(
        self,
        team: PartialTeam,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a partial team for the generator.

        Args:
            team: The partial team to encode
            device: Device for tensors

        Returns:
            Dict with:
            - filled_mask: (6,) bool tensor, True for filled slots
            - pokemon_encodings: (6, pokemon_dim) tensor, zeros for empty
            - team_encoding: (team_dim,) tensor
        """
        device = device or torch.device("cpu")

        # Create mask for filled slots
        filled_mask = torch.tensor(
            [not s.is_empty() for s in team.slots],
            dtype=torch.bool,
            device=device,
        )

        # Encode each filled slot
        # TODO: Implement actual encoding once encoders are set
        pokemon_encodings = torch.zeros(6, self.pokemon_dim, device=device)

        # Encode team-level features
        team_encoding = torch.zeros(self.team_dim, device=device)

        return {
            "filled_mask": filled_mask,
            "pokemon_encodings": pokemon_encodings,
            "team_encoding": team_encoding,
            "num_filled": team.num_filled(),
            "species_list": team.get_species_list(),
        }


def team_to_showdown_paste(team: Team) -> str:
    """Convert a team to Showdown paste format.

    Example output:
    Dragapult @ Choice Specs
    Ability: Infiltrator
    Tera Type: Ghost
    EVs: 252 SpA / 4 SpD / 252 Spe
    Timid Nature
    - Shadow Ball
    - Draco Meteor
    - Flamethrower
    - U-turn

    (blank line between Pokemon)
    """
    lines = []

    for pokemon in team.pokemon:
        # Name and item
        if pokemon.item:
            lines.append(f"{pokemon.species} @ {pokemon.item}")
        else:
            lines.append(pokemon.species)

        # Ability
        lines.append(f"Ability: {pokemon.ability}")

        # Tera type (Gen 9)
        if pokemon.tera_type:
            lines.append(f"Tera Type: {pokemon.tera_type}")

        # EVs
        if pokemon.evs:
            ev_parts = []
            stat_names = {"hp": "HP", "atk": "Atk", "def": "Def",
                         "spa": "SpA", "spd": "SpD", "spe": "Spe"}
            for stat, value in pokemon.evs.items():
                if value > 0:
                    ev_parts.append(f"{value} {stat_names.get(stat, stat)}")
            if ev_parts:
                lines.append(f"EVs: {' / '.join(ev_parts)}")

        # Nature
        lines.append(f"{pokemon.nature} Nature")

        # IVs (only if non-standard)
        if pokemon.ivs:
            non_max = {k: v for k, v in pokemon.ivs.items() if v != 31}
            if non_max:
                stat_names = {"hp": "HP", "atk": "Atk", "def": "Def",
                             "spa": "SpA", "spd": "SpD", "spe": "Spe"}
                iv_parts = [f"{v} {stat_names.get(k, k)}" for k, v in non_max.items()]
                lines.append(f"IVs: {' / '.join(iv_parts)}")

        # Moves
        for move in pokemon.moves:
            lines.append(f"- {move}")

        # Blank line between Pokemon
        lines.append("")

    return "\n".join(lines)


def parse_showdown_paste(paste: str, format: str = "gen9ou") -> Team | None:
    """Parse a team from Showdown paste format.

    Args:
        paste: The paste text
        format: The format for the team

    Returns:
        Parsed Team or None if parsing fails
    """
    pokemon_list: list[TeamSet] = []
    current_pokemon: dict[str, Any] = {}

    lines = paste.strip().split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            # End of a Pokemon
            if current_pokemon:
                pokemon_list.append(_dict_to_teamset(current_pokemon))
                current_pokemon = {}
            continue

        if "@" in line:
            # Name @ Item
            parts = line.split("@")
            current_pokemon["species"] = parts[0].strip()
            if len(parts) > 1:
                current_pokemon["item"] = parts[1].strip()
        elif line.startswith("Ability:"):
            current_pokemon["ability"] = line[8:].strip()
        elif line.startswith("Tera Type:"):
            current_pokemon["tera_type"] = line[10:].strip()
        elif line.startswith("EVs:"):
            current_pokemon["evs"] = _parse_evs(line[4:])
        elif line.startswith("IVs:"):
            current_pokemon["ivs"] = _parse_evs(line[4:])  # Same format
        elif line.endswith("Nature"):
            current_pokemon["nature"] = line[:-7].strip()
        elif line.startswith("-"):
            if "moves" not in current_pokemon:
                current_pokemon["moves"] = []
            current_pokemon["moves"].append(line[1:].strip())
        elif not any(line.startswith(x) for x in ["Level:", "Shiny:", "Happiness:"]):
            # Might be just a species name (no item)
            if "species" not in current_pokemon:
                current_pokemon["species"] = line

    # Don't forget the last Pokemon
    if current_pokemon:
        pokemon_list.append(_dict_to_teamset(current_pokemon))

    if len(pokemon_list) == 0:
        return None

    return Team(
        name=None,
        format=format,
        pokemon=pokemon_list,
        source="paste",
    )


def _dict_to_teamset(d: dict[str, Any]) -> TeamSet:
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


def _parse_evs(ev_str: str) -> dict[str, int]:
    """Parse EVs/IVs from string like '252 SpA / 4 SpD / 252 Spe'."""
    result = {}
    stat_map = {
        "hp": "hp", "atk": "atk", "def": "def",
        "spa": "spa", "spd": "spd", "spe": "spe",
        "attack": "atk", "defense": "def",
        "special attack": "spa", "special defense": "spd",
        "speed": "spe",
    }

    for part in ev_str.split("/"):
        part = part.strip()
        if not part:
            continue

        # Split into value and stat name
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

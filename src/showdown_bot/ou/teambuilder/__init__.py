"""Teambuilder module for Gen 9 OU.

The teambuilder constructs competitive teams using:
1. Autoregressive generation (one Pokemon at a time)
2. Learned value function for team quality
3. Usage stats priors for move/item selection
4. Player feedback for optimization

Key components:
- TeamGenerator: Generates complete teams autoregressively (neural + usage-based)
- UsageBasedGenerator: Generates teams purely from Smogon usage statistics
- TeamEvaluator: Predicts expected win rate for a team
- TeamRepresentation: Encodes teams for the neural network
- PartialTeam/PokemonSlot: Team building data structures

See ../README.md for architecture details.
"""

from showdown_bot.ou.teambuilder.generator import (
    TeamGenerator,
    UsageBasedGenerator,
    normalize_name,
    format_name,
    sample_weighted,
    parse_spread,
    DEFAULT_SPREADS,
    DEFAULT_ITEMS,
    ALL_TYPES,
    NAME_FORMAT_MAP,
)
from showdown_bot.ou.teambuilder.evaluator import TeamEvaluator, TeamTensorEncoder
from showdown_bot.ou.teambuilder.team_repr import (
    TeamRepresentation,
    PartialTeam,
    PokemonSlot,
    team_to_showdown_paste,
    parse_showdown_paste,
)

__all__ = [
    "TeamGenerator",
    "UsageBasedGenerator",
    "TeamEvaluator",
    "TeamTensorEncoder",
    "TeamRepresentation",
    "PartialTeam",
    "PokemonSlot",
    "team_to_showdown_paste",
    "parse_showdown_paste",
    "normalize_name",
    "format_name",
    "sample_weighted",
    "parse_spread",
    "DEFAULT_SPREADS",
    "DEFAULT_ITEMS",
    "ALL_TYPES",
    "NAME_FORMAT_MAP",
]

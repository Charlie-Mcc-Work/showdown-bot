"""Gen 9 OU module for Pokemon Showdown.

This module provides:
- Teambuilder: Constructs competitive teams using learned policies
- Player: Plays OU battles with team preview and full team knowledge
- Shared: Common embeddings and encoders used by both systems

See README.md for architecture overview and implementation details.
"""

# Version of the OU module
__version__ = "0.1.0"

# Shared components
from showdown_bot.ou.shared.embeddings import (
    SharedEmbeddings,
    EmbeddingConfig,
    PokemonEmbedding,
    MoveEmbedding,
    ItemEmbedding,
    AbilityEmbedding,
)
from showdown_bot.ou.shared.encoders import (
    PokemonEncoder,
    TeamEncoder,
    SynergyEncoder,
    FullTeamEncoder,
)
from showdown_bot.ou.shared.data_loader import (
    PokemonDataLoader,
    UsageStatsLoader,
    TeamLoader,
    Team,
    TeamSet,
)

# Teambuilder components
from showdown_bot.ou.teambuilder.team_repr import (
    PokemonSlot,
    PartialTeam,
    TeamRepresentation,
    team_to_showdown_paste,
    parse_showdown_paste,
)
from showdown_bot.ou.teambuilder.generator import (
    TeamGenerator,
    SpeciesSelector,
    MovesetSelector,
)
from showdown_bot.ou.teambuilder.evaluator import (
    TeamEvaluator,
    MatchupPredictor,
    CoverageAnalyzer,
    TeamEvaluatorTrainer,
)

# Player components
from showdown_bot.ou.player.state_encoder import OUStateEncoder, OUEncodedState
from showdown_bot.ou.player.network import OUPlayerNetwork, OUActorCritic
from showdown_bot.ou.player.team_preview import (
    TeamPreviewSelector,
    HeuristicLeadSelector,
    TeamPreviewAnalyzer,
)

__all__ = [
    # Shared
    "SharedEmbeddings",
    "EmbeddingConfig",
    "PokemonEmbedding",
    "MoveEmbedding",
    "ItemEmbedding",
    "AbilityEmbedding",
    "PokemonEncoder",
    "TeamEncoder",
    "SynergyEncoder",
    "FullTeamEncoder",
    "PokemonDataLoader",
    "UsageStatsLoader",
    "TeamLoader",
    "Team",
    "TeamSet",
    # Teambuilder
    "PokemonSlot",
    "PartialTeam",
    "TeamRepresentation",
    "team_to_showdown_paste",
    "parse_showdown_paste",
    "TeamGenerator",
    "SpeciesSelector",
    "MovesetSelector",
    "TeamEvaluator",
    "MatchupPredictor",
    "CoverageAnalyzer",
    "TeamEvaluatorTrainer",
    # Player
    "OUStateEncoder",
    "OUEncodedState",
    "OUPlayerNetwork",
    "OUActorCritic",
    "TeamPreviewSelector",
    "HeuristicLeadSelector",
    "TeamPreviewAnalyzer",
]

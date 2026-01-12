"""Shared components for OU player and teambuilder.

This module contains:
- Embeddings: Learned dense representations for Pokemon, moves, items, abilities
- Encoders: Neural network components for encoding battle/team state
- Data loaders: Loading Pokemon data, usage stats, and sample teams

These components are shared between the player and teambuilder to ensure
consistent representations and enable transfer of learned features.
"""

from showdown_bot.ou.shared.embeddings import (
    PokemonEmbedding,
    MoveEmbedding,
    ItemEmbedding,
    AbilityEmbedding,
    EmbeddingConfig,
)
from showdown_bot.ou.shared.encoders import (
    PokemonEncoder,
    TeamEncoder,
    SynergyEncoder,
)
from showdown_bot.ou.shared.data_loader import (
    PokemonDataLoader,
    UsageStatsLoader,
)

__all__ = [
    # Embeddings
    "PokemonEmbedding",
    "MoveEmbedding",
    "ItemEmbedding",
    "AbilityEmbedding",
    "EmbeddingConfig",
    # Encoders
    "PokemonEncoder",
    "TeamEncoder",
    "SynergyEncoder",
    # Data
    "PokemonDataLoader",
    "UsageStatsLoader",
]

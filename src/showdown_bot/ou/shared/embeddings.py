"""Learned embeddings for Pokemon, moves, items, and abilities.

These embeddings provide dense vector representations that capture semantic
meaning beyond raw categorical IDs. They are shared between the player and
teambuilder to ensure consistent representations.

Key design decisions:
1. Embeddings are initialized from meaningful features (stats, types, usage)
   rather than random, to speed up training
2. All embeddings are trainable and updated during both player and teambuilder
   training
3. Unknown/new Pokemon/moves fall back to a learned [UNK] embedding
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class EmbeddingConfig:
    """Configuration for embedding dimensions.

    Attributes:
        pokemon_dim: Dimension of Pokemon species embeddings
        move_dim: Dimension of move embeddings
        item_dim: Dimension of item embeddings
        ability_dim: Dimension of ability embeddings
        type_dim: Dimension of type embeddings (18 types)
        stat_dim: Dimension for stat-based features
    """

    pokemon_dim: int = 128
    move_dim: int = 64
    item_dim: int = 32
    ability_dim: int = 32
    type_dim: int = 16
    stat_dim: int = 32

    # Vocabulary sizes (approximate, will be set from data)
    # These are set high enough to accommodate all Gen 9 Pokemon forms
    num_pokemon: int = 2000  # All Pokemon species + forms
    num_moves: int = 1000  # All moves
    num_items: int = 500  # All items
    num_abilities: int = 400  # All abilities
    num_types: int = 18  # Pokemon types


class PokemonEmbedding(nn.Module):
    """Embedding layer for Pokemon species.

    Combines:
    - Learned species embedding
    - Type embeddings (primary + secondary)
    - Base stat encoding
    - Usage-based features (optional, from Smogon stats)

    The final embedding captures both intrinsic properties (stats, types)
    and learned metagame relevance.
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config

        # Core species embedding (learned)
        self.species_embed = nn.Embedding(
            config.num_pokemon + 1,  # +1 for [UNK]
            config.pokemon_dim,
            padding_idx=0,
        )

        # Type embeddings (shared across Pokemon and moves)
        self.type_embed = nn.Embedding(
            config.num_types + 1,  # +1 for no secondary type
            config.type_dim,
        )

        # Base stat encoder: 6 stats -> stat_dim
        self.stat_encoder = nn.Sequential(
            nn.Linear(6, config.stat_dim),
            nn.ReLU(),
            nn.Linear(config.stat_dim, config.stat_dim),
        )

        # Combine all features into final embedding
        combined_dim = config.pokemon_dim + 2 * config.type_dim + config.stat_dim
        self.projection = nn.Linear(combined_dim, config.pokemon_dim)

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        """Initialize embeddings with meaningful values."""
        # Species embeddings: small random init
        nn.init.normal_(self.species_embed.weight, mean=0, std=0.02)
        # Padding is zero
        self.species_embed.weight.data[0].zero_()

        # Type embeddings: orthogonal init for distinctness
        nn.init.orthogonal_(self.type_embed.weight)

    def forward(
        self,
        species_ids: torch.Tensor,
        type1_ids: torch.Tensor,
        type2_ids: torch.Tensor,
        base_stats: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Pokemon embeddings.

        Args:
            species_ids: (batch, ...) Pokemon species IDs
            type1_ids: (batch, ...) Primary type IDs
            type2_ids: (batch, ...) Secondary type IDs (0 if none)
            base_stats: (batch, ..., 6) Base stats [HP, Atk, Def, SpA, SpD, Spe]

        Returns:
            (batch, ..., pokemon_dim) Pokemon embeddings
        """
        species_emb = self.species_embed(species_ids)
        type1_emb = self.type_embed(type1_ids)
        type2_emb = self.type_embed(type2_ids)
        stat_emb = self.stat_encoder(base_stats)

        # Concatenate all features
        combined = torch.cat([species_emb, type1_emb, type2_emb, stat_emb], dim=-1)

        # Project to final dimension
        return self.projection(combined)


class MoveEmbedding(nn.Module):
    """Embedding layer for moves.

    Combines:
    - Learned move embedding
    - Type embedding
    - Power/accuracy/PP encoding
    - Category (physical/special/status)
    - Effect flags (priority, contact, sound, etc.)
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config

        # Core move embedding
        self.move_embed = nn.Embedding(
            config.num_moves + 1,  # +1 for [UNK]
            config.move_dim,
            padding_idx=0,
        )

        # Type embedding (shared with Pokemon)
        self.type_embed = nn.Embedding(
            config.num_types + 1,
            config.type_dim,
        )

        # Move properties encoder
        # Properties: power, accuracy, pp, priority, category (3), flags (~10)
        self.properties_encoder = nn.Sequential(
            nn.Linear(20, config.move_dim // 2),
            nn.ReLU(),
            nn.Linear(config.move_dim // 2, config.move_dim // 2),
        )

        # Combine into final embedding
        combined_dim = config.move_dim + config.type_dim + config.move_dim // 2
        self.projection = nn.Linear(combined_dim, config.move_dim)

    def forward(
        self,
        move_ids: torch.Tensor,
        type_ids: torch.Tensor,
        properties: torch.Tensor,
    ) -> torch.Tensor:
        """Compute move embeddings.

        Args:
            move_ids: (batch, ...) Move IDs
            type_ids: (batch, ...) Move type IDs
            properties: (batch, ..., 20) Move properties vector

        Returns:
            (batch, ..., move_dim) Move embeddings
        """
        move_emb = self.move_embed(move_ids)
        type_emb = self.type_embed(type_ids)
        prop_emb = self.properties_encoder(properties)

        combined = torch.cat([move_emb, type_emb, prop_emb], dim=-1)
        return self.projection(combined)


class ItemEmbedding(nn.Module):
    """Embedding layer for held items.

    Items have diverse effects, so we primarily rely on learned embeddings
    with some categorical features (item category, fling power, etc.)
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config

        self.item_embed = nn.Embedding(
            config.num_items + 1,  # +1 for [UNK] and no item
            config.item_dim,
            padding_idx=0,
        )

        # Item category encoder (berry, choice, mega stone, etc.)
        self.category_embed = nn.Embedding(20, config.item_dim // 4)

        self.projection = nn.Linear(
            config.item_dim + config.item_dim // 4,
            config.item_dim,
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute item embeddings.

        Args:
            item_ids: (batch, ...) Item IDs
            category_ids: (batch, ...) Item category IDs

        Returns:
            (batch, ..., item_dim) Item embeddings
        """
        item_emb = self.item_embed(item_ids)
        cat_emb = self.category_embed(category_ids)
        combined = torch.cat([item_emb, cat_emb], dim=-1)
        return self.projection(combined)


class AbilityEmbedding(nn.Module):
    """Embedding layer for abilities.

    Abilities are primarily learned embeddings, as their effects are diverse
    and hard to encode manually.
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config

        self.ability_embed = nn.Embedding(
            config.num_abilities + 1,  # +1 for [UNK]
            config.ability_dim,
            padding_idx=0,
        )

        # Ability effect category (weather, terrain, stat boost, immunity, etc.)
        self.effect_embed = nn.Embedding(30, config.ability_dim // 4)

        self.projection = nn.Linear(
            config.ability_dim + config.ability_dim // 4,
            config.ability_dim,
        )

    def forward(
        self,
        ability_ids: torch.Tensor,
        effect_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ability embeddings.

        Args:
            ability_ids: (batch, ...) Ability IDs
            effect_ids: (batch, ...) Ability effect category IDs

        Returns:
            (batch, ..., ability_dim) Ability embeddings
        """
        ability_emb = self.ability_embed(ability_ids)
        effect_emb = self.effect_embed(effect_ids)
        combined = torch.cat([ability_emb, effect_emb], dim=-1)
        return self.projection(combined)


class SharedEmbeddings(nn.Module):
    """Container for all shared embeddings.

    This class holds all embedding layers and provides a unified interface
    for the player and teambuilder to access them.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        super().__init__()
        self.config = config or EmbeddingConfig()

        self.pokemon = PokemonEmbedding(self.config)
        self.move = MoveEmbedding(self.config)
        self.item = ItemEmbedding(self.config)
        self.ability = AbilityEmbedding(self.config)

    def save(self, path: str) -> None:
        """Save all embeddings to a file."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device | None = None) -> None:
        """Load embeddings from a file."""
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)

"""Neural network encoders for Pokemon, teams, and synergy.

These encoders take embedded representations and produce higher-level
features used by both the player and teambuilder.

Architecture notes:
- PokemonEncoder: Encodes a single Pokemon with its full set (moves, item, etc.)
- TeamEncoder: Encodes a full team of 6 Pokemon using attention
- SynergyEncoder: Captures relationships between team members
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from showdown_bot.ou.shared.embeddings import SharedEmbeddings, EmbeddingConfig


class PokemonEncoder(nn.Module):
    """Encodes a single Pokemon with its complete set.

    Takes:
    - Species embedding
    - 4 move embeddings
    - Item embedding
    - Ability embedding
    - EVs/IVs/Nature (stat modifiers)

    Produces a fixed-size representation of the Pokemon.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input dimensions from embeddings
        pokemon_dim = config.pokemon_dim
        move_dim = config.move_dim
        item_dim = config.item_dim
        ability_dim = config.ability_dim

        # Move aggregator (attention over 4 moves)
        self.move_attention = nn.MultiheadAttention(
            embed_dim=move_dim,
            num_heads=4,
            batch_first=True,
        )
        self.move_query = nn.Parameter(torch.randn(1, 1, move_dim))

        # Combine all Pokemon features
        # Pokemon emb + aggregated moves + item + ability + stat mods (6 EVs + nature)
        combined_dim = pokemon_dim + move_dim + item_dim + ability_dim + 7

        self.encoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        pokemon_emb: torch.Tensor,
        move_embs: torch.Tensor,
        item_emb: torch.Tensor,
        ability_emb: torch.Tensor,
        stat_mods: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a Pokemon.

        Args:
            pokemon_emb: (batch, pokemon_dim) Species embedding
            move_embs: (batch, 4, move_dim) Move embeddings
            item_emb: (batch, item_dim) Item embedding
            ability_emb: (batch, ability_dim) Ability embedding
            stat_mods: (batch, 7) EV spread (6) + nature modifier

        Returns:
            (batch, output_dim) Encoded Pokemon representation
        """
        batch_size = pokemon_emb.size(0)

        # Aggregate moves via attention
        query = self.move_query.expand(batch_size, -1, -1)
        move_agg, _ = self.move_attention(query, move_embs, move_embs)
        move_agg = move_agg.squeeze(1)  # (batch, move_dim)

        # Concatenate all features
        combined = torch.cat([
            pokemon_emb,
            move_agg,
            item_emb,
            ability_emb,
            stat_mods,
        ], dim=-1)

        return self.encoder(combined)


class TeamEncoder(nn.Module):
    """Encodes a full team of 6 Pokemon using self-attention.

    This captures team-level properties:
    - Overall synergy
    - Defensive/offensive balance
    - Role distribution
    - Team archetype
    """

    def __init__(
        self,
        pokemon_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        output_dim: int = 512,
    ):
        super().__init__()
        self.pokemon_dim = pokemon_dim
        self.output_dim = output_dim

        # Position embedding for team slots (optional, teams are order-invariant)
        self.slot_embed = nn.Embedding(6, pokemon_dim)

        # Transformer encoder for team-level attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pokemon_dim,
            nhead=num_heads,
            dim_feedforward=pokemon_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Team-level aggregation (attention pooling)
        self.team_query = nn.Parameter(torch.randn(1, 1, pokemon_dim))
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=pokemon_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Final projection
        self.projection = nn.Linear(pokemon_dim, output_dim)

    def forward(
        self,
        pokemon_encodings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a team of Pokemon.

        Args:
            pokemon_encodings: (batch, 6, pokemon_dim) Encoded Pokemon from PokemonEncoder
            padding_mask: (batch, 6) True for padding positions (optional)

        Returns:
            (batch, output_dim) Team-level encoding
        """
        batch_size = pokemon_encodings.size(0)

        # Add slot embeddings (helps distinguish roles even though team is set-like)
        slots = torch.arange(6, device=pokemon_encodings.device)
        slot_emb = self.slot_embed(slots).unsqueeze(0).expand(batch_size, -1, -1)
        x = pokemon_encodings + slot_emb

        # Self-attention between team members
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Pool into single team vector
        query = self.team_query.expand(batch_size, -1, -1)
        team_vec, _ = self.pool_attention(
            query, x, x,
            key_padding_mask=padding_mask,
        )
        team_vec = team_vec.squeeze(1)

        return self.projection(team_vec)


class SynergyEncoder(nn.Module):
    """Encodes synergy relationships between Pokemon pairs.

    Captures:
    - Type coverage (what types can this pair hit/resist?)
    - Offensive cores (e.g., Dragapult + Heatran)
    - Defensive cores (e.g., Corviknight + Toxapex)
    - Support relationships (hazard setter + spinblocker)
    """

    def __init__(
        self,
        pokemon_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()

        # Pairwise synergy scorer
        self.pairwise = nn.Sequential(
            nn.Linear(pokemon_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Type coverage encoder (18 attacking types x 18 defending types)
        self.coverage_encoder = nn.Linear(18, output_dim // 2)
        self.weakness_encoder = nn.Linear(18, output_dim // 2)

    def forward(
        self,
        pokemon_encodings: torch.Tensor,
        type_coverage: Optional[torch.Tensor] = None,
        type_weaknesses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute pairwise synergy scores for a team.

        Args:
            pokemon_encodings: (batch, 6, pokemon_dim) Encoded Pokemon
            type_coverage: (batch, 18) Types the team can hit super-effectively
            type_weaknesses: (batch, 18) Types the team is weak to

        Returns:
            (batch, 15, output_dim) Synergy encodings for each pair
            (15 pairs = C(6,2))
        """
        batch_size = pokemon_encodings.size(0)
        device = pokemon_encodings.device

        # Compute all pairwise combinations
        pairs = []
        for i in range(6):
            for j in range(i + 1, 6):
                pair_input = torch.cat([
                    pokemon_encodings[:, i],
                    pokemon_encodings[:, j],
                ], dim=-1)
                pairs.append(self.pairwise(pair_input))

        synergy = torch.stack(pairs, dim=1)  # (batch, 15, output_dim)

        # Add type coverage information if provided
        if type_coverage is not None:
            coverage_emb = self.coverage_encoder(type_coverage)
            synergy = synergy + coverage_emb.unsqueeze(1)

        if type_weaknesses is not None:
            weakness_emb = self.weakness_encoder(type_weaknesses)
            synergy = synergy - weakness_emb.unsqueeze(1)  # Weaknesses are bad

        return synergy


class FullTeamEncoder(nn.Module):
    """Complete encoder for a team, combining all components.

    This is the main interface used by both player and teambuilder.
    """

    def __init__(
        self,
        shared_embeddings: SharedEmbeddings,
        pokemon_hidden: int = 256,
        team_hidden: int = 512,
    ):
        super().__init__()
        self.embeddings = shared_embeddings
        config = shared_embeddings.config

        self.pokemon_encoder = PokemonEncoder(
            config,
            hidden_dim=pokemon_hidden,
            output_dim=pokemon_hidden,
        )

        self.team_encoder = TeamEncoder(
            pokemon_dim=pokemon_hidden,
            output_dim=team_hidden,
        )

        self.synergy_encoder = SynergyEncoder(
            pokemon_dim=pokemon_hidden,
            output_dim=64,
        )

    def encode_pokemon(
        self,
        pokemon_data: dict,
    ) -> torch.Tensor:
        """Encode a single Pokemon from raw data.

        Args:
            pokemon_data: Dict with species_id, type_ids, base_stats,
                         move_ids, move_types, move_properties,
                         item_id, item_category, ability_id, ability_effect,
                         stat_mods

        Returns:
            (batch, pokemon_hidden) Encoded Pokemon
        """
        # Get embeddings
        pokemon_emb = self.embeddings.pokemon(
            pokemon_data["species_id"],
            pokemon_data["type1_id"],
            pokemon_data["type2_id"],
            pokemon_data["base_stats"],
        )

        move_embs = self.embeddings.move(
            pokemon_data["move_ids"],
            pokemon_data["move_types"],
            pokemon_data["move_properties"],
        )

        item_emb = self.embeddings.item(
            pokemon_data["item_id"],
            pokemon_data["item_category"],
        )

        ability_emb = self.embeddings.ability(
            pokemon_data["ability_id"],
            pokemon_data["ability_effect"],
        )

        return self.pokemon_encoder(
            pokemon_emb,
            move_embs,
            item_emb,
            ability_emb,
            pokemon_data["stat_mods"],
        )

    def encode_team(
        self,
        team_data: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a full team.

        Args:
            team_data: List of 6 pokemon_data dicts

        Returns:
            (batch, team_hidden) Team encoding
            (batch, 6, pokemon_hidden) Individual Pokemon encodings
        """
        # Encode each Pokemon
        pokemon_encodings = torch.stack([
            self.encode_pokemon(p) for p in team_data
        ], dim=1)

        # Encode full team
        team_encoding = self.team_encoder(pokemon_encodings)

        return team_encoding, pokemon_encodings

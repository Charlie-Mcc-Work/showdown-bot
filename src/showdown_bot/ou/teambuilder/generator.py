"""Team generator for autoregressive team construction.

The generator builds teams one Pokemon at a time, conditioning each
selection on the already-chosen team members. This allows it to:
1. Ensure type coverage
2. Build around specific cores
3. Avoid redundancy
4. Respect team roles (lead, sweeper, wall, etc.)

Architecture:
- Input: Partial team encoding + generation step
- Output: Distribution over next Pokemon/moves/item choices
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from showdown_bot.ou.shared.embeddings import SharedEmbeddings, EmbeddingConfig
from showdown_bot.ou.shared.encoders import FullTeamEncoder
from showdown_bot.ou.teambuilder.team_repr import PartialTeam, PokemonSlot


class TeamGenerator(nn.Module):
    """Autoregressive team generator.

    Generation proceeds in steps:
    1. Select Pokemon 1 (species + moves + item + ability)
    2. Select Pokemon 2, conditioning on Pokemon 1
    3. ... repeat until team is complete

    Each Pokemon selection is further broken down:
    1.1 Select species (from OU-viable Pokemon, ~100 options)
    1.2 Select moveset (4 moves from species' movepool)
    1.3 Select item
    1.4 Select ability
    1.5 Select nature/EVs (can use templates for simplicity)

    The generator uses a Transformer to encode the partial team and
    produce distributions over each choice.
    """

    def __init__(
        self,
        shared_embeddings: SharedEmbeddings,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_viable_pokemon: int = 100,  # OU-viable species
    ):
        super().__init__()

        self.embeddings = shared_embeddings
        config = shared_embeddings.config
        self.hidden_dim = hidden_dim
        self.num_viable_pokemon = num_viable_pokemon

        # Team encoder for conditioning
        self.team_encoder = FullTeamEncoder(
            shared_embeddings,
            pokemon_hidden=hidden_dim,
            team_hidden=hidden_dim,
        )

        # Generation step embedding (which slot are we filling?)
        self.step_embed = nn.Embedding(6, hidden_dim)

        # Transformer for autoregressive generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heads for each generation choice
        self.species_head = nn.Linear(hidden_dim, num_viable_pokemon)
        self.move_head = nn.Linear(hidden_dim, config.num_moves)
        self.item_head = nn.Linear(hidden_dim, config.num_items)
        self.ability_head = nn.Linear(hidden_dim, config.num_abilities)
        self.nature_head = nn.Linear(hidden_dim, 25)  # 25 natures
        self.ev_head = nn.Linear(hidden_dim, 50)  # Common EV spreads

        # Viable Pokemon mask (which species are OU-viable)
        self.register_buffer(
            "viable_pokemon_mask",
            torch.ones(num_viable_pokemon, dtype=torch.bool),
        )

        # Viable Pokemon IDs (will be set from data)
        self.viable_pokemon_ids: list[int] = []

    def set_viable_pokemon(self, pokemon_ids: list[int]) -> None:
        """Set which Pokemon are viable for team building.

        Args:
            pokemon_ids: List of species IDs that are OU-viable
        """
        self.viable_pokemon_ids = pokemon_ids[:self.num_viable_pokemon]

    def forward(
        self,
        partial_team: PartialTeam,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate next Pokemon for the team.

        Args:
            partial_team: The current partial team
            device: Device for tensors

        Returns:
            Dict with logits for each choice:
            - species_logits: (num_viable_pokemon,)
            - move_logits: (num_moves,) for each of 4 move slots
            - item_logits: (num_items,)
            - ability_logits: (num_abilities,)
            - etc.
        """
        device = device or next(self.parameters()).device

        # Encode current partial team
        # TODO: Implement actual encoding
        batch_size = 1
        team_encoding = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Get current generation step
        step = partial_team.num_filled()
        step_emb = self.step_embed(torch.tensor([step], device=device))

        # Combine team encoding with step embedding
        context = team_encoding + step_emb

        # Decode to get generation distribution
        # For now, just use a simple forward pass
        # In full implementation, would use proper autoregressive decoding
        hidden = context.unsqueeze(1)  # (batch, 1, hidden)

        # Get output logits
        species_logits = self.species_head(hidden.squeeze(1))
        move_logits = self.move_head(hidden.squeeze(1))
        item_logits = self.item_head(hidden.squeeze(1))
        ability_logits = self.ability_head(hidden.squeeze(1))

        # Mask out already-selected Pokemon
        selected_species = partial_team.get_species_list()
        for species in selected_species:
            # TODO: Map species name to viable_pokemon index and mask
            pass

        return {
            "species_logits": species_logits,
            "move_logits": move_logits,
            "item_logits": item_logits,
            "ability_logits": ability_logits,
        }

    def sample_pokemon(
        self,
        partial_team: PartialTeam,
        temperature: float = 1.0,
        device: torch.device | None = None,
    ) -> PokemonSlot:
        """Sample a complete Pokemon for the next slot.

        Args:
            partial_team: Current partial team
            temperature: Sampling temperature (higher = more random)
            device: Device for tensors

        Returns:
            A complete PokemonSlot
        """
        logits = self.forward(partial_team, device)

        # Sample species
        species_probs = F.softmax(logits["species_logits"] / temperature, dim=-1)
        species_idx = torch.multinomial(species_probs, 1).item()

        # TODO: Map index to species name
        # TODO: Sample moves, item, ability based on species

        return PokemonSlot(
            species="placeholder",  # TODO
            moves=["move1", "move2", "move3", "move4"],  # TODO
            item="leftovers",  # TODO
            ability="ability",  # TODO
            nature="Jolly",  # TODO
            evs={"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},  # TODO
            tera_type="Normal",  # TODO
        )

    def generate_team(
        self,
        temperature: float = 1.0,
        device: torch.device | None = None,
    ) -> PartialTeam:
        """Generate a complete team from scratch.

        Args:
            temperature: Sampling temperature
            device: Device for tensors

        Returns:
            A complete PartialTeam
        """
        team = PartialTeam()

        for i in range(6):
            slot = self.sample_pokemon(team, temperature, device)
            team.slots[i] = slot

        return team


class SpeciesSelector(nn.Module):
    """Dedicated module for species selection.

    Uses:
    - Current team composition
    - Type coverage needs
    - Role distribution
    - Usage-based priors

    To select the next species.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_species: int = 100,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_species = num_species

        # Species embeddings
        self.species_embed = nn.Embedding(num_species, hidden_dim)

        # Team context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Selection head
        self.selection_head = nn.Linear(hidden_dim, num_species)

    def forward(
        self,
        team_context: torch.Tensor,
        species_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute species selection logits.

        Args:
            team_context: (batch, hidden_dim) Encoded partial team
            species_mask: (batch, num_species) True for valid selections

        Returns:
            (batch, num_species) Selection logits
        """
        context = self.context_encoder(team_context)
        logits = self.selection_head(context)

        # Mask invalid species
        logits = logits.masked_fill(~species_mask, float("-inf"))

        return logits


class MovesetSelector(nn.Module):
    """Dedicated module for moveset selection.

    Given a species, selects 4 moves from its movepool.
    Uses:
    - Species role (sweeper, wall, support, etc.)
    - Team coverage needs
    - Common sets from usage data
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_moves: int = 800,
        max_movepool: int = 100,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_moves = num_moves
        self.max_movepool = max_movepool

        # Move embeddings
        self.move_embed = nn.Embedding(num_moves, hidden_dim // 2)

        # Context processor
        self.context_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Move selection (4 sequential selections with attention)
        self.move_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )

        self.selection_head = nn.Linear(hidden_dim, num_moves)

    def forward(
        self,
        species_context: torch.Tensor,
        movepool_ids: torch.Tensor,
        movepool_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select moves for a species.

        Args:
            species_context: (batch, hidden_dim) Species + team context
            movepool_ids: (batch, max_movepool) Move IDs in species' movepool
            movepool_mask: (batch, max_movepool) True for valid moves

        Returns:
            (batch, 4, num_moves) Move selection logits for each slot
        """
        # TODO: Implement proper autoregressive move selection
        batch_size = species_context.size(0)
        device = species_context.device

        # Placeholder: return uniform logits
        logits = torch.zeros(batch_size, 4, self.num_moves, device=device)
        return logits

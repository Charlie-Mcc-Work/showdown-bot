"""Neural network architecture for the Pokemon Showdown RL agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from showdown_bot.config import training_config
from showdown_bot.environment.state_encoder import StateEncoder


class PokemonEncoder(nn.Module):
    """Encodes a single Pokemon's features into a latent representation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Pokemon features (batch, num_pokemon, features)

        Returns:
            Encoded Pokemon (batch, num_pokemon, output_dim)
        """
        return self.mlp(x)


class TeamEncoder(nn.Module):
    """Encodes a team of Pokemon using attention."""

    def __init__(
        self,
        pokemon_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.pokemon_encoder = PokemonEncoder(
            input_dim=StateEncoder.POKEMON_FEATURES,
            hidden_dim=hidden_dim,
            output_dim=pokemon_dim,
        )

        # Transformer encoder for team-level reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pokemon_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable position embeddings for team slots
        self.position_embed = nn.Parameter(torch.randn(6, pokemon_dim) * 0.02)

    def forward(
        self,
        team: torch.Tensor,
        active_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            team: Team features (batch, 6, pokemon_features)
            active_idx: Index of active Pokemon (batch,)

        Returns:
            team_encoding: Full team encoding (batch, 6, pokemon_dim)
            active_encoding: Active Pokemon encoding (batch, pokemon_dim)
        """
        batch_size = team.shape[0]

        # Encode individual Pokemon
        encoded = self.pokemon_encoder(team)  # (batch, 6, pokemon_dim)

        # Add position embeddings
        encoded = encoded + self.position_embed.unsqueeze(0)

        # Apply transformer
        team_encoding = self.transformer(encoded)  # (batch, 6, pokemon_dim)

        # Extract active Pokemon encoding
        if active_idx is not None:
            # Gather active Pokemon encoding
            idx = active_idx.view(batch_size, 1, 1).expand(-1, -1, team_encoding.shape[-1])
            active_encoding = team_encoding.gather(1, idx).squeeze(1)
        else:
            # Default to first slot
            active_encoding = team_encoding[:, 0, :]

        return team_encoding, active_encoding


class PolicyValueNetwork(nn.Module):
    """Policy and value network for PPO.

    Architecture:
    - Encode player team with attention
    - Encode opponent team with attention
    - Encode field state
    - Combine and produce policy/value heads
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        pokemon_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_actions: int = 9,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pokemon_dim = pokemon_dim
        self.num_actions = num_actions

        # Team encoders
        self.player_encoder = TeamEncoder(
            pokemon_dim=pokemon_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.opponent_encoder = TeamEncoder(
            pokemon_dim=pokemon_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        # Field encoder
        self.field_encoder = nn.Sequential(
            nn.Linear(StateEncoder.FIELD_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pokemon_dim),
        )

        # Cross-attention between player active and opponent team
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=pokemon_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Combined encoder
        # Input: player_active + opponent_active + player_team_pool + opp_team_pool + field
        combined_dim = pokemon_dim * 5
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        player_pokemon: torch.Tensor,
        opponent_pokemon: torch.Tensor,
        player_active_idx: torch.Tensor,
        opponent_active_idx: torch.Tensor,
        field_state: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            player_pokemon: Player team (batch, 6, pokemon_features)
            opponent_pokemon: Opponent team (batch, 6, pokemon_features)
            player_active_idx: Player active Pokemon index (batch,)
            opponent_active_idx: Opponent active Pokemon index (batch,)
            field_state: Field conditions (batch, field_features)
            action_mask: Legal action mask (batch, num_actions)

        Returns:
            policy_logits: Action logits (batch, num_actions)
            value: State value (batch, 1)
        """
        # Encode teams
        player_team, player_active = self.player_encoder(player_pokemon, player_active_idx)
        opponent_team, opponent_active = self.opponent_encoder(opponent_pokemon, opponent_active_idx)

        # Encode field
        field_encoded = self.field_encoder(field_state)

        # Cross-attention: player active attends to opponent team
        player_query = player_active.unsqueeze(1)  # (batch, 1, pokemon_dim)
        cross_out, _ = self.cross_attention(
            query=player_query,
            key=opponent_team,
            value=opponent_team,
        )
        cross_out = cross_out.squeeze(1)  # (batch, pokemon_dim)

        # Pool team representations (mean over Pokemon)
        player_pool = player_team.mean(dim=1)
        opponent_pool = opponent_team.mean(dim=1)

        # Combine all features
        combined = torch.cat(
            [player_active, opponent_active, player_pool, opponent_pool, field_encoded],
            dim=-1,
        )
        hidden = self.combiner(combined)

        # Policy and value heads
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden)

        # Apply action mask (set illegal actions to very negative)
        masked_logits = policy_logits.clone()
        masked_logits[action_mask == 0] = float("-inf")

        return masked_logits, value

    def get_action_and_value(
        self,
        player_pokemon: torch.Tensor,
        opponent_pokemon: torch.Tensor,
        player_active_idx: torch.Tensor,
        opponent_active_idx: torch.Tensor,
        field_state: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value.

        Args:
            ... (same as forward)
            action: Optional action to evaluate (batch,)

        Returns:
            action: Sampled or provided action (batch,)
            log_prob: Log probability of action (batch,)
            entropy: Policy entropy (batch,)
            value: State value (batch,)
        """
        logits, value = self.forward(
            player_pokemon,
            opponent_pokemon,
            player_active_idx,
            opponent_active_idx,
            field_state,
            action_mask,
        )

        # Create categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    @classmethod
    def from_config(cls) -> "PolicyValueNetwork":
        """Create network from global config."""
        return cls(
            hidden_dim=training_config.hidden_dim,
            pokemon_dim=training_config.embedding_dim * 2,
            num_heads=training_config.num_attention_heads,
            num_layers=training_config.num_transformer_layers,
            num_actions=StateEncoder.NUM_ACTIONS,
        )

"""Neural network for OU battle decisions.

The player network takes encoded battle state and outputs:
1. Move/switch policy (with tera consideration)
2. State value estimate

Architecture differs from RandomBattles:
- Handles full team knowledge
- Processes opponent predictions
- Includes tera decision making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from showdown_bot.ou.player.state_encoder import OUEncodedState, OUStateEncoder


class OUPlayerNetwork(nn.Module):
    """Policy-value network for OU battles.

    Takes encoded battle state and produces:
    - Action probabilities (moves, switches, tera moves)
    - Value estimate

    Architecture:
    1. Process our team with self-attention
    2. Process opponent (revealed + predicted) with cross-attention
    3. Combine with field state
    4. Output policy and value heads
    """

    def __init__(
        self,
        pokemon_dim: int = 128,
        field_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        num_actions: int = 13,  # 4 moves + 4 tera moves + 5 switches
    ):
        super().__init__()

        self.pokemon_dim = pokemon_dim
        self.field_dim = field_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # Project Pokemon features to hidden dim
        self.pokemon_proj = nn.Linear(pokemon_dim, hidden_dim)

        # Self-attention over our team
        self.our_team_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Cross-attention: our active Pokemon attends to opponent team
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Transformer for combined processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Field state projection
        self.field_proj = nn.Linear(field_dim, hidden_dim)

        # Tera decision embedding
        self.tera_available_embed = nn.Embedding(2, hidden_dim // 4)

        # Combine all features
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 3 + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
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

        # Tera advantage head (should we tera this turn?)
        self.tera_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state: OUEncodedState,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the network.

        Args:
            state: Encoded battle state

        Returns:
            Dict with:
            - policy_logits: (batch, num_actions) raw logits
            - policy: (batch, num_actions) masked probabilities
            - value: (batch, 1) state value estimate
            - tera_advantage: (batch, 1) expected value of terastallizing
        """
        # Add batch dimension if needed
        our_team = state.our_team
        if our_team.dim() == 2:
            our_team = our_team.unsqueeze(0)
            opp_revealed = state.opp_revealed.unsqueeze(0)
            opp_predicted = state.opp_predicted.unsqueeze(0)
            field_state = state.field_state.unsqueeze(0)
            action_mask = state.action_mask.unsqueeze(0)
            opp_mask = state.opp_revealed_mask.unsqueeze(0)
        else:
            opp_revealed = state.opp_revealed
            opp_predicted = state.opp_predicted
            field_state = state.field_state
            action_mask = state.action_mask
            opp_mask = state.opp_revealed_mask

        batch_size = our_team.size(0)
        device = our_team.device

        # Project Pokemon to hidden dim
        our_team_h = self.pokemon_proj(our_team)  # (B, 6, hidden)
        opp_revealed_h = self.pokemon_proj(opp_revealed)  # (B, 6, hidden)
        opp_predicted_h = self.pokemon_proj(opp_predicted)  # (B, 6, hidden)

        # Combine revealed and predicted opponent info
        # Use revealed where available, predicted otherwise
        opp_mask_expanded = opp_mask.unsqueeze(-1).float()
        opp_team_h = opp_revealed_h * opp_mask_expanded + opp_predicted_h * (1 - opp_mask_expanded)

        # Self-attention over our team
        our_team_attended, _ = self.our_team_attention(
            our_team_h, our_team_h, our_team_h
        )

        # Get our active Pokemon representation
        active_idx = state.our_active_idx
        if isinstance(active_idx, int):
            our_active = our_team_attended[:, active_idx:active_idx+1, :]
        else:
            # Batch of indices
            our_active = torch.stack([
                our_team_attended[i, idx:idx+1, :]
                for i, idx in enumerate(active_idx)
            ])

        # Cross-attention: our active attends to opponent team
        cross_attended, _ = self.cross_attention(
            our_active, opp_team_h, opp_team_h
        )

        # Process through transformer
        # Combine our team and opponent team for joint reasoning
        combined = torch.cat([our_team_attended, opp_team_h], dim=1)  # (B, 12, hidden)
        transformed = self.transformer(combined)

        # Pool to get global representation
        global_repr = transformed.mean(dim=1)  # (B, hidden)

        # Process field state
        field_h = self.field_proj(field_state)  # (B, hidden)

        # Tera availability embedding
        tera_avail = torch.tensor(
            [0 if state.our_tera_used else 1],
            device=device,
        ).expand(batch_size)
        tera_embed = self.tera_available_embed(tera_avail)  # (B, hidden//4)

        # Combine all features
        active_repr = cross_attended.squeeze(1)  # (B, hidden)
        combined_features = torch.cat([
            global_repr,
            active_repr,
            field_h,
            tera_embed,
        ], dim=-1)

        hidden = self.combiner(combined_features)

        # Policy
        policy_logits = self.policy_head(hidden)

        # Apply action mask
        masked_logits = policy_logits.masked_fill(
            action_mask == 0,
            float("-inf"),
        )
        policy = F.softmax(masked_logits, dim=-1)

        # Value
        value = self.value_head(hidden)

        # Tera advantage
        tera_advantage = self.tera_head(hidden)

        return {
            "policy_logits": policy_logits,
            "policy": policy,
            "value": value,
            "tera_advantage": tera_advantage,
        }

    def get_action(
        self,
        state: OUEncodedState,
        deterministic: bool = False,
    ) -> tuple[int, dict[str, torch.Tensor]]:
        """Select an action given the current state.

        Args:
            state: Encoded battle state
            deterministic: If True, take argmax. If False, sample.

        Returns:
            (action_idx, info_dict)
        """
        with torch.no_grad():
            output = self.forward(state)
            policy = output["policy"]

            if deterministic:
                action = policy.argmax(dim=-1).item()
            else:
                action = torch.multinomial(policy, 1).item()

            return action, output

    def get_value(self, state: OUEncodedState) -> float:
        """Get value estimate for a state."""
        with torch.no_grad():
            output = self.forward(state)
            return output["value"].item()


class OUActorCritic(nn.Module):
    """Combined actor-critic for PPO training.

    Wraps OUPlayerNetwork with additional training utilities.
    """

    def __init__(
        self,
        state_encoder: OUStateEncoder,
        network: OUPlayerNetwork | None = None,
        **network_kwargs,
    ):
        super().__init__()

        self.encoder = state_encoder
        self.network = network or OUPlayerNetwork(**network_kwargs)

    def forward(self, state: OUEncodedState) -> dict[str, torch.Tensor]:
        """Forward pass."""
        return self.network(state)

    def evaluate_actions(
        self,
        states: list[OUEncodedState],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Args:
            states: List of encoded states
            actions: (batch,) action indices

        Returns:
            (log_probs, values, entropies)
        """
        log_probs = []
        values = []
        entropies = []

        for i, state in enumerate(states):
            output = self.forward(state)
            policy = output["policy"]
            value = output["value"]

            # Get log prob of taken action
            action_idx = actions[i].item()
            log_prob = torch.log(policy[0, action_idx] + 1e-8)
            log_probs.append(log_prob)

            values.append(value.squeeze())

            # Entropy for exploration
            entropy = -(policy * torch.log(policy + 1e-8)).sum()
            entropies.append(entropy)

        return (
            torch.stack(log_probs),
            torch.stack(values),
            torch.stack(entropies),
        )

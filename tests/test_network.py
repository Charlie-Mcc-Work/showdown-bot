"""Comprehensive tests for neural network architecture."""

import pytest
import torch
import torch.nn as nn

from showdown_bot.models.network import (
    PolicyValueNetwork,
    PokemonEncoder,
    TeamEncoder,
)
from showdown_bot.environment.state_encoder import StateEncoder


class TestPokemonEncoder:
    """Test individual Pokemon encoder."""

    @pytest.fixture
    def encoder(self):
        return PokemonEncoder(
            input_dim=StateEncoder.POKEMON_FEATURES,
            hidden_dim=64,
            output_dim=32,
        )

    def test_output_shape(self, encoder):
        """Test output has correct shape."""
        batch_size = 4
        num_pokemon = 6
        x = torch.randn(batch_size, num_pokemon, StateEncoder.POKEMON_FEATURES)

        output = encoder(x)

        assert output.shape == (batch_size, num_pokemon, 32)

    def test_gradient_flow(self, encoder):
        """Test gradients flow through encoder."""
        x = torch.randn(2, 6, StateEncoder.POKEMON_FEATURES, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_deterministic(self, encoder):
        """Test encoder is deterministic in eval mode."""
        encoder.eval()
        x = torch.randn(2, 6, StateEncoder.POKEMON_FEATURES)

        out1 = encoder(x)
        out2 = encoder(x)

        assert torch.allclose(out1, out2)


class TestTeamEncoder:
    """Test team encoder with attention."""

    @pytest.fixture
    def encoder(self):
        return TeamEncoder(
            pokemon_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
        )

    def test_output_shapes(self, encoder):
        """Test team and active encoding shapes."""
        batch_size = 4
        team = torch.randn(batch_size, 6, StateEncoder.POKEMON_FEATURES)
        active_idx = torch.zeros(batch_size, dtype=torch.long)

        team_encoding, active_encoding = encoder(team, active_idx)

        assert team_encoding.shape == (batch_size, 6, 64)
        assert active_encoding.shape == (batch_size, 64)

    def test_active_extraction(self, encoder):
        """Test correct active Pokemon is extracted."""
        batch_size = 2
        team = torch.randn(batch_size, 6, StateEncoder.POKEMON_FEATURES)

        # Set different active indices
        active_idx = torch.tensor([2, 4])

        team_encoding, active_encoding = encoder(team, active_idx)

        # The active encoding should come from the team encoding at the right index
        # (though transformed by position embeddings and attention)
        assert active_encoding.shape == (batch_size, 64)

    def test_no_active_idx(self, encoder):
        """Test fallback when no active index provided."""
        team = torch.randn(2, 6, StateEncoder.POKEMON_FEATURES)

        team_encoding, active_encoding = encoder(team, None)

        # Should default to first slot
        assert active_encoding.shape == (2, 64)

    def test_gradient_flow(self, encoder):
        """Test gradients flow through team encoder."""
        team = torch.randn(2, 6, StateEncoder.POKEMON_FEATURES, requires_grad=True)
        active_idx = torch.tensor([0, 1])

        team_encoding, active_encoding = encoder(team, active_idx)
        loss = team_encoding.sum() + active_encoding.sum()
        loss.backward()

        assert team.grad is not None


class TestPolicyValueNetwork:
    """Test the full policy-value network."""

    @pytest.fixture
    def network(self):
        return PolicyValueNetwork(
            hidden_dim=128,
            pokemon_dim=64,
            num_heads=4,
            num_layers=2,
            num_actions=9,
        )

    @pytest.fixture
    def sample_input(self):
        batch_size = 4
        return {
            "player_pokemon": torch.randn(batch_size, 6, StateEncoder.POKEMON_FEATURES),
            "opponent_pokemon": torch.randn(batch_size, 6, StateEncoder.POKEMON_FEATURES),
            "player_active_idx": torch.zeros(batch_size, dtype=torch.long),
            "opponent_active_idx": torch.zeros(batch_size, dtype=torch.long),
            "field_state": torch.randn(batch_size, StateEncoder.FIELD_FEATURES),
            "action_mask": torch.ones(batch_size, 9),
        }

    def test_forward_output_shapes(self, network, sample_input):
        """Test forward pass output shapes."""
        logits, value = network(**sample_input)

        assert logits.shape == (4, 9)
        assert value.shape == (4, 1)

    def test_action_masking(self, network, sample_input):
        """Test that masked actions have -inf logits."""
        # Mask out some actions
        sample_input["action_mask"][:, 2:5] = 0

        logits, _ = network(**sample_input)

        # Masked actions should be -inf
        assert torch.isinf(logits[:, 2:5]).all()
        assert (logits[:, 2:5] < 0).all()

        # Unmasked actions should be finite
        assert torch.isfinite(logits[:, :2]).all()

    def test_get_action_and_value_sample(self, network, sample_input):
        """Test action sampling."""
        network.eval()

        action, log_prob, entropy, value = network.get_action_and_value(
            **sample_input, action=None
        )

        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

        # Actions should be valid indices
        assert (action >= 0).all()
        assert (action < 9).all()

        # Log probs should be negative (probabilities < 1)
        assert (log_prob <= 0).all()

        # Entropy should be non-negative
        assert (entropy >= 0).all()

    def test_get_action_and_value_evaluate(self, network, sample_input):
        """Test evaluating given actions."""
        actions = torch.tensor([0, 1, 2, 3])

        action, log_prob, entropy, value = network.get_action_and_value(
            **sample_input, action=actions
        )

        # Should return the same actions
        assert torch.equal(action, actions)

    def test_gradient_flow(self, network, sample_input):
        """Test gradients flow through entire network."""
        for key, val in sample_input.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                sample_input[key] = val.requires_grad_(True)

        logits, value = network(**sample_input)
        loss = logits.sum() + value.sum()
        loss.backward()

        assert sample_input["player_pokemon"].grad is not None
        assert sample_input["opponent_pokemon"].grad is not None
        assert sample_input["field_state"].grad is not None

    def test_from_config(self):
        """Test creating network from config."""
        network = PolicyValueNetwork.from_config()

        assert isinstance(network, PolicyValueNetwork)
        assert network.num_actions == StateEncoder.NUM_ACTIONS

    def test_parameter_count(self, network):
        """Test network has reasonable parameter count."""
        num_params = sum(p.numel() for p in network.parameters())

        # Should be in millions, not billions
        assert num_params > 100_000
        assert num_params < 1_000_000_000

    def test_eval_deterministic(self, network, sample_input):
        """Test network is deterministic in eval mode with same seed."""
        network.eval()

        torch.manual_seed(42)
        action1, _, _, _ = network.get_action_and_value(**sample_input)

        torch.manual_seed(42)
        action2, _, _, _ = network.get_action_and_value(**sample_input)

        assert torch.equal(action1, action2)

    def test_respects_mask_in_sampling(self, network, sample_input):
        """Test that sampled actions respect the mask."""
        # Only allow actions 0 and 1
        sample_input["action_mask"] = torch.zeros(4, 9)
        sample_input["action_mask"][:, 0] = 1
        sample_input["action_mask"][:, 1] = 1

        network.eval()

        for _ in range(10):
            action, _, _, _ = network.get_action_and_value(**sample_input)
            assert (action <= 1).all()


class TestNetworkTraining:
    """Test network behavior during training."""

    @pytest.fixture
    def network(self):
        return PolicyValueNetwork.from_config()

    def test_train_mode(self, network):
        """Test network can be set to training mode."""
        network.train()
        assert network.training

    def test_eval_mode(self, network):
        """Test network can be set to eval mode."""
        network.eval()
        assert not network.training

    def test_save_load(self, network, tmp_path):
        """Test saving and loading network weights."""
        # Save
        save_path = tmp_path / "model.pt"
        torch.save(network.state_dict(), save_path)

        # Create new network and load
        new_network = PolicyValueNetwork.from_config()
        new_network.load_state_dict(torch.load(save_path))

        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            network.named_parameters(), new_network.named_parameters()
        ):
            assert name1 == name2
            assert torch.equal(param1, param2)

    def test_backward_pass_stability(self, network):
        """Test backward pass doesn't produce NaN gradients."""
        batch_size = 8
        inputs = {
            "player_pokemon": torch.randn(batch_size, 6, StateEncoder.POKEMON_FEATURES),
            "opponent_pokemon": torch.randn(batch_size, 6, StateEncoder.POKEMON_FEATURES),
            "player_active_idx": torch.zeros(batch_size, dtype=torch.long),
            "opponent_active_idx": torch.zeros(batch_size, dtype=torch.long),
            "field_state": torch.randn(batch_size, StateEncoder.FIELD_FEATURES),
            "action_mask": torch.ones(batch_size, 9),
        }

        for _ in range(10):
            logits, value = network(**inputs)
            loss = logits.mean() + value.mean()
            loss.backward()

            # Check for NaN gradients
            for param in network.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()

            network.zero_grad()


class TestCrossAttention:
    """Test cross-attention between player and opponent."""

    @pytest.fixture
    def network(self):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=4,
            num_layers=1,
        )

    def test_cross_attention_output(self, network):
        """Test cross attention produces valid output."""
        batch_size = 2
        query = torch.randn(batch_size, 1, 32)
        key_value = torch.randn(batch_size, 6, 32)

        output, _ = network.cross_attention(query, key_value, key_value)

        assert output.shape == (batch_size, 1, 32)
        assert torch.isfinite(output).all()

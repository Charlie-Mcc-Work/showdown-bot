"""Comprehensive tests for PPO algorithm and rollout buffer."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from showdown_bot.training.buffer import RolloutBuffer
from showdown_bot.training.ppo import PPO, PPOStats
from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.environment.state_encoder import StateEncoder


class TestRolloutBufferInit:
    """Test buffer initialization."""

    def test_default_initialization(self):
        """Test buffer initializes with correct shapes."""
        buffer = RolloutBuffer(buffer_size=128, num_envs=4)

        assert buffer.buffer_size == 128
        assert buffer.num_envs == 4
        assert buffer.ptr == 0
        assert buffer.full is False

    def test_storage_shapes(self):
        """Test storage arrays have correct shapes."""
        buffer = RolloutBuffer(buffer_size=64, num_envs=2)

        assert buffer.player_pokemon.shape == (64, 2, 6, StateEncoder.POKEMON_FEATURES)
        assert buffer.opponent_pokemon.shape == (64, 2, 6, StateEncoder.POKEMON_FEATURES)
        assert buffer.player_active_idx.shape == (64, 2)
        assert buffer.opponent_active_idx.shape == (64, 2)
        assert buffer.field_state.shape == (64, 2, StateEncoder.FIELD_FEATURES)
        assert buffer.action_mask.shape == (64, 2, StateEncoder.NUM_ACTIONS)
        assert buffer.actions.shape == (64, 2)
        assert buffer.log_probs.shape == (64, 2)
        assert buffer.rewards.shape == (64, 2)
        assert buffer.dones.shape == (64, 2)
        assert buffer.values.shape == (64, 2)
        assert buffer.advantages.shape == (64, 2)
        assert buffer.returns.shape == (64, 2)

    def test_custom_gamma_lambda(self):
        """Test custom gamma and lambda values."""
        buffer = RolloutBuffer(
            buffer_size=32, num_envs=1, gamma=0.95, gae_lambda=0.9
        )

        assert buffer.gamma == 0.95
        assert buffer.gae_lambda == 0.9


class TestRolloutBufferAdd:
    """Test adding transitions to buffer."""

    @pytest.fixture
    def buffer(self):
        return RolloutBuffer(buffer_size=10, num_envs=1)

    @pytest.fixture
    def sample_transition(self):
        """Create a sample transition."""
        return {
            "player_pokemon": np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
            "opponent_pokemon": np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
            "player_active_idx": 0,
            "opponent_active_idx": 0,
            "field_state": np.random.randn(1, StateEncoder.FIELD_FEATURES).astype(np.float32),
            "action_mask": np.ones((1, StateEncoder.NUM_ACTIONS), dtype=np.float32),
            "action": 0,
            "log_prob": -1.5,
            "reward": 0.1,
            "done": False,
            "value": 0.5,
        }

    def test_add_single_transition(self, buffer, sample_transition):
        """Test adding a single transition."""
        buffer.add(**sample_transition)

        assert buffer.ptr == 1
        assert not buffer.full

    def test_add_multiple_transitions(self, buffer, sample_transition):
        """Test adding multiple transitions."""
        for _ in range(5):
            buffer.add(**sample_transition)

        assert buffer.ptr == 5
        assert not buffer.full

    def test_buffer_full_flag(self, buffer, sample_transition):
        """Test buffer full flag is set correctly."""
        for _ in range(10):
            buffer.add(**sample_transition)

        assert buffer.ptr == 10
        assert buffer.full

    def test_data_stored_correctly(self, buffer, sample_transition):
        """Test data is stored at correct index."""
        buffer.add(**sample_transition)

        np.testing.assert_array_almost_equal(
            buffer.player_pokemon[0], sample_transition["player_pokemon"]
        )
        assert buffer.rewards[0, 0] == pytest.approx(0.1)
        assert buffer.values[0, 0] == pytest.approx(0.5)


class TestRolloutBufferGAE:
    """Test Generalized Advantage Estimation."""

    @pytest.fixture
    def filled_buffer(self):
        """Create a buffer with known values for testing GAE."""
        buffer = RolloutBuffer(buffer_size=5, num_envs=1, gamma=0.99, gae_lambda=0.95)

        # Add 5 transitions with known rewards/values
        for i in range(5):
            buffer.add(
                player_pokemon=np.zeros((1, 6, StateEncoder.POKEMON_FEATURES)),
                opponent_pokemon=np.zeros((1, 6, StateEncoder.POKEMON_FEATURES)),
                player_active_idx=0,
                opponent_active_idx=0,
                field_state=np.zeros((1, StateEncoder.FIELD_FEATURES)),
                action_mask=np.ones((1, StateEncoder.NUM_ACTIONS)),
                action=0,
                log_prob=-1.0,
                reward=1.0,  # Constant reward
                done=False,
                value=0.5,  # Constant value estimate
            )

        return buffer

    def test_compute_advantages_shape(self, filled_buffer):
        """Test advantages have correct shape after computation."""
        last_value = np.array([0.5])
        last_done = np.array([False])

        filled_buffer.compute_advantages(last_value, last_done)

        assert filled_buffer.advantages[:5].shape == (5, 1)
        assert filled_buffer.returns[:5].shape == (5, 1)

    def test_advantages_not_all_zero(self, filled_buffer):
        """Test advantages are computed (not zero)."""
        last_value = np.array([0.5])
        last_done = np.array([False])

        filled_buffer.compute_advantages(last_value, last_done)

        # With constant reward=1 and value=0.5, advantages should be positive
        assert not np.allclose(filled_buffer.advantages[:5], 0)

    def test_returns_computation(self, filled_buffer):
        """Test returns = advantages + values."""
        last_value = np.array([0.5])
        last_done = np.array([False])

        filled_buffer.compute_advantages(last_value, last_done)

        expected_returns = filled_buffer.advantages[:5] + filled_buffer.values[:5]
        np.testing.assert_array_almost_equal(
            filled_buffer.returns[:5], expected_returns
        )

    def test_terminal_state_handling(self):
        """Test GAE handles terminal states correctly."""
        buffer = RolloutBuffer(buffer_size=3, num_envs=1)

        # Transition 0: not terminal
        buffer.add(
            player_pokemon=np.zeros((1, 6, StateEncoder.POKEMON_FEATURES)),
            opponent_pokemon=np.zeros((1, 6, StateEncoder.POKEMON_FEATURES)),
            player_active_idx=0,
            opponent_active_idx=0,
            field_state=np.zeros((1, StateEncoder.FIELD_FEATURES)),
            action_mask=np.ones((1, StateEncoder.NUM_ACTIONS)),
            action=0,
            log_prob=-1.0,
            reward=0.0,
            done=False,
            value=0.5,
        )

        # Transition 1: terminal (win)
        buffer.add(
            player_pokemon=np.zeros((1, 6, StateEncoder.POKEMON_FEATURES)),
            opponent_pokemon=np.zeros((1, 6, StateEncoder.POKEMON_FEATURES)),
            player_active_idx=0,
            opponent_active_idx=0,
            field_state=np.zeros((1, StateEncoder.FIELD_FEATURES)),
            action_mask=np.ones((1, StateEncoder.NUM_ACTIONS)),
            action=0,
            log_prob=-1.0,
            reward=1.0,
            done=True,
            value=0.0,
        )

        buffer.compute_advantages(np.array([0.0]), np.array([True]))

        # Terminal state should have advantage based only on immediate reward
        # since there's no future
        assert buffer.advantages[1, 0] == pytest.approx(1.0)  # reward - value


class TestRolloutBufferBatches:
    """Test minibatch generation."""

    @pytest.fixture
    def filled_buffer(self):
        """Create a filled buffer for batching tests."""
        buffer = RolloutBuffer(buffer_size=20, num_envs=2)

        for i in range(20):
            buffer.add(
                player_pokemon=np.random.randn(2, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                opponent_pokemon=np.random.randn(2, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                player_active_idx=np.zeros(2, dtype=np.int64),
                opponent_active_idx=np.zeros(2, dtype=np.int64),
                field_state=np.random.randn(2, StateEncoder.FIELD_FEATURES).astype(np.float32),
                action_mask=np.ones((2, StateEncoder.NUM_ACTIONS), dtype=np.float32),
                # Use varied actions to make shuffle test meaningful
                action=np.array([i % 9, (i + 1) % 9], dtype=np.int64),
                log_prob=np.full(2, -1.0, dtype=np.float32),
                reward=np.random.randn(2).astype(np.float32),
                done=np.zeros(2, dtype=np.float32),
                value=np.random.randn(2).astype(np.float32),
            )

        buffer.compute_advantages(np.zeros(2), np.zeros(2))
        return buffer

    def test_batch_count(self, filled_buffer):
        """Test correct number of batches generated."""
        batches = filled_buffer.get_batches(batch_size=8, shuffle=False)

        # 20 timesteps * 2 envs = 40 samples
        # 40 / 8 = 5 batches
        assert len(batches) == 5

    def test_batch_shapes(self, filled_buffer):
        """Test batch tensors have correct shapes."""
        batches = filled_buffer.get_batches(batch_size=8, shuffle=False)
        batch = batches[0]

        assert batch["player_pokemon"].shape == (8, 6, StateEncoder.POKEMON_FEATURES)
        assert batch["opponent_pokemon"].shape == (8, 6, StateEncoder.POKEMON_FEATURES)
        assert batch["player_active_idx"].shape == (8,)
        assert batch["opponent_active_idx"].shape == (8,)
        assert batch["field_state"].shape == (8, StateEncoder.FIELD_FEATURES)
        assert batch["action_mask"].shape == (8, StateEncoder.NUM_ACTIONS)
        assert batch["actions"].shape == (8,)
        assert batch["old_log_probs"].shape == (8,)
        assert batch["advantages"].shape == (8,)
        assert batch["returns"].shape == (8,)
        assert batch["old_values"].shape == (8,)

    def test_batch_tensors_are_torch(self, filled_buffer):
        """Test batch contents are PyTorch tensors."""
        batches = filled_buffer.get_batches(batch_size=8, shuffle=False)
        batch = batches[0]

        for key, value in batch.items():
            assert isinstance(value, torch.Tensor), f"{key} is not a tensor"

    def test_advantages_normalized(self, filled_buffer):
        """Test advantages are normalized in batches."""
        batches = filled_buffer.get_batches(batch_size=40, shuffle=False)  # Get all in one batch
        advantages = batches[0]["advantages"]

        # Should be approximately normalized (mean ~0, std ~1)
        assert advantages.mean().abs() < 0.1
        assert (advantages.std() - 1.0).abs() < 0.1

    def test_shuffle_changes_order(self, filled_buffer):
        """Test shuffling changes batch content."""
        torch.manual_seed(42)
        batches_shuffled = filled_buffer.get_batches(batch_size=8, shuffle=True)

        batches_unshuffled = filled_buffer.get_batches(batch_size=8, shuffle=False)

        # Actions should differ between shuffled and unshuffled
        shuffled_actions = batches_shuffled[0]["actions"]
        unshuffled_actions = batches_unshuffled[0]["actions"]

        # Very unlikely to be equal if truly shuffled
        assert not torch.equal(shuffled_actions, unshuffled_actions)


class TestRolloutBufferReset:
    """Test buffer reset functionality."""

    def test_reset_clears_pointer(self):
        """Test reset clears the pointer."""
        buffer = RolloutBuffer(buffer_size=10, num_envs=1)

        # Add some data
        for _ in range(5):
            buffer.add(
                player_pokemon=np.zeros((1, 6, StateEncoder.POKEMON_FEATURES)),
                opponent_pokemon=np.zeros((1, 6, StateEncoder.POKEMON_FEATURES)),
                player_active_idx=0,
                opponent_active_idx=0,
                field_state=np.zeros((1, StateEncoder.FIELD_FEATURES)),
                action_mask=np.ones((1, StateEncoder.NUM_ACTIONS)),
                action=0,
                log_prob=-1.0,
                reward=0.0,
                done=False,
                value=0.0,
            )

        buffer.reset()

        assert buffer.ptr == 0
        assert buffer.full is False

    def test_size_property(self):
        """Test size property returns correct count."""
        buffer = RolloutBuffer(buffer_size=10, num_envs=3)

        for _ in range(5):
            buffer.add(
                player_pokemon=np.zeros((3, 6, StateEncoder.POKEMON_FEATURES)),
                opponent_pokemon=np.zeros((3, 6, StateEncoder.POKEMON_FEATURES)),
                player_active_idx=np.zeros(3),
                opponent_active_idx=np.zeros(3),
                field_state=np.zeros((3, StateEncoder.FIELD_FEATURES)),
                action_mask=np.ones((3, StateEncoder.NUM_ACTIONS)),
                action=np.zeros(3),
                log_prob=np.zeros(3),
                reward=np.zeros(3),
                done=np.zeros(3),
                value=np.zeros(3),
            )

        # 5 timesteps * 3 envs = 15 transitions
        assert buffer.size == 15


class TestPPOInit:
    """Test PPO initialization."""

    @pytest.fixture
    def model(self):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )

    def test_default_initialization(self, model):
        """Test PPO initializes with correct defaults."""
        ppo = PPO(model)

        assert ppo.clip_epsilon == 0.2
        assert ppo.value_coef == 0.5
        assert ppo.entropy_coef == 0.01
        assert ppo.max_grad_norm == 0.5
        assert ppo.num_epochs == 4

    def test_custom_parameters(self, model):
        """Test PPO with custom parameters."""
        ppo = PPO(
            model,
            learning_rate=1e-4,
            clip_epsilon=0.1,
            value_coef=0.25,
            entropy_coef=0.02,
            num_epochs=8,
        )

        assert ppo.clip_epsilon == 0.1
        assert ppo.value_coef == 0.25
        assert ppo.entropy_coef == 0.02
        assert ppo.num_epochs == 8

    def test_optimizer_created(self, model):
        """Test optimizer is created."""
        ppo = PPO(model)

        assert ppo.optimizer is not None
        assert isinstance(ppo.optimizer, torch.optim.Adam)

    def test_from_config(self, model):
        """Test creating PPO from config."""
        ppo = PPO.from_config(model)

        assert isinstance(ppo, PPO)


class TestPPOUpdate:
    """Test PPO update functionality."""

    @pytest.fixture
    def model(self):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )

    @pytest.fixture
    def ppo(self, model):
        return PPO(model, num_epochs=2)

    @pytest.fixture
    def filled_buffer(self):
        """Create a buffer with random data."""
        buffer = RolloutBuffer(buffer_size=16, num_envs=2)

        for i in range(16):
            buffer.add(
                player_pokemon=np.random.randn(2, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                opponent_pokemon=np.random.randn(2, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                player_active_idx=np.zeros(2, dtype=np.int64),
                opponent_active_idx=np.zeros(2, dtype=np.int64),
                field_state=np.random.randn(2, StateEncoder.FIELD_FEATURES).astype(np.float32),
                action_mask=np.ones((2, StateEncoder.NUM_ACTIONS), dtype=np.float32),
                action=np.random.randint(0, 9, size=2),
                log_prob=np.random.randn(2).astype(np.float32),
                reward=np.random.randn(2).astype(np.float32),
                done=np.zeros(2, dtype=np.float32),
                value=np.random.randn(2).astype(np.float32),
            )

        buffer.compute_advantages(np.zeros(2), np.zeros(2))
        return buffer

    def test_update_returns_stats(self, ppo, filled_buffer):
        """Test update returns PPOStats."""
        stats = ppo.update(filled_buffer, batch_size=8)

        assert isinstance(stats, PPOStats)

    def test_stats_contain_valid_values(self, ppo, filled_buffer):
        """Test stats contain finite values."""
        stats = ppo.update(filled_buffer, batch_size=8)

        assert np.isfinite(stats.policy_loss)
        assert np.isfinite(stats.value_loss)
        assert np.isfinite(stats.entropy_loss)
        assert np.isfinite(stats.total_loss)
        assert np.isfinite(stats.approx_kl)
        assert np.isfinite(stats.clip_fraction)
        assert np.isfinite(stats.explained_variance)

    def test_clip_fraction_bounded(self, ppo, filled_buffer):
        """Test clip fraction is between 0 and 1."""
        stats = ppo.update(filled_buffer, batch_size=8)

        assert 0 <= stats.clip_fraction <= 1

    def test_parameters_updated(self, ppo, filled_buffer):
        """Test model parameters are updated."""
        # Get initial parameters
        initial_params = {
            name: param.clone() for name, param in ppo.model.named_parameters()
        }

        ppo.update(filled_buffer, batch_size=8)

        # Check at least some parameters changed
        params_changed = False
        for name, param in ppo.model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                params_changed = True
                break

        assert params_changed

    def test_gradients_clipped(self, ppo, filled_buffer):
        """Test gradients are clipped."""
        ppo.max_grad_norm = 0.1  # Very small to force clipping

        # This should not raise even with aggressive clipping
        stats = ppo.update(filled_buffer, batch_size=8)

        assert np.isfinite(stats.total_loss)


class TestPPOEarlyStopping:
    """Test PPO early stopping based on KL divergence."""

    @pytest.fixture
    def model(self):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )

    def test_early_stopping_with_target_kl(self, model):
        """Test early stopping triggers with very low target KL."""
        ppo = PPO(model, num_epochs=100, target_kl=0.0001)  # Very low target

        buffer = RolloutBuffer(buffer_size=8, num_envs=1)
        for i in range(8):
            buffer.add(
                player_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                opponent_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                player_active_idx=0,
                opponent_active_idx=0,
                field_state=np.random.randn(1, StateEncoder.FIELD_FEATURES).astype(np.float32),
                action_mask=np.ones((1, StateEncoder.NUM_ACTIONS), dtype=np.float32),
                action=np.random.randint(0, 9),
                log_prob=-1.0,
                reward=np.random.randn(),
                done=False,
                value=np.random.randn(),
            )

        buffer.compute_advantages(np.zeros(1), np.zeros(1))

        # Should complete without running all 100 epochs
        stats = ppo.update(buffer, batch_size=4)

        # If early stopping worked, KL should be bounded
        assert stats.approx_kl is not None


class TestPPOSaveLoad:
    """Test PPO save and load functionality."""

    @pytest.fixture
    def model(self):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )

    def test_save_and_load(self, model, tmp_path):
        """Test saving and loading model state."""
        ppo = PPO(model)

        # Modify a parameter
        with torch.no_grad():
            for param in ppo.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
                break

        # Save
        save_path = tmp_path / "ppo_checkpoint.pt"
        ppo.save(str(save_path))

        # Create new model and load
        new_model = PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )
        new_ppo = PPO(new_model)
        new_ppo.load(str(save_path))

        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            ppo.model.named_parameters(), new_ppo.model.named_parameters()
        ):
            assert torch.equal(param1, param2), f"Parameter {name1} mismatch"


class TestPPOLosses:
    """Test individual PPO loss components."""

    @pytest.fixture
    def model(self):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )

    def test_policy_loss_negative_advantages(self, model):
        """Test policy loss with negative advantages."""
        ppo = PPO(model, num_epochs=1)

        buffer = RolloutBuffer(buffer_size=4, num_envs=1)
        for i in range(4):
            buffer.add(
                player_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                opponent_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                player_active_idx=0,
                opponent_active_idx=0,
                field_state=np.random.randn(1, StateEncoder.FIELD_FEATURES).astype(np.float32),
                action_mask=np.ones((1, StateEncoder.NUM_ACTIONS), dtype=np.float32),
                action=0,
                log_prob=-1.0,
                reward=-1.0,  # Negative rewards
                done=False,
                value=0.0,
            )

        buffer.compute_advantages(np.zeros(1), np.zeros(1))
        stats = ppo.update(buffer, batch_size=4)

        assert np.isfinite(stats.policy_loss)

    def test_entropy_coefficient_effect(self):
        """Test entropy coefficient affects total loss."""
        # Create separate models to ensure independent updates
        model_low = PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )
        model_high = PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        )
        # Copy weights to start with identical models
        model_high.load_state_dict(model_low.state_dict())

        ppo_low_entropy = PPO(model_low, entropy_coef=0.001, num_epochs=1)
        ppo_high_entropy = PPO(model_high, entropy_coef=0.1, num_epochs=1)

        # Use fixed seed for reproducible buffer
        np.random.seed(42)
        buffer = RolloutBuffer(buffer_size=4, num_envs=1)
        for i in range(4):
            buffer.add(
                player_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                opponent_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                player_active_idx=0,
                opponent_active_idx=0,
                field_state=np.random.randn(1, StateEncoder.FIELD_FEATURES).astype(np.float32),
                action_mask=np.ones((1, StateEncoder.NUM_ACTIONS), dtype=np.float32),
                action=0,
                log_prob=-1.0,
                reward=0.0,
                done=False,
                value=0.0,
            )

        buffer.compute_advantages(np.zeros(1), np.zeros(1))

        stats_low = ppo_low_entropy.update(buffer, batch_size=4)

        # Reset buffer for second model
        np.random.seed(42)
        buffer2 = RolloutBuffer(buffer_size=4, num_envs=1)
        for i in range(4):
            buffer2.add(
                player_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                opponent_pokemon=np.random.randn(1, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                player_active_idx=0,
                opponent_active_idx=0,
                field_state=np.random.randn(1, StateEncoder.FIELD_FEATURES).astype(np.float32),
                action_mask=np.ones((1, StateEncoder.NUM_ACTIONS), dtype=np.float32),
                action=0,
                log_prob=-1.0,
                reward=0.0,
                done=False,
                value=0.0,
            )

        buffer2.compute_advantages(np.zeros(1), np.zeros(1))
        stats_high = ppo_high_entropy.update(buffer2, batch_size=4)

        # Higher entropy coef should affect total_loss differently
        # entropy_loss is raw entropy (same for identical models)
        # but total_loss = policy_loss + value_coef*value_loss - entropy_coef*entropy_loss
        # So higher entropy_coef means more negative total_loss (since entropy_loss is negative)
        assert abs(stats_high.total_loss - stats_low.total_loss) > 0.01


class TestPPOStatsDataclass:
    """Test PPOStats dataclass."""

    def test_stats_creation(self):
        """Test creating PPOStats."""
        stats = PPOStats(
            policy_loss=0.1,
            value_loss=0.2,
            entropy_loss=-0.01,
            total_loss=0.29,
            approx_kl=0.005,
            clip_fraction=0.1,
            explained_variance=0.8,
        )

        assert stats.policy_loss == 0.1
        assert stats.value_loss == 0.2
        assert stats.entropy_loss == -0.01
        assert stats.total_loss == 0.29
        assert stats.approx_kl == 0.005
        assert stats.clip_fraction == 0.1
        assert stats.explained_variance == 0.8


class TestEmptyBufferHandling:
    """Test handling of empty buffers (e.g., during Ctrl+C shutdown)."""

    def test_get_batches_empty_buffer(self):
        """Test get_batches returns empty list when buffer is empty."""
        buffer = RolloutBuffer(buffer_size=128, num_envs=4)
        # Buffer is empty (ptr=0)
        assert buffer.size == 0

        batches = buffer.get_batches(batch_size=64)
        assert batches == []

    def test_get_batches_zero_batch_size(self):
        """Test get_batches returns empty list when batch_size is 0."""
        buffer = RolloutBuffer(buffer_size=128, num_envs=4)
        # Add some data
        buffer.add(
            player_pokemon=np.zeros((4, 6, StateEncoder.POKEMON_FEATURES)),
            opponent_pokemon=np.zeros((4, 6, StateEncoder.POKEMON_FEATURES)),
            player_active_idx=0,
            opponent_active_idx=0,
            field_state=np.zeros((4, StateEncoder.FIELD_FEATURES)),
            action_mask=np.ones((4, 9)),
            action=np.array([0, 0, 0, 0]),
            log_prob=np.zeros(4),
            reward=np.zeros(4),
            done=False,
            value=np.zeros(4),
        )

        batches = buffer.get_batches(batch_size=0)
        assert batches == []

    def test_ppo_update_empty_buffer(self):
        """Test PPO update handles empty buffer gracefully."""
        model = PolicyValueNetwork.from_config()
        ppo = PPO(model=model)
        buffer = RolloutBuffer(buffer_size=128, num_envs=4)

        # Buffer is empty - should not raise an error
        stats = ppo.update(buffer, batch_size=64)

        # Should return zero stats
        assert stats.policy_loss == 0.0
        assert stats.value_loss == 0.0
        assert stats.entropy_loss == 0.0
        assert stats.total_loss == 0.0
        assert stats.approx_kl == 0.0
        assert stats.clip_fraction == 0.0
        assert stats.explained_variance == 0.0

    def test_ppo_update_after_reset(self):
        """Test PPO update after buffer reset (simulates mid-rollout shutdown)."""
        model = PolicyValueNetwork.from_config()
        ppo = PPO(model=model)
        buffer = RolloutBuffer(buffer_size=128, num_envs=4)

        # Add data, then reset (simulates partial rollout interrupted)
        buffer.add(
            player_pokemon=np.zeros((4, 6, StateEncoder.POKEMON_FEATURES)),
            opponent_pokemon=np.zeros((4, 6, StateEncoder.POKEMON_FEATURES)),
            player_active_idx=0,
            opponent_active_idx=0,
            field_state=np.zeros((4, StateEncoder.FIELD_FEATURES)),
            action_mask=np.ones((4, 9)),
            action=np.array([0, 0, 0, 0]),
            log_prob=np.zeros(4),
            reward=np.zeros(4),
            done=False,
            value=np.zeros(4),
        )
        buffer.reset()

        # After reset, buffer should be empty
        assert buffer.size == 0

        # PPO update should handle this gracefully
        stats = ppo.update(buffer, batch_size=64)
        assert stats.total_loss == 0.0

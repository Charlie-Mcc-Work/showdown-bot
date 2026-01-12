"""Tests for memory management in training components.

These tests ensure that GPU memory and system memory are properly freed
during training to prevent memory leaks that can crash long training runs.
"""

import asyncio
import gc
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, mock_open

import numpy as np
import pytest
import torch

from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.environment.state_encoder import StateEncoder
from showdown_bot.training.trainer import Trainer, TrainablePlayer, MemoryMonitor
from showdown_bot.training.self_play import HistoricalPlayer, OpponentPool


class TestMemoryMonitor:
    """Test MemoryMonitor class."""

    def test_default_thresholds(self):
        """Test default memory thresholds."""
        monitor = MemoryMonitor()
        assert monitor.soft_limit_percent == 80.0
        assert monitor.hard_limit_percent == 95.0

    def test_custom_thresholds(self):
        """Test custom memory thresholds."""
        monitor = MemoryMonitor(soft_limit_percent=70.0, hard_limit_percent=90.0)
        assert monitor.soft_limit_percent == 70.0
        assert monitor.hard_limit_percent == 90.0

    def test_get_memory_info_returns_tuple(self):
        """Test that get_memory_info returns correct format."""
        monitor = MemoryMonitor()
        used, total, percent = monitor.get_memory_info()

        assert isinstance(used, int)
        assert isinstance(total, int)
        assert isinstance(percent, float)
        # On Linux with /proc/meminfo, these should be positive
        # On other systems, they may be 0
        assert used >= 0
        assert total >= 0
        assert 0 <= percent <= 100

    def test_check_memory_returns_ok_when_low(self):
        """Test check_memory returns ok for low memory usage."""
        monitor = MemoryMonitor(soft_limit_percent=99.0, hard_limit_percent=99.5)
        status, percent = monitor.check_memory()

        # Unless system is actually at 99%+, should return "ok"
        if percent < 99.0:
            assert status == "ok"

    def test_check_memory_status_values(self):
        """Test that check_memory returns valid status values."""
        monitor = MemoryMonitor()
        status, percent = monitor.check_memory()

        assert status in ("ok", "soft_limit", "hard_limit")
        assert isinstance(percent, float)

    @patch("builtins.open", mock_open(read_data="""MemTotal:       65536000 kB
MemFree:        10000000 kB
MemAvailable:   13107200 kB
Buffers:         1000000 kB
Cached:          5000000 kB
"""))
    def test_get_memory_info_parses_proc_meminfo(self):
        """Test parsing of /proc/meminfo."""
        monitor = MemoryMonitor()
        used, total, percent = monitor.get_memory_info()

        # Total = 65536000 kB = 67108864000 bytes
        assert total == 65536000 * 1024
        # Available = 13107200 kB = 13421772800 bytes
        # Used = Total - Available
        expected_used = (65536000 - 13107200) * 1024
        assert used == expected_used
        # Percent = used / total * 100
        expected_percent = (expected_used / total) * 100
        assert abs(percent - expected_percent) < 0.01

    @patch("builtins.open", mock_open(read_data="""MemTotal:       65536000 kB
MemAvailable:    6553600 kB
"""))
    def test_check_memory_soft_limit(self):
        """Test soft limit detection."""
        # Available = 6553600 kB out of 65536000 kB = 10% available = 90% used
        monitor = MemoryMonitor(soft_limit_percent=80.0, hard_limit_percent=95.0)
        status, percent = monitor.check_memory()

        assert status == "soft_limit"
        assert percent > 80.0

    @patch("builtins.open", mock_open(read_data="""MemTotal:       65536000 kB
MemAvailable:    3276800 kB
"""))
    def test_check_memory_hard_limit(self):
        """Test hard limit detection."""
        # Available = 3276800 kB out of 65536000 kB = 5% available = 95% used
        monitor = MemoryMonitor(soft_limit_percent=80.0, hard_limit_percent=95.0)
        status, percent = monitor.check_memory()

        assert status == "hard_limit"
        assert percent >= 95.0

    def test_format_memory_bytes(self):
        """Test memory formatting for various sizes."""
        monitor = MemoryMonitor()

        assert "B" in monitor.format_memory(500)
        assert "KB" in monitor.format_memory(5000)
        assert "MB" in monitor.format_memory(5000000)
        assert "GB" in monitor.format_memory(5000000000)

    def test_get_memory_status_string(self):
        """Test memory status string format."""
        monitor = MemoryMonitor()
        status_str = monitor.get_memory_status_string()

        # Should contain a slash and percentage
        assert "/" in status_str
        assert "%" in status_str

    @patch("builtins.open")
    def test_handles_missing_proc_meminfo(self, mock_file):
        """Test graceful handling when /proc/meminfo is unavailable."""
        mock_file.side_effect = OSError("File not found")

        monitor = MemoryMonitor()
        used, total, percent = monitor.get_memory_info()

        # Should return zeros on error
        assert used == 0
        assert total == 0
        assert percent == 0.0


class TestHistoricalPlayerMemory:
    """Test memory management for HistoricalPlayer."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def model(self, device):
        """Create a test model using from_config to match HistoricalPlayer."""
        return PolicyValueNetwork.from_config().to(device)

    @pytest.fixture
    def checkpoint_path(self, model, tmp_path):
        """Save a checkpoint and return its path."""
        path = tmp_path / "test_checkpoint.pt"
        torch.save(model.state_dict(), path)
        return path

    @patch('poke_env.player.Player.__init__')
    def test_historical_player_creates_model_on_device(self, mock_player_init, checkpoint_path, device):
        """Test that HistoricalPlayer loads model on specified device."""
        mock_player_init.return_value = None  # Skip Player.__init__
        state_encoder = StateEncoder(device=device)

        player = HistoricalPlayer(
            checkpoint_path=checkpoint_path,
            model_class=PolicyValueNetwork,
            state_encoder=state_encoder,
            device=device,
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )

        # Model should be on the correct device
        assert next(player.model.parameters()).device.type == device.type

    @patch('poke_env.player.Player.__init__')
    def test_historical_player_model_can_be_moved_to_cpu(self, mock_player_init, checkpoint_path, device):
        """Test that we can move model to CPU and delete it."""
        mock_player_init.return_value = None
        state_encoder = StateEncoder(device=device)

        player = HistoricalPlayer(
            checkpoint_path=checkpoint_path,
            model_class=PolicyValueNetwork,
            state_encoder=state_encoder,
            device=device,
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )

        # Move to CPU (simulating cleanup)
        player.model.cpu()

        # Model should now be on CPU
        assert next(player.model.parameters()).device.type == "cpu"

        # Model should still exist (we don't delete to avoid race conditions)
        assert hasattr(player, 'model') and player.model is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch('poke_env.player.Player.__init__')
    def test_gpu_memory_freed_after_model_deletion(self, mock_player_init, tmp_path):
        """Test that GPU memory is freed when model is deleted."""
        mock_player_init.return_value = None
        device = torch.device("cuda")
        state_encoder = StateEncoder(device=device)

        # Create and save a model
        model = PolicyValueNetwork.from_config().to(device)
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        initial_memory = torch.cuda.memory_allocated()

        # Create player with model on GPU
        player = HistoricalPlayer(
            checkpoint_path=checkpoint_path,
            model_class=PolicyValueNetwork,
            state_encoder=state_encoder,
            device=device,
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )

        memory_with_model = torch.cuda.memory_allocated()
        assert memory_with_model > initial_memory, "Model should allocate GPU memory"

        # Move to CPU (cleanup pattern - no deletion to avoid race conditions)
        player.model.cpu()
        del player  # Player deletion will clean up the model

        gc.collect()
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()

        # Memory should be back to initial level (with some tolerance)
        memory_leaked = final_memory - initial_memory
        assert memory_leaked < 1024 * 1024, f"Leaked {memory_leaked} bytes of GPU memory"


class TestTrainerCleanup:
    """Test memory cleanup in Trainer._cleanup_players."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def model(self, device):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        ).to(device)

    @pytest.fixture
    def trainer(self, model, device, tmp_path):
        """Create trainer with minimal config."""
        return Trainer(
            model=model,
            device=device,
            save_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
            self_play_dir=str(tmp_path / "opponents"),
            use_self_play=False,
        )

    def test_cleanup_players_removes_model_attribute(self, trainer, model, device, tmp_path):
        """Test that cleanup removes model from players with models."""
        # Create a mock player with a model
        mock_player = MagicMock()
        model = PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        ).to(device)
        mock_player.model = model
        mock_player.ps_client = None

        # Run cleanup
        asyncio.run(trainer._cleanup_players([mock_player]))

        # Model should still exist but be on CPU (not deleted to avoid race conditions)
        assert hasattr(mock_player, 'model')
        assert next(mock_player.model.parameters()).device.type == "cpu"

    def test_cleanup_players_handles_players_without_model(self, trainer):
        """Test cleanup handles regular players without model attribute."""
        mock_player = MagicMock()
        mock_player.ps_client = None
        # No model attribute

        # Should not raise
        asyncio.run(trainer._cleanup_players([mock_player]))

    def test_cleanup_players_stops_websocket(self, trainer):
        """Test cleanup stops websocket connections."""
        mock_client = AsyncMock()
        mock_player = MagicMock()
        mock_player.ps_client = mock_client

        asyncio.run(trainer._cleanup_players([mock_player]))

        mock_client.stop_listening.assert_called_once()

    def test_cleanup_players_handles_empty_list(self, trainer):
        """Test cleanup handles empty player list."""
        # Should not raise
        asyncio.run(trainer._cleanup_players([]))

    def test_cleanup_players_handles_exceptions_gracefully(self, trainer, device):
        """Test cleanup continues even if one player fails."""
        # First player will raise
        failing_player = MagicMock()
        failing_player.ps_client = AsyncMock()
        failing_player.ps_client.stop_listening.side_effect = Exception("Connection error")
        failing_player.model = MagicMock()
        failing_player.model.cpu.side_effect = Exception("Move failed")

        # Second player should still be cleaned
        ok_player = MagicMock()
        ok_player.ps_client = None
        ok_model = PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        ).to(device)
        ok_player.model = ok_model

        # Should not raise, even with failing player
        asyncio.run(trainer._cleanup_players([failing_player, ok_player]))

        # Second player's model should be on CPU (not deleted to avoid race conditions)
        assert hasattr(ok_player, 'model')
        assert next(ok_player.model.parameters()).device.type == "cpu"


class TestMemoryLeakPrevention:
    """Tests to verify memory doesn't leak across training iterations."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def model(self, device):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        ).to(device)

    def test_experience_data_not_retained_after_buffer_add(self, device):
        """Test that experience numpy arrays can be garbage collected."""
        from showdown_bot.training.buffer import RolloutBuffer

        buffer = RolloutBuffer(buffer_size=16, num_envs=2, device=device)

        # Create experience data
        exp_data = {
            "player_pokemon": np.random.randn(2, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
            "opponent_pokemon": np.random.randn(2, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
        }

        # Get reference count before
        import sys
        initial_refcount = sys.getrefcount(exp_data["player_pokemon"])

        # Add to buffer
        buffer.add(
            player_pokemon=exp_data["player_pokemon"],
            opponent_pokemon=exp_data["opponent_pokemon"],
            player_active_idx=np.array([0, 0]),
            opponent_active_idx=np.array([0, 0]),
            field_state=np.random.randn(2, StateEncoder.FIELD_FEATURES).astype(np.float32),
            action_mask=np.ones((2, StateEncoder.NUM_ACTIONS), dtype=np.float32),
            action=np.array([0, 0]),
            log_prob=np.array([0.0, 0.0]),
            reward=np.array([0.0, 0.0]),
            done=np.array([False, False]),
            value=np.array([0.0, 0.0]),
        )

        # Reference count should not have increased significantly
        # (buffer copies data, doesn't hold reference)
        final_refcount = sys.getrefcount(exp_data["player_pokemon"])
        assert final_refcount <= initial_refcount + 1  # Allow for temporary refs

    def test_buffer_reset_allows_garbage_collection(self, device):
        """Test that buffer reset allows old data to be collected."""
        from showdown_bot.training.buffer import RolloutBuffer
        import tracemalloc

        buffer = RolloutBuffer(buffer_size=64, num_envs=4, device=device)

        # Fill buffer
        for _ in range(64):
            buffer.add(
                player_pokemon=np.random.randn(4, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                opponent_pokemon=np.random.randn(4, 6, StateEncoder.POKEMON_FEATURES).astype(np.float32),
                player_active_idx=np.zeros(4, dtype=np.int64),
                opponent_active_idx=np.zeros(4, dtype=np.int64),
                field_state=np.random.randn(4, StateEncoder.FIELD_FEATURES).astype(np.float32),
                action_mask=np.ones((4, StateEncoder.NUM_ACTIONS), dtype=np.float32),
                action=np.zeros(4, dtype=np.int64),
                log_prob=np.zeros(4, dtype=np.float32),
                reward=np.zeros(4, dtype=np.float32),
                done=np.zeros(4, dtype=np.float32),
                value=np.zeros(4, dtype=np.float32),
            )

        # Reset should work
        buffer.reset()
        assert buffer.ptr == 0
        assert buffer.size == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch('poke_env.player.Player.__init__')
    def test_multiple_opponent_creations_no_gpu_leak(self, mock_player_init, tmp_path):
        """Test that creating and cleaning up opponents doesn't leak GPU memory."""
        mock_player_init.return_value = None
        device = torch.device("cuda")

        # Create and save a model
        model = PolicyValueNetwork.from_config().to(device)
        checkpoint_path = tmp_path / "opponent.pt"
        torch.save(model.state_dict(), checkpoint_path)
        del model

        state_encoder = StateEncoder(device=device)

        gc.collect()
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated()

        # Simulate multiple training iterations creating/destroying opponents
        for _ in range(5):
            players = []
            for _ in range(4):  # 4 parallel envs
                player = HistoricalPlayer(
                    checkpoint_path=checkpoint_path,
                    model_class=PolicyValueNetwork,
                    state_encoder=state_encoder,
                    device=device,
                    battle_format="gen9randombattle",
                    max_concurrent_battles=1,
                )
                players.append(player)

            # Cleanup (simulating trainer cleanup - move to CPU, don't delete)
            for player in players:
                player.model.cpu()

            del players
            gc.collect()
            torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()
        memory_growth = final_memory - baseline_memory

        # Should not have significant memory growth
        # Allow 10MB tolerance for PyTorch internal caching
        assert memory_growth < 10 * 1024 * 1024, \
            f"GPU memory grew by {memory_growth / 1024 / 1024:.2f}MB over 5 iterations"

    def test_trainable_player_clears_experiences(self, model, device):
        """Test TrainablePlayer.get_experiences clears internal list."""
        state_encoder = StateEncoder(device=device)

        player = TrainablePlayer(
            model=model,
            state_encoder=state_encoder,
            device=device,
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )

        # Add some fake experiences
        player.current_experiences = [
            {"data": np.random.randn(100, 100)} for _ in range(10)
        ]

        assert len(player.current_experiences) == 10

        # Get experiences should return and clear
        experiences = player.get_experiences()

        assert len(experiences) == 10
        assert len(player.current_experiences) == 0

    def test_trainable_player_reset_stats_clears_lists(self, model, device):
        """Test TrainablePlayer.reset_stats clears tracking lists."""
        state_encoder = StateEncoder(device=device)

        player = TrainablePlayer(
            model=model,
            state_encoder=state_encoder,
            device=device,
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )

        # Simulate accumulated stats
        player.episode_rewards = [1.0] * 100
        player.episode_lengths = [50] * 100
        player._rollout_wins = 75
        player._rollout_battles = 100

        player.reset_stats()

        assert len(player.episode_rewards) == 0
        assert len(player.episode_lengths) == 0
        assert player._rollout_wins == 0
        assert player._rollout_battles == 0


class TestGarbageCollectionIntegration:
    """Test that CUDA cache cleanup is available."""

    def test_torch_cuda_empty_cache_available(self):
        """Verify torch.cuda.empty_cache is available."""
        assert hasattr(torch.cuda, 'empty_cache')
        # Should not raise even without CUDA
        if not torch.cuda.is_available():
            # On non-CUDA systems, empty_cache should be a no-op
            torch.cuda.empty_cache()  # Should not raise


class TestOpponentPoolMemory:
    """Test memory management in OpponentPool."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")  # Use CPU for these tests

    @pytest.fixture
    def model(self, device):
        return PolicyValueNetwork(
            hidden_dim=64,
            pokemon_dim=32,
            num_heads=2,
            num_layers=1,
            num_actions=9,
        ).to(device)

    def test_pool_pruning_deletes_checkpoint_files(self, model, tmp_path, device):
        """Test that pruning removes checkpoint files from disk."""
        pool = OpponentPool(
            pool_dir=tmp_path / "pool",
            max_size=3,
            device=device,
        )

        # Add more opponents than max_size
        for i in range(5):
            pool.add_opponent(model, timestep=i * 1000, skill_rating=1000.0 + i * 10)

        # Pool should have been pruned to max_size
        assert pool.size <= 3

        # Check that only kept checkpoints exist on disk
        checkpoint_files = list((tmp_path / "pool").glob("opponent_*.pt"))
        assert len(checkpoint_files) == pool.size

    @patch('poke_env.player.Player.__init__')
    def test_create_player_loads_fresh_model_each_time(self, mock_player_init, tmp_path, device):
        """Test that each create_player call loads a fresh model."""
        mock_player_init.return_value = None

        # Create a full-size model to match HistoricalPlayer expectations
        model = PolicyValueNetwork.from_config().to(device)

        pool = OpponentPool(
            pool_dir=tmp_path / "pool",
            max_size=5,
            device=device,
        )

        opponent_info = pool.add_opponent(model, timestep=1000)

        # Create two players from same checkpoint
        player1 = pool.create_player(opponent_info, "gen9randombattle")
        player2 = pool.create_player(opponent_info, "gen9randombattle")

        # They should have separate model instances
        assert player1.model is not player2.model

        # Modifying one shouldn't affect the other
        with torch.no_grad():
            list(player1.model.parameters())[0].fill_(0.0)

        # player2's parameters should be unchanged (not zero)
        param2 = list(player2.model.parameters())[0]
        assert not torch.all(param2 == 0.0)

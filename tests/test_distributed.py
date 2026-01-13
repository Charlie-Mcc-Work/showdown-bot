"""Tests for distributed training functionality."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_distributed import (
    DistributedPPO,
    setup_distributed,
    cleanup_distributed,
    find_latest_checkpoint,
)
from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.training.buffer import RolloutBuffer


class TestDistributedPPOInit:
    """Tests for DistributedPPO initialization."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        return PolicyValueNetwork.from_config()

    def test_default_world_size(self, model):
        """Test default world size is 1."""
        ppo = DistributedPPO(model=model)
        assert ppo.world_size == 1

    def test_custom_world_size(self, model):
        """Test custom world size is set correctly."""
        ppo = DistributedPPO(model=model, world_size=4)
        assert ppo.world_size == 4

    def test_inherits_from_ppo(self, model):
        """Test DistributedPPO inherits from PPO."""
        from showdown_bot.training.ppo import PPO

        ppo = DistributedPPO(model=model, world_size=2)
        assert isinstance(ppo, PPO)


class TestDistributedPPOGradientSync:
    """Tests for gradient synchronization."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        return PolicyValueNetwork.from_config()

    def test_sync_gradients_skips_when_world_size_1(self, model):
        """Test gradient sync is skipped with single worker."""
        ppo = DistributedPPO(model=model, world_size=1)

        # Set some gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param)

        original_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.all_reduce") as mock_reduce:
                ppo._sync_gradients()
                mock_reduce.assert_not_called()

        # Gradients should be unchanged
        for orig, param in zip(original_grads, model.parameters()):
            if param.grad is not None:
                assert torch.equal(orig, param.grad)

    def test_sync_gradients_skips_when_not_initialized(self, model):
        """Test gradient sync is skipped when dist not initialized."""
        ppo = DistributedPPO(model=model, world_size=4)

        with patch("torch.distributed.is_initialized", return_value=False):
            with patch("torch.distributed.all_reduce") as mock_reduce:
                ppo._sync_gradients()
                mock_reduce.assert_not_called()

    def test_sync_gradients_calls_all_reduce(self, model):
        """Test gradient sync calls all_reduce for each parameter."""
        ppo = DistributedPPO(model=model, world_size=4)

        # Set gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param)

        num_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.all_reduce") as mock_reduce:
                ppo._sync_gradients()
                assert mock_reduce.call_count == num_params_with_grad

    def test_sync_gradients_divides_by_world_size(self, model):
        """Test gradients are divided by world size after all_reduce."""
        world_size = 4
        ppo = DistributedPPO(model=model, world_size=world_size)

        # Set known gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param) * world_size

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.all_reduce"):  # Mock to avoid actual sync
                ppo._sync_gradients()

        # Gradients should be divided by world_size
        for param in model.parameters():
            if param.grad is not None:
                # After division by 4, values should be 1.0
                assert torch.allclose(param.grad, torch.ones_like(param.grad))

    def test_sync_gradients_handles_none_gradients(self, model):
        """Test gradient sync handles parameters without gradients."""
        ppo = DistributedPPO(model=model, world_size=4)

        # Clear all gradients
        for param in model.parameters():
            param.grad = None

        # Should not raise
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.all_reduce") as mock_reduce:
                ppo._sync_gradients()
                mock_reduce.assert_not_called()


class TestDistributedPPOUpdate:
    """Tests for the distributed update method."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        return PolicyValueNetwork.from_config()

    def test_update_with_empty_buffer(self, model):
        """Test update with empty buffer returns zero stats."""
        ppo = DistributedPPO(model=model, world_size=1)
        buffer = RolloutBuffer(buffer_size=64, num_envs=1)

        stats = ppo.update(buffer, batch_size=32)
        assert stats.policy_loss == 0.0
        assert stats.value_loss == 0.0
        assert stats.total_loss == 0.0


class TestSetupDistributed:
    """Tests for distributed setup utilities."""

    def test_setup_without_env_vars(self):
        """Test setup returns single-process mode without env vars."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any distributed env vars
            os.environ.pop("RANK", None)
            os.environ.pop("WORLD_SIZE", None)
            os.environ.pop("LOCAL_RANK", None)

            rank, world_size, is_distributed = setup_distributed()

            assert rank == 0
            assert world_size == 1
            assert is_distributed is False

    def test_setup_with_env_vars_gloo_backend(self):
        """Test setup uses gloo backend when single GPU."""
        with patch.dict(os.environ, {
            "RANK": "0",
            "WORLD_SIZE": "4",
            "LOCAL_RANK": "0",
        }):
            with patch("torch.cuda.device_count", return_value=1):
                with patch("torch.cuda.is_available", return_value=True):
                    with patch("torch.cuda.set_device"):
                        with patch("torch.distributed.init_process_group") as mock_init:
                            rank, world_size, is_distributed = setup_distributed()

                            assert rank == 0
                            assert world_size == 4
                            assert is_distributed is True
                            mock_init.assert_called_once_with(backend="gloo")


class TestFindLatestCheckpoint:
    """Tests for checkpoint finding utility."""

    def test_returns_none_for_missing_dir(self, tmp_path):
        """Test returns None if directory doesn't exist."""
        result = find_latest_checkpoint(str(tmp_path / "nonexistent"))
        assert result is None

    def test_returns_best_model_first(self, tmp_path):
        """Test returns best_model.pt if it exists."""
        best = tmp_path / "best_model.pt"
        best.touch()
        latest = tmp_path / "latest.pt"
        latest.touch()

        result = find_latest_checkpoint(str(tmp_path))
        assert result == best

    def test_returns_latest_if_no_best(self, tmp_path):
        """Test returns latest.pt if best_model.pt doesn't exist."""
        latest = tmp_path / "latest.pt"
        latest.touch()

        result = find_latest_checkpoint(str(tmp_path))
        assert result == latest

    def test_returns_highest_numbered_checkpoint(self, tmp_path):
        """Test returns checkpoint with highest timestep."""
        (tmp_path / "checkpoint_100000.pt").touch()
        (tmp_path / "checkpoint_500000.pt").touch()
        (tmp_path / "checkpoint_300000.pt").touch()

        result = find_latest_checkpoint(str(tmp_path))
        assert result == tmp_path / "checkpoint_500000.pt"

    def test_returns_none_for_empty_dir(self, tmp_path):
        """Test returns None for empty directory."""
        result = find_latest_checkpoint(str(tmp_path))
        assert result is None


class TestCleanupDistributed:
    """Tests for distributed cleanup."""

    def test_cleanup_destroys_process_group(self):
        """Test cleanup destroys process group when initialized."""
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.destroy_process_group") as mock_destroy:
                cleanup_distributed()
                mock_destroy.assert_called_once()

    def test_cleanup_skips_when_not_initialized(self):
        """Test cleanup skips when not initialized."""
        with patch("torch.distributed.is_initialized", return_value=False):
            with patch("torch.distributed.destroy_process_group") as mock_destroy:
                cleanup_distributed()
                mock_destroy.assert_not_called()

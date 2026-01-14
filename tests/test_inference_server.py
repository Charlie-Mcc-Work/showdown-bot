"""Tests for the batched inference server."""

import threading
import time

import pytest
import torch

from showdown_bot.environment.state_encoder import EncodedState, StateEncoder
from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.training.inference_server import BatchedInferenceServer


@pytest.fixture
def device():
    """Get test device (CPU for tests)."""
    return torch.device("cpu")


@pytest.fixture
def model(device):
    """Create a test model."""
    model = PolicyValueNetwork(
        hidden_dim=64,
        pokemon_dim=32,
        num_heads=2,
        num_layers=1,
        num_actions=9,
    )
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def inference_server(model, device):
    """Create a test inference server."""
    server = BatchedInferenceServer(
        model=model,
        device=device,
        max_batch_size=4,
        max_wait_ms=10.0,
    )
    server.start()
    yield server
    server.stop()


def create_dummy_state(device: torch.device) -> EncodedState:
    """Create a dummy encoded state for testing."""
    return EncodedState(
        player_pokemon=torch.randn(6, StateEncoder.POKEMON_FEATURES),
        opponent_pokemon=torch.randn(6, StateEncoder.POKEMON_FEATURES),
        player_active_idx=0,
        opponent_active_idx=0,
        field_state=torch.randn(StateEncoder.FIELD_FEATURES),
        action_mask=torch.ones(9),  # All actions valid
    )


def test_server_start_stop(model, device):
    """Test that server can start and stop cleanly."""
    server = BatchedInferenceServer(model=model, device=device)

    # Start server
    server.start()
    assert server._running
    assert server._thread is not None
    assert server._thread.is_alive()

    # Stop server
    server.stop()
    assert not server._running


def test_single_inference(inference_server, device):
    """Test a single inference request."""
    state = create_dummy_state(device)

    result = inference_server.infer(state, timeout=5.0)

    assert result.action in range(9)  # Valid action index
    assert isinstance(result.log_prob, float)
    assert isinstance(result.value, float)


def test_multiple_sequential_inferences(inference_server, device):
    """Test multiple sequential inference requests."""
    results = []
    for _ in range(5):
        state = create_dummy_state(device)
        result = inference_server.infer(state, timeout=5.0)
        results.append(result)

    assert len(results) == 5
    for result in results:
        assert result.action in range(9)


def test_concurrent_inferences(inference_server, device):
    """Test concurrent inference requests from multiple threads."""
    results = []
    errors = []

    def worker():
        try:
            state = create_dummy_state(device)
            result = inference_server.infer(state, timeout=5.0)
            results.append(result)
        except Exception as e:
            errors.append(e)

    # Start 4 threads concurrently
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert len(errors) == 0, f"Errors: {errors}"
    assert len(results) == 4


def test_batching_occurs(model, device):
    """Test that requests are actually batched together."""
    server = BatchedInferenceServer(
        model=model,
        device=device,
        max_batch_size=8,
        max_wait_ms=50.0,  # Longer wait to collect batch
    )
    server.start()

    try:
        results = []
        errors = []

        def worker():
            try:
                state = create_dummy_state(device)
                result = server.infer(state, timeout=5.0)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start 4 threads nearly simultaneously
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 4

        # Check that batching occurred (should have fewer batches than requests)
        stats = server.get_stats()
        assert stats["total_requests"] == 4
        # With concurrent requests and 50ms wait, should batch at least 2 together
        assert stats["avg_batch_size"] >= 1.0

    finally:
        server.stop()


def test_stats_tracking(inference_server, device):
    """Test that statistics are tracked correctly."""
    # Initial stats should be zero
    stats = inference_server.get_stats()
    assert stats["total_requests"] == 0
    assert stats["total_batches"] == 0

    # Make some requests
    for _ in range(3):
        state = create_dummy_state(device)
        inference_server.infer(state, timeout=5.0)

    # Stats should be updated
    stats = inference_server.get_stats()
    assert stats["total_requests"] == 3
    assert stats["total_batches"] > 0
    assert stats["avg_batch_size"] > 0


def test_stats_reset(inference_server, device):
    """Test that stats can be reset."""
    # Make some requests
    state = create_dummy_state(device)
    inference_server.infer(state, timeout=5.0)

    # Verify stats are non-zero
    stats = inference_server.get_stats()
    assert stats["total_requests"] > 0

    # Reset stats
    inference_server.reset_stats()

    # Stats should be zero again
    stats = inference_server.get_stats()
    assert stats["total_requests"] == 0
    assert stats["total_batches"] == 0


def test_action_mask_respected(inference_server, device):
    """Test that action mask is respected in inference."""
    # Create state with only action 0 valid
    state = EncodedState(
        player_pokemon=torch.randn(6, StateEncoder.POKEMON_FEATURES),
        opponent_pokemon=torch.randn(6, StateEncoder.POKEMON_FEATURES),
        player_active_idx=0,
        opponent_active_idx=0,
        field_state=torch.randn(StateEncoder.FIELD_FEATURES),
        action_mask=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    result = inference_server.infer(state, timeout=5.0)

    # Only action 0 should be selected (it's the only valid one)
    assert result.action == 0


def test_server_handles_rapid_requests(inference_server, device):
    """Test server handles rapid fire requests."""
    results = []

    # Send 20 requests rapidly
    for _ in range(20):
        state = create_dummy_state(device)
        result = inference_server.infer(state, timeout=5.0)
        results.append(result)

    assert len(results) == 20

    stats = inference_server.get_stats()
    assert stats["total_requests"] == 20

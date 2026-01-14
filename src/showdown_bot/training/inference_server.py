"""Batched inference server for efficient GPU utilization during rollout collection.

Instead of each player doing individual forward passes with batch size 1,
this server batches multiple inference requests together to maximize GPU throughput.

Uses threading to work with poke-env's synchronous choose_move() method.
"""

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import torch
from torch.amp import autocast

from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.environment.state_encoder import EncodedState


@dataclass
class InferenceRequest:
    """A single inference request from a player."""

    state: EncodedState
    future: Future
    timestamp: float


@dataclass
class InferenceResult:
    """Result of a single inference request."""

    action: int
    log_prob: float
    value: float


class BatchedInferenceServer:
    """Server that batches inference requests for efficient GPU utilization.

    Players submit inference requests which are collected and processed
    together in batches. This is more efficient than individual forward
    passes because:
    1. GPU kernels have better utilization with larger batches
    2. Memory transfers are amortized across the batch
    3. Better cache locality for model weights

    Uses a separate thread for batch processing to work with poke-env's
    synchronous choose_move() method. The main thread can block on futures
    while the processor thread handles batching.
    """

    def __init__(
        self,
        model: PolicyValueNetwork,
        device: torch.device,
        max_batch_size: int = 32,
        max_wait_ms: float = 5.0,
    ):
        """Initialize the inference server.

        Args:
            model: The policy network for inference
            device: Device to run inference on
            max_batch_size: Maximum requests to batch together
            max_wait_ms: Maximum time to wait for batch to fill (milliseconds)
        """
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        # Thread-safe request queue
        self._request_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

        # Lock for thread-safe stats updates
        self._stats_lock = threading.Lock()

        # Statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_size = 0

        # Mixed precision - disabled for small models
        self.use_amp = False

    def start(self) -> None:
        """Start the inference server."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the inference server."""
        self._running = False
        if self._thread:
            # Put a sentinel to wake up the thread if it's waiting
            self._request_queue.put(None)  # type: ignore
            self._thread.join(timeout=5.0)
            self._thread = None

    def infer(self, state: EncodedState, timeout: float = 30.0) -> InferenceResult:
        """Submit a state for inference and wait for the result.

        This is a blocking call that can be used from synchronous code
        (like poke-env's choose_move method).

        Args:
            state: The encoded battle state
            timeout: Maximum time to wait for result (seconds)

        Returns:
            InferenceResult with action, log_prob, and value
        """
        # Create a threading Future for this request
        future: Future[InferenceResult] = Future()

        request = InferenceRequest(
            state=state,
            future=future,
            timestamp=time.monotonic(),
        )

        self._request_queue.put(request)
        return future.result(timeout=timeout)

    def _process_loop(self) -> None:
        """Main processing loop that batches and processes requests.

        Runs in a separate thread.
        """
        while self._running:
            try:
                # Collect requests for batching
                requests = self._collect_batch()

                if not requests:
                    continue

                # Process the batch
                results = self._process_batch(requests)

                # Deliver results to waiting futures
                for request, result in zip(requests, results):
                    if not request.future.done():
                        request.future.set_result(result)

            except Exception as e:
                # Log error but continue processing
                print(f"Inference server error: {e}")
                continue

    def _collect_batch(self) -> list[InferenceRequest]:
        """Collect requests into a batch.

        Waits up to max_wait_ms for the batch to fill, or returns early
        if max_batch_size is reached.
        """
        requests: list[InferenceRequest] = []
        deadline = time.monotonic() + (self.max_wait_ms / 1000.0)

        # Wait for at least one request
        try:
            first_request = self._request_queue.get(timeout=0.1)
            # Check for sentinel (None) indicating shutdown
            if first_request is None:
                return []
            requests.append(first_request)
        except queue.Empty:
            return []

        # Collect more requests until batch is full or deadline reached
        while len(requests) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            try:
                request = self._request_queue.get(timeout=remaining)
                # Check for sentinel
                if request is None:
                    break
                requests.append(request)
            except queue.Empty:
                break

        return requests

    def _process_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        """Process a batch of requests through the model.

        Args:
            requests: List of inference requests to process

        Returns:
            List of results, one per request
        """
        batch_size = len(requests)

        # Update stats thread-safely
        with self._stats_lock:
            self.total_requests += batch_size
            self.total_batches += 1
            self.total_batch_size += batch_size

        # Stack all states into batch tensors
        player_pokemon = torch.stack([r.state.player_pokemon for r in requests]).to(self.device)
        opponent_pokemon = torch.stack([r.state.opponent_pokemon for r in requests]).to(self.device)
        player_active_idx = torch.tensor(
            [r.state.player_active_idx for r in requests],
            dtype=torch.long,
            device=self.device
        )
        opponent_active_idx = torch.tensor(
            [r.state.opponent_active_idx for r in requests],
            dtype=torch.long,
            device=self.device
        )
        field_state = torch.stack([r.state.field_state for r in requests]).to(self.device)
        action_mask = torch.stack([r.state.action_mask for r in requests]).to(self.device)

        # Run batched inference
        with torch.no_grad(), autocast("cuda", enabled=self.use_amp):
            actions, log_probs, _, values = self.model.get_action_and_value(
                player_pokemon,
                opponent_pokemon,
                player_active_idx,
                opponent_active_idx,
                field_state,
                action_mask,
            )

        # Convert to results
        results = []
        for i in range(batch_size):
            results.append(InferenceResult(
                action=actions[i].item(),
                log_prob=log_probs[i].item(),
                value=values[i].item(),
            ))

        return results

    def get_stats(self) -> dict[str, float]:
        """Get server statistics."""
        with self._stats_lock:
            avg_batch_size = (
                self.total_batch_size / self.total_batches
                if self.total_batches > 0
                else 0.0
            )
            return {
                "total_requests": self.total_requests,
                "total_batches": self.total_batches,
                "avg_batch_size": avg_batch_size,
            }

    def reset_stats(self) -> None:
        """Reset server statistics."""
        with self._stats_lock:
            self.total_requests = 0
            self.total_batches = 0
            self.total_batch_size = 0

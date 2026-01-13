"""Rollout buffer for storing experiences during training."""

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout experiences for PPO training.

    Stores transitions from multiple parallel environments and computes
    advantages using Generalized Advantage Estimation (GAE).

    Episode boundaries are tracked so that GAE is computed correctly
    even when experiences from different environments are interleaved.
    """

    buffer_size: int
    num_envs: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # Storage - initialized in __post_init__
    player_pokemon: np.ndarray = field(init=False)
    opponent_pokemon: np.ndarray = field(init=False)
    player_active_idx: np.ndarray = field(init=False)
    opponent_active_idx: np.ndarray = field(init=False)
    field_state: np.ndarray = field(init=False)
    action_mask: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    log_probs: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    advantages: np.ndarray = field(init=False)
    returns: np.ndarray = field(init=False)

    ptr: int = field(init=False, default=0)
    path_start_idx: int = field(init=False, default=0)
    full: bool = field(init=False, default=False)

    # Episode boundary tracking for correct GAE computation
    episode_ends: list = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Initialize storage arrays."""
        # Import here to avoid circular imports
        from showdown_bot.environment.state_encoder import StateEncoder

        # State components
        self.player_pokemon = np.zeros(
            (self.buffer_size, self.num_envs, 6, StateEncoder.POKEMON_FEATURES),
            dtype=np.float32,
        )
        self.opponent_pokemon = np.zeros(
            (self.buffer_size, self.num_envs, 6, StateEncoder.POKEMON_FEATURES),
            dtype=np.float32,
        )
        self.player_active_idx = np.zeros(
            (self.buffer_size, self.num_envs), dtype=np.int64
        )
        self.opponent_active_idx = np.zeros(
            (self.buffer_size, self.num_envs), dtype=np.int64
        )
        self.field_state = np.zeros(
            (self.buffer_size, self.num_envs, StateEncoder.FIELD_FEATURES),
            dtype=np.float32,
        )
        self.action_mask = np.zeros(
            (self.buffer_size, self.num_envs, StateEncoder.NUM_ACTIONS),
            dtype=np.float32,
        )

        # Actions and values
        self.actions = np.zeros((self.buffer_size, self.num_envs), dtype=np.int64)
        self.log_probs = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)

        # Computed during finalization
        self.advantages = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)

    def add(
        self,
        player_pokemon: np.ndarray,
        opponent_pokemon: np.ndarray,
        player_active_idx: int | np.ndarray,
        opponent_active_idx: int | np.ndarray,
        field_state: np.ndarray,
        action_mask: np.ndarray,
        action: int | np.ndarray,
        log_prob: float | np.ndarray,
        reward: float | np.ndarray,
        done: bool | np.ndarray,
        value: float | np.ndarray,
    ) -> None:
        """Add a transition to the buffer.

        Tracks episode boundaries when done=True for correct GAE computation.
        """
        self.player_pokemon[self.ptr] = player_pokemon
        self.opponent_pokemon[self.ptr] = opponent_pokemon
        self.player_active_idx[self.ptr] = player_active_idx
        self.opponent_active_idx[self.ptr] = opponent_active_idx
        self.field_state[self.ptr] = field_state
        self.action_mask[self.ptr] = action_mask
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value

        # Track episode boundaries - done flag marks end of episode
        done_val = done[0] if isinstance(done, np.ndarray) else done
        if done_val:
            self.episode_ends.append(self.ptr)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_advantages(self, last_value: np.ndarray, last_done: np.ndarray) -> None:
        """Compute GAE advantages and returns.

        Processes each episode separately to avoid bleeding advantages across
        episode boundaries when experiences from multiple environments are
        interleaved in the buffer.

        Args:
            last_value: Value estimate for the state after the last stored transition
            last_done: Whether the last state was terminal
        """
        if self.ptr == 0:
            return

        # Build episode segments: [(start, end), ...]
        # Each segment is a contiguous episode that ends with done=True
        # or is an incomplete episode at the end of the buffer
        segments = []
        start = 0
        for end_idx in self.episode_ends:
            # end_idx is the index where done=True was stored (before ptr increment)
            # So the episode is [start, end_idx] inclusive
            segments.append((start, end_idx + 1))  # +1 to make it exclusive end
            start = end_idx + 1

        # Add final segment if there's data after the last episode end
        if start < self.ptr:
            segments.append((start, self.ptr))

        # If no segments were created (no episode_ends), treat entire buffer as one segment
        if not segments:
            segments = [(0, self.ptr)]

        # Process each segment independently
        for seg_start, seg_end in segments:
            # Check if the last step of this segment has done=True
            last_step = seg_end - 1
            segment_is_terminal = bool(self.dones[last_step, 0] > 0.5)

            # Bootstrap value for this segment
            if segment_is_terminal:
                # Episode ended with done=True - bootstrap from 0
                segment_last_value = np.zeros((1,), dtype=np.float32)
                segment_last_done = np.ones((1,), dtype=np.float32)
            elif seg_end == self.ptr:
                # Final incomplete segment - episode still ongoing
                # Use provided last_value for bootstrapping, but mark as non-terminal
                # so we properly bootstrap from value instead of 0
                segment_last_value = last_value
                # Force non-terminal since the episode didn't actually end
                segment_last_done = np.zeros((1,), dtype=np.float32)
            else:
                # Middle incomplete segment (shouldn't happen with current logic)
                # but handle defensively - treat as non-terminal
                segment_last_value = np.zeros((1,), dtype=np.float32)
                segment_last_done = np.zeros((1,), dtype=np.float32)

            # Compute GAE for this segment
            last_gae = 0.0
            for t in reversed(range(seg_start, seg_end)):
                if t == seg_end - 1:
                    next_non_terminal = 1.0 - segment_last_done
                    next_value = segment_last_value
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_value = self.values[t + 1]

                delta = (
                    self.rewards[t]
                    + self.gamma * next_value * next_non_terminal
                    - self.values[t]
                )
                self.advantages[t] = last_gae = (
                    delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                )

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_batches(
        self, batch_size: int, shuffle: bool = True
    ) -> list[dict[str, torch.Tensor]]:
        """Get minibatches for training.

        Args:
            batch_size: Size of each minibatch
            shuffle: Whether to shuffle the data

        Returns:
            List of dictionaries containing batched tensors
        """
        total_size = self.ptr * self.num_envs

        # Handle empty buffer or zero batch size
        if total_size == 0 or batch_size <= 0:
            return []

        indices = np.arange(total_size)

        if shuffle:
            np.random.shuffle(indices)

        # Flatten the buffer for batching
        def flatten(arr: np.ndarray) -> np.ndarray:
            return arr[:self.ptr].reshape(-1, *arr.shape[2:])

        flat_player_pokemon = flatten(self.player_pokemon)
        flat_opponent_pokemon = flatten(self.opponent_pokemon)
        flat_player_active_idx = flatten(self.player_active_idx)
        flat_opponent_active_idx = flatten(self.opponent_active_idx)
        flat_field_state = flatten(self.field_state)
        flat_action_mask = flatten(self.action_mask)
        flat_actions = flatten(self.actions)
        flat_log_probs = flatten(self.log_probs)
        flat_advantages = flatten(self.advantages)
        flat_returns = flatten(self.returns)
        flat_values = flatten(self.values)

        # Normalize advantages
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        batches = []
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]

            batch = {
                "player_pokemon": torch.tensor(
                    flat_player_pokemon[batch_indices], device=self.device
                ),
                "opponent_pokemon": torch.tensor(
                    flat_opponent_pokemon[batch_indices], device=self.device
                ),
                "player_active_idx": torch.tensor(
                    flat_player_active_idx[batch_indices], device=self.device
                ),
                "opponent_active_idx": torch.tensor(
                    flat_opponent_active_idx[batch_indices], device=self.device
                ),
                "field_state": torch.tensor(
                    flat_field_state[batch_indices], device=self.device
                ),
                "action_mask": torch.tensor(
                    flat_action_mask[batch_indices], device=self.device
                ),
                "actions": torch.tensor(
                    flat_actions[batch_indices], device=self.device
                ),
                "old_log_probs": torch.tensor(
                    flat_log_probs[batch_indices], device=self.device
                ),
                "advantages": torch.tensor(
                    flat_advantages[batch_indices], device=self.device
                ),
                "returns": torch.tensor(
                    flat_returns[batch_indices], device=self.device
                ),
                "old_values": torch.tensor(
                    flat_values[batch_indices], device=self.device
                ),
            }
            batches.append(batch)

        return batches

    def reset(self) -> None:
        """Reset the buffer for a new rollout."""
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False
        self.episode_ends = []

    @property
    def size(self) -> int:
        """Current number of transitions stored."""
        return self.ptr * self.num_envs

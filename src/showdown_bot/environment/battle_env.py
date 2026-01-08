"""Battle environment wrapper using poke-env."""

import asyncio
from typing import Any

import numpy as np
import torch
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import BattleOrder
from poke_env.battle import AbstractBattle

from showdown_bot.config import server_config, training_config
from showdown_bot.environment.state_encoder import EncodedState, StateEncoder


class RLPlayer(Player):
    """A Pokemon Showdown player that can be controlled by an RL agent."""

    def __init__(
        self,
        state_encoder: StateEncoder | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.state_encoder = state_encoder or StateEncoder()
        self._current_battle: AbstractBattle | None = None
        self._action_to_take: int | None = None
        self._waiting_for_action = asyncio.Event()
        self._action_ready = asyncio.Event()

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Called by poke-env when it's time to choose a move.

        This method bridges the async poke-env interface with our RL training loop.
        """
        self._current_battle = battle

        # Signal that we're waiting for an action
        self._waiting_for_action.set()

        # For now, return a random move if no action is set
        # In training, we'll use get_state() and set_action()
        if self._action_to_take is not None:
            order = self.state_encoder.action_to_battle_order(
                self._action_to_take, battle
            )
            self._action_to_take = None
            if order:
                return order

        # Fallback to random
        return self.choose_random_move(battle)

    def get_state(self) -> EncodedState | None:
        """Get the current battle state encoded for the neural network."""
        if self._current_battle is None:
            return None
        return self.state_encoder.encode_battle(self._current_battle)

    def set_action(self, action: int) -> None:
        """Set the action to take on the next move selection."""
        self._action_to_take = action
        self._action_ready.set()

    @property
    def current_battle(self) -> AbstractBattle | None:
        """Get the current battle object."""
        return self._current_battle


class MaxDamagePlayer(Player):
    """A simple heuristic player that always picks the highest damage move."""

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose the move that deals maximum expected damage."""
        if not battle.available_moves:
            return self.choose_random_move(battle)

        # Calculate expected damage for each move
        best_move = None
        best_damage = -1

        opponent = battle.opponent_active_pokemon
        if opponent is None:
            return self.choose_random_move(battle)

        for move in battle.available_moves:
            # Simple damage estimation
            damage = move.base_power

            # STAB bonus
            if battle.active_pokemon and move.type in [
                battle.active_pokemon.type_1,
                battle.active_pokemon.type_2,
            ]:
                damage *= 1.5

            # Type effectiveness (simplified)
            multiplier = opponent.damage_multiplier(move)
            damage *= multiplier

            if damage > best_damage:
                best_damage = damage
                best_move = move

        if best_move:
            return self.create_order(best_move)

        return self.choose_random_move(battle)


class NeuralNetworkPlayer(Player):
    """Player controlled by a neural network."""

    def __init__(
        self,
        model: torch.nn.Module,
        state_encoder: StateEncoder | None = None,
        device: torch.device | None = None,
        deterministic: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.state_encoder = state_encoder or StateEncoder()
        self.device = device or torch.device("cpu")
        self.deterministic = deterministic

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose a move using the neural network."""
        # Encode state
        state = self.state_encoder.encode_battle(battle)
        state = state.to_device(self.device)

        # Add batch dimension
        player_pokemon = state.player_pokemon.unsqueeze(0)
        opponent_pokemon = state.opponent_pokemon.unsqueeze(0)
        player_active_idx = torch.tensor([state.player_active_idx], device=self.device)
        opponent_active_idx = torch.tensor([state.opponent_active_idx], device=self.device)
        field_state = state.field_state.unsqueeze(0)
        action_mask = state.action_mask.unsqueeze(0)

        # Get action from model
        with torch.no_grad():
            self.model.eval()
            if self.deterministic:
                logits, _ = self.model(
                    player_pokemon,
                    opponent_pokemon,
                    player_active_idx,
                    opponent_active_idx,
                    field_state,
                    action_mask,
                )
                action = logits.argmax(dim=-1).item()
            else:
                action, _, _, _ = self.model.get_action_and_value(
                    player_pokemon,
                    opponent_pokemon,
                    player_active_idx,
                    opponent_active_idx,
                    field_state,
                    action_mask,
                )
                action = action.item()

        # Convert to battle order
        order = self.state_encoder.action_to_battle_order(action, battle)
        if order:
            return order

        # Fallback to random
        return self.choose_random_move(battle)


def calculate_reward(
    battle: AbstractBattle,
    prev_hp_fraction: float | None = None,
    prev_opponent_hp_fraction: float | None = None,
) -> float:
    """Calculate reward for the current battle state.

    Args:
        battle: Current battle state
        prev_hp_fraction: Previous total HP fraction for player
        prev_opponent_hp_fraction: Previous total HP fraction for opponent

    Returns:
        Reward value
    """
    # Terminal rewards
    if battle.won:
        return 1.0
    if battle.lost:
        return -1.0

    # Calculate current HP fractions
    our_hp = sum(
        p.current_hp_fraction for p in battle.team.values() if not p.fainted
    ) / 6
    opp_hp = sum(
        p.current_hp_fraction for p in battle.opponent_team.values() if not p.fainted
    ) / 6

    reward = 0.0

    # HP differential reward
    if prev_hp_fraction is not None and prev_opponent_hp_fraction is not None:
        our_hp_delta = our_hp - prev_hp_fraction
        opp_hp_delta = opp_hp - prev_opponent_hp_fraction

        # Reward for damaging opponent, penalty for taking damage
        reward += 0.1 * (opp_hp_delta * -1)  # Positive when opponent loses HP
        reward += 0.1 * our_hp_delta  # Positive when we don't lose HP

    # Knockout differential
    our_fainted = sum(1 for p in battle.team.values() if p.fainted)
    opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
    reward += 0.05 * (opp_fainted - our_fainted)

    return reward

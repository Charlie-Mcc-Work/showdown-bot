"""OU Player implementation using poke-env.

Provides player classes for:
1. OUTrainablePlayer - Collects experiences for training
2. OURLPlayer - RL training player with external action control
3. OUNeuralNetworkPlayer - Inference player using trained network
"""

import asyncio
import logging
from typing import Any

import numpy as np
import torch
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder
from poke_env.battle import AbstractBattle
from poke_env.teambuilder import Teambuilder
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon
from poke_env.data import to_id_str

from showdown_bot.ou.player.state_encoder import OUStateEncoder, OUEncodedState
from showdown_bot.ou.player.network import OUPlayerNetwork
from showdown_bot.ou.player.team_preview import HeuristicLeadSelector
from showdown_bot.ou.shared.data_loader import Team, TeamLoader

logger = logging.getLogger(__name__)

# Shared lead selector instance
_lead_selector = HeuristicLeadSelector()


def _get_team_preview_order(battle: AbstractBattle) -> str:
    """Generate team preview order string using heuristic lead selection.

    Args:
        battle: Battle in team preview phase

    Returns:
        Team order string like "/team 312456"
    """
    # Extract Pokemon species from both teams
    our_team = [p.species for p in battle.team.values()]
    opp_team = [p.species for p in battle.opponent_team.values()]

    # Get lead index (0-5)
    lead_idx = _lead_selector.select_lead(our_team, opp_team)

    # Build ordering: lead first, then rest in original order
    # Team preview uses 1-indexed positions
    ordering = [lead_idx + 1]  # Lead (1-indexed)
    for i in range(6):
        if i != lead_idx:
            ordering.append(i + 1)

    return "/team " + "".join(str(x) for x in ordering)


def calculate_ou_reward(
    battle: AbstractBattle,
    prev_hp_fraction: float | None = None,
    prev_opponent_hp_fraction: float | None = None,
) -> float:
    """Calculate reward for the current OU battle state.

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


class OUTrainablePlayer(Player):
    """OU Player that collects experiences for training.

    This player uses the network to select actions and records all
    transitions (state, action, log_prob, value, reward) for PPO training.
    """

    def __init__(
        self,
        network: OUPlayerNetwork,
        state_encoder: OUStateEncoder,
        device: torch.device,
        teams: list[Team] | None = None,
        current_team: Team | None = None,
        **kwargs: Any,
    ):
        """Initialize the trainable player.

        Args:
            network: The OUPlayerNetwork to use for decisions
            state_encoder: State encoder for battles
            device: Device for inference
            teams: Teams to use (picks randomly)
            current_team: Specific team to use this battle
            **kwargs: Arguments passed to poke-env Player
        """
        if teams:
            kwargs.setdefault("team", OUTeambuilder(teams))

        super().__init__(**kwargs)
        self.network = network
        self.state_encoder = state_encoder
        self.device = device
        self.current_team = current_team

        # Experience storage for current battle
        self.current_experiences: list[dict[str, Any]] = []
        self.prev_hp_fraction: float | None = None
        self.prev_opp_hp_fraction: float | None = None

        # Statistics - track per rollout
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._rollout_wins: int = 0
        self._rollout_battles: int = 0

    def teampreview(self, battle: AbstractBattle) -> str:
        """Select lead Pokemon during team preview phase."""
        return _get_team_preview_order(battle)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose a move and store the experience."""
        # Encode state
        state = self.state_encoder.encode_battle(battle)
        state = state.to_device(self.device)

        # Get action from network
        with torch.no_grad():
            self.network.eval()
            output = self.network(state)
            policy = output["policy"]
            value = output["value"]

            # Sample action from policy
            if policy.dim() == 1:
                policy = policy.unsqueeze(0)

            action = torch.multinomial(policy, 1).item()
            log_prob = torch.log(policy[0, action] + 1e-8).item()
            value_val = value.item() if value.dim() == 0 else value.squeeze().item()

        # Calculate reward
        reward = calculate_ou_reward(
            battle,
            self.prev_hp_fraction,
            self.prev_opp_hp_fraction,
        )

        # Update HP tracking
        self.prev_hp_fraction = sum(
            p.current_hp_fraction for p in battle.team.values() if not p.fainted
        ) / 6
        self.prev_opp_hp_fraction = sum(
            p.current_hp_fraction for p in battle.opponent_team.values() if not p.fainted
        ) / 6

        # Store experience (keep state on CPU for memory efficiency)
        self.current_experiences.append({
            "state": state.to_device(torch.device("cpu")),
            "action": action,
            "log_prob": log_prob,
            "value": value_val,
            "reward": reward,
            "done": False,
            "team": self.current_team,
        })

        # Convert action to battle order
        order = self.state_encoder.action_to_battle_order(action, battle)
        if order:
            return order

        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        """Called when a battle finishes."""
        # Calculate final reward
        if battle.won:
            final_reward = 1.0
            self._rollout_wins += 1
        elif battle.lost:
            final_reward = -1.0
        else:
            final_reward = 0.0

        self._rollout_battles += 1

        # Update last experience with terminal reward
        if self.current_experiences:
            self.current_experiences[-1]["reward"] = final_reward
            self.current_experiences[-1]["done"] = True

            # Track episode stats
            total_reward = sum(exp["reward"] for exp in self.current_experiences)
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(len(self.current_experiences))

        # Reset for next battle
        self.prev_hp_fraction = None
        self.prev_opp_hp_fraction = None

    def get_experiences(self) -> list[dict[str, Any]]:
        """Get collected experiences and clear the buffer."""
        experiences = self.current_experiences.copy()
        self.current_experiences = []
        return experiences

    def reset_stats(self) -> None:
        """Reset episode statistics for new rollout."""
        self.episode_rewards = []
        self.episode_lengths = []
        self._rollout_wins = 0
        self._rollout_battles = 0

    def get_rollout_stats(self) -> dict[str, float]:
        """Get stats for the current rollout."""
        return {
            "battles": self._rollout_battles,
            "wins": self._rollout_wins,
            "win_rate": self._rollout_wins / self._rollout_battles if self._rollout_battles > 0 else 0.0,
            "avg_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "avg_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
        }


class OUTeambuilder(Teambuilder):
    """Teambuilder that uses pre-loaded teams.

    Returns teams in packed format as required by poke-env.
    """

    def __init__(self, teams: list[Team]):
        """Initialize with a list of teams.

        Args:
            teams: List of Team objects to use
        """
        self.teams = teams
        self._team_idx = 0

    def yield_team(self) -> str:
        """Yield the next team in packed format (required by poke-env)."""
        if not self.teams:
            return ""

        team = self.teams[self._team_idx % len(self.teams)]
        self._team_idx += 1
        return self._team_to_packed(team)

    def _team_to_packed(self, team: Team) -> str:
        """Convert a Team to poke-env packed format.

        Packed format: nickname|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|happiness,hp,,,tera
        Pokemon separated by ]
        """
        pokemon_packed = []

        for pokemon in team.pokemon:
            # Convert EVs dict to list [hp, atk, def, spa, spd, spe]
            ev_order = ["hp", "atk", "def", "spa", "spd", "spe"]
            evs = [pokemon.evs.get(stat, 0) for stat in ev_order] if pokemon.evs else [0] * 6

            # Convert IVs dict to list
            ivs = [pokemon.ivs.get(stat, 31) for stat in ev_order] if pokemon.ivs else [31] * 6

            # Create TeambuilderPokemon
            tb_pokemon = TeambuilderPokemon(
                nickname=pokemon.nickname,
                species=pokemon.species,
                item=pokemon.item,
                ability=pokemon.ability,
                moves=pokemon.moves,
                nature=pokemon.nature,
                evs=evs,
                ivs=ivs,
                tera_type=pokemon.tera_type,
            )

            pokemon_packed.append(tb_pokemon.packed)

        return "]".join(pokemon_packed)


class OURLPlayer(Player):
    """OU Player for RL training.

    Similar to RLPlayer but for OU format with team support.
    """

    def __init__(
        self,
        state_encoder: OUStateEncoder | None = None,
        teams: list[Team] | None = None,
        team_loader: TeamLoader | None = None,
        **kwargs: Any,
    ):
        """Initialize the OU RL player.

        Args:
            state_encoder: State encoder for battles
            teams: Pre-loaded teams to use
            team_loader: TeamLoader instance to load teams
            **kwargs: Arguments passed to poke-env Player
        """
        # Set up teambuilder if teams provided
        if teams:
            kwargs.setdefault("team", OUTeambuilder(teams))
        elif team_loader:
            loaded_teams = team_loader.load_teams("gen9ou")
            if loaded_teams:
                kwargs.setdefault("team", OUTeambuilder(loaded_teams))

        super().__init__(**kwargs)
        self.state_encoder = state_encoder or OUStateEncoder()

        self._current_battle: AbstractBattle | None = None
        self._action_to_take: int | None = None
        self._waiting_for_action = asyncio.Event()
        self._action_ready = asyncio.Event()

    def teampreview(self, battle: AbstractBattle) -> str:
        """Select lead Pokemon during team preview phase."""
        return _get_team_preview_order(battle)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Called by poke-env when it's time to choose a move."""
        self._current_battle = battle

        # Signal that we're waiting for an action
        self._waiting_for_action.set()

        # Use the action if set
        if self._action_to_take is not None:
            order = self.state_encoder.action_to_battle_order(
                self._action_to_take, battle
            )
            self._action_to_take = None
            if order:
                return order

        # Fallback to random
        return self.choose_random_move(battle)

    def get_state(self) -> OUEncodedState | None:
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


class OUNeuralNetworkPlayer(Player):
    """OU Player controlled by a neural network for inference."""

    def __init__(
        self,
        network: OUPlayerNetwork,
        state_encoder: OUStateEncoder | None = None,
        device: torch.device | None = None,
        deterministic: bool = False,
        teams: list[Team] | None = None,
        **kwargs: Any,
    ):
        """Initialize the neural network player.

        Args:
            network: The trained OUPlayerNetwork
            state_encoder: State encoder for battles
            device: Device for inference
            deterministic: If True, always pick best action
            teams: Teams to use
            **kwargs: Arguments passed to poke-env Player
        """
        if teams:
            kwargs.setdefault("team", OUTeambuilder(teams))

        super().__init__(**kwargs)
        self.network = network
        self.state_encoder = state_encoder or OUStateEncoder()
        self.device = device or torch.device("cpu")
        self.deterministic = deterministic

    def teampreview(self, battle: AbstractBattle) -> str:
        """Select lead Pokemon during team preview phase."""
        return _get_team_preview_order(battle)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose a move using the neural network."""
        # Handle cleanup race condition
        if not hasattr(self, 'network') or self.network is None:
            logger.warning(
                f"OUNeuralNetworkPlayer network is None during choose_move "
                f"(battle turn {battle.turn}) - using random move."
            )
            return self.choose_random_move(battle)

        # Encode state
        state = self.state_encoder.encode_battle(battle)
        state = state.to_device(self.device)

        # Get action from network
        with torch.no_grad():
            self.network.eval()
            action, _ = self.network.get_action(state, deterministic=self.deterministic)

        # Convert to battle order
        order = self.state_encoder.action_to_battle_order(action, battle)
        if order:
            return order

        # Fallback to random
        return self.choose_random_move(battle)


class OUMaxDamagePlayer(Player):
    """Simple heuristic player for OU that picks highest damage moves."""

    def __init__(self, teams: list[Team] | None = None, **kwargs: Any):
        if teams:
            kwargs.setdefault("team", OUTeambuilder(teams))
        super().__init__(**kwargs)

    def teampreview(self, battle: AbstractBattle) -> str:
        """Select lead Pokemon during team preview phase."""
        return _get_team_preview_order(battle)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose the move that deals maximum expected damage."""
        if not battle.available_moves:
            return self.choose_random_move(battle)

        best_move = None
        best_damage = -1

        opponent = battle.opponent_active_pokemon
        if opponent is None:
            return self.choose_random_move(battle)

        for move in battle.available_moves:
            damage = move.base_power or 0

            # STAB bonus
            if battle.active_pokemon and move.type in [
                battle.active_pokemon.type_1,
                battle.active_pokemon.type_2,
            ]:
                damage *= 1.5

            # Type effectiveness
            multiplier = opponent.damage_multiplier(move)
            damage *= multiplier

            if damage > best_damage:
                best_damage = damage
                best_move = move

        if best_move:
            return self.create_order(best_move)

        return self.choose_random_move(battle)

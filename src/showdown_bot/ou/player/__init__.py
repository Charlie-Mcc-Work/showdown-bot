"""OU Player module for battle decision-making.

The player handles:
1. Team preview lead selection
2. In-battle move/switch decisions
3. Opponent modeling and prediction

Key differences from Random Battles player:
- Full knowledge of own team (moves, items, EVs)
- Team preview phase before battle starts
- More sophisticated opponent prediction (usage-based)
- Tera type decision making

See ../README.md for architecture details.
"""

from showdown_bot.ou.player.state_encoder import OUStateEncoder, OUEncodedState
from showdown_bot.ou.player.network import OUPlayerNetwork, OUActorCritic
from showdown_bot.ou.player.team_preview import (
    TeamPreviewSelector,
    HeuristicLeadSelector,
    TeamPreviewAnalyzer,
)
from showdown_bot.ou.player.ou_player import (
    OURLPlayer,
    OUTrainablePlayer,
    OUNeuralNetworkPlayer,
    OUMaxDamagePlayer,
    OUTeambuilder,
    calculate_ou_reward,
    _get_team_preview_order,
)

__all__ = [
    "OUStateEncoder",
    "OUEncodedState",
    "OUPlayerNetwork",
    "OUActorCritic",
    "TeamPreviewSelector",
    "HeuristicLeadSelector",
    "TeamPreviewAnalyzer",
    "OURLPlayer",
    "OUTrainablePlayer",
    "OUNeuralNetworkPlayer",
    "OUMaxDamagePlayer",
    "OUTeambuilder",
    "calculate_ou_reward",
    "_get_team_preview_order",
]

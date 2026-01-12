"""Teambuilder module for Gen 9 OU.

The teambuilder constructs competitive teams using:
1. Autoregressive generation (one Pokemon at a time)
2. Learned value function for team quality
3. Usage stats priors for move/item selection
4. Player feedback for optimization

Key components:
- TeamGenerator: Generates complete teams autoregressively
- TeamEvaluator: Predicts expected win rate for a team
- TeamRepresentation: Encodes teams for the neural network

See ../README.md for architecture details.
"""

from showdown_bot.ou.teambuilder.generator import TeamGenerator
from showdown_bot.ou.teambuilder.evaluator import TeamEvaluator
from showdown_bot.ou.teambuilder.team_repr import TeamRepresentation

__all__ = [
    "TeamGenerator",
    "TeamEvaluator",
    "TeamRepresentation",
]

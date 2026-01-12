"""Team preview lead selection for OU battles.

In OU (unlike Random Battles), players see both teams before battle
and choose their lead Pokemon. This module handles:
1. Analyzing opponent team threats
2. Evaluating lead matchups
3. Selecting optimal lead
4. (Future) Predicting opponent's likely lead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from showdown_bot.ou.shared.embeddings import SharedEmbeddings
from showdown_bot.ou.shared.encoders import FullTeamEncoder


class TeamPreviewSelector(nn.Module):
    """Selects lead Pokemon during team preview.

    Takes both teams' compositions and outputs a distribution
    over which of our 6 Pokemon to lead with.

    Architecture:
    1. Encode both teams
    2. Compute matchup features for each of our Pokemon vs opponent team
    3. Score each potential lead
    4. Output selection distribution
    """

    def __init__(
        self,
        shared_embeddings: SharedEmbeddings | None = None,
        hidden_dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Team encoders (will use shared if provided)
        self.our_team_encoder: nn.Module | None = None
        self.opp_team_encoder: nn.Module | None = None

        # If no shared embeddings, create simple encoders
        if shared_embeddings is None:
            self.our_team_proj = nn.Linear(128, hidden_dim)  # Placeholder
            self.opp_team_proj = nn.Linear(128, hidden_dim)
        else:
            self.our_team_encoder = FullTeamEncoder(
                shared_embeddings,
                pokemon_hidden=hidden_dim,
                team_hidden=hidden_dim,
            )
            self.opp_team_encoder = FullTeamEncoder(
                shared_embeddings,
                pokemon_hidden=hidden_dim,
                team_hidden=hidden_dim,
            )

        # Cross-attention: each of our Pokemon attends to opponent team
        self.matchup_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Lead scoring head
        self.lead_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Opponent lead prediction (auxiliary task)
        self.opp_lead_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(
        self,
        our_team_encoding: torch.Tensor,
        opp_team_encoding: torch.Tensor,
        our_pokemon_encodings: torch.Tensor,
        opp_pokemon_encodings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute lead selection scores.

        Args:
            our_team_encoding: (batch, hidden_dim) Global team encoding
            opp_team_encoding: (batch, hidden_dim) Opponent team encoding
            our_pokemon_encodings: (batch, 6, hidden_dim) Per-Pokemon encodings
            opp_pokemon_encodings: (batch, 6, hidden_dim) Opponent per-Pokemon

        Returns:
            Dict with:
            - lead_logits: (batch, 6) Raw scores for each lead
            - lead_probs: (batch, 6) Probability distribution over leads
            - opp_lead_logits: (batch, 6) Predicted opponent lead probs
        """
        batch_size = our_team_encoding.size(0)

        # Cross-attention: our Pokemon attend to opponent team
        # This computes matchup-aware representations
        matchup_repr, matchup_attn = self.matchup_attention(
            our_pokemon_encodings,
            opp_pokemon_encodings,
            opp_pokemon_encodings,
        )  # (batch, 6, hidden_dim)

        # Combine Pokemon repr with matchup context
        combined = torch.cat([our_pokemon_encodings, matchup_repr], dim=-1)

        # Score each potential lead
        lead_scores = self.lead_scorer(combined).squeeze(-1)  # (batch, 6)

        # Predict opponent's lead (auxiliary task for training)
        team_context = torch.cat([our_team_encoding, opp_team_encoding], dim=-1)
        opp_lead_logits = self.opp_lead_predictor(team_context)

        return {
            "lead_logits": lead_scores,
            "lead_probs": F.softmax(lead_scores, dim=-1),
            "opp_lead_logits": opp_lead_logits,
            "matchup_attention": matchup_attn,
        }

    def select_lead(
        self,
        our_team_encoding: torch.Tensor,
        opp_team_encoding: torch.Tensor,
        our_pokemon_encodings: torch.Tensor,
        opp_pokemon_encodings: torch.Tensor,
        deterministic: bool = True,
    ) -> int:
        """Select a lead Pokemon.

        Args:
            our_team_encoding: Global team encoding
            opp_team_encoding: Opponent team encoding
            our_pokemon_encodings: Per-Pokemon encodings
            opp_pokemon_encodings: Opponent per-Pokemon encodings
            deterministic: If True, take argmax. If False, sample.

        Returns:
            Index of selected lead (0-5)
        """
        with torch.no_grad():
            output = self.forward(
                our_team_encoding,
                opp_team_encoding,
                our_pokemon_encodings,
                opp_pokemon_encodings,
            )

            lead_probs = output["lead_probs"]

            if deterministic:
                return lead_probs.argmax(dim=-1).item()
            else:
                return torch.multinomial(lead_probs, 1).item()


class HeuristicLeadSelector:
    """Rule-based lead selector for bootstrapping.

    Uses simple heuristics before the neural network is trained:
    1. Type advantage against opponent team
    2. Speed tier (fast Pokemon often good leads)
    3. Role (hazard setters, pivots often good leads)
    4. Matchup against likely opponent leads
    """

    # Common lead Pokemon and their roles
    LEAD_ROLES = {
        "great tusk": ["hazard setter", "spinner"],
        "gliscor": ["hazard setter", "pivot"],
        "kingambit": ["sweeper"],
        "gholdengo": ["hazard blocker", "special attacker"],
        "dragapult": ["fast attacker", "pivot"],
        "iron valiant": ["fast attacker"],
        "dragonite": ["setup sweeper"],
        "landorus": ["hazard setter", "pivot"],
        "zamazenta": ["hazard setter", "pivot"],
        "kyurem": ["wallbreaker"],
    }

    # Priority for lead selection
    ROLE_PRIORITY = [
        "hazard setter",
        "hazard blocker",
        "pivot",
        "fast attacker",
        "setup sweeper",
        "wallbreaker",
        "wall",
    ]

    def __init__(self):
        pass

    def select_lead(
        self,
        our_team: list[str],
        opp_team: list[str],
    ) -> int:
        """Select lead using heuristics.

        Args:
            our_team: List of our Pokemon species names
            opp_team: List of opponent Pokemon species names

        Returns:
            Index of selected lead (0-5)
        """
        scores = []

        for i, pokemon in enumerate(our_team):
            score = self._score_lead(pokemon.lower(), opp_team)
            scores.append((score, i))

        # Return index of highest scoring lead
        scores.sort(reverse=True)
        return scores[0][1]

    def _score_lead(self, pokemon: str, opp_team: list[str]) -> float:
        """Score a Pokemon as a potential lead."""
        score = 0.0

        # Check if it's a known good lead
        if pokemon in self.LEAD_ROLES:
            roles = self.LEAD_ROLES[pokemon]

            # Hazard setter is high priority if opponent has no spinner
            if "hazard setter" in roles:
                score += 3.0

            # Hazard blocker is good vs hazard-heavy teams
            if "hazard blocker" in roles:
                score += 2.0

            if "pivot" in roles:
                score += 1.5

            if "fast attacker" in roles:
                score += 1.0

        # TODO: Add type matchup analysis
        # TODO: Add speed tier analysis

        return score


class TeamPreviewAnalyzer:
    """Analyzes team preview for strategic insights.

    Provides interpretable analysis of:
    - Predicted opponent team archetype
    - Threat assessment
    - Win condition identification
    - Recommended game plan
    """

    # Team archetypes
    ARCHETYPES = [
        "hyper offense",
        "bulky offense",
        "balance",
        "semi-stall",
        "stall",
        "rain",
        "sun",
        "sand",
        "trick room",
        "volt-turn",
    ]

    def __init__(self):
        pass

    def analyze_teams(
        self,
        our_team: list[str],
        opp_team: list[str],
    ) -> dict:
        """Analyze both teams and provide strategic insights.

        Args:
            our_team: List of our Pokemon species
            opp_team: List of opponent Pokemon species

        Returns:
            Dict with analysis results
        """
        return {
            "our_archetype": self._detect_archetype(our_team),
            "opp_archetype": self._detect_archetype(opp_team),
            "threats": self._identify_threats(opp_team),
            "win_conditions": self._identify_win_conditions(our_team, opp_team),
            "recommended_lead": self._recommend_lead(our_team, opp_team),
        }

    def _detect_archetype(self, team: list[str]) -> str:
        """Detect team archetype from Pokemon composition."""
        # TODO: Implement based on Pokemon roles and playstyle
        return "balance"  # Placeholder

    def _identify_threats(self, opp_team: list[str]) -> list[str]:
        """Identify threatening Pokemon on opponent's team."""
        # TODO: Implement based on our team's weaknesses
        return opp_team[:2]  # Placeholder

    def _identify_win_conditions(
        self,
        our_team: list[str],
        opp_team: list[str],
    ) -> list[str]:
        """Identify paths to victory."""
        # TODO: Implement based on matchup analysis
        return ["sweep with setup mon", "chip and clean"]  # Placeholder

    def _recommend_lead(
        self,
        our_team: list[str],
        opp_team: list[str],
    ) -> tuple[int, str]:
        """Recommend lead with reasoning."""
        selector = HeuristicLeadSelector()
        lead_idx = selector.select_lead(our_team, opp_team)
        return (lead_idx, f"Lead with {our_team[lead_idx]} for momentum")

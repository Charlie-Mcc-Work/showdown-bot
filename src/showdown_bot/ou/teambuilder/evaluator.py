"""Team evaluator for predicting team quality.

The evaluator predicts expected win rate for a given team. This is used to:
1. Guide team generation (MCTS-style search)
2. Filter generated teams
3. Provide training signal for the generator

Training data comes from actual battles played by the trained player.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from showdown_bot.ou.shared.embeddings import SharedEmbeddings
from showdown_bot.ou.shared.encoders import FullTeamEncoder
from showdown_bot.ou.shared.data_loader import Team, TeamSet, PokemonDataLoader
from showdown_bot.ou.teambuilder.team_repr import PartialTeam, PokemonSlot


# Nature modifiers: (boosted_stat, reduced_stat)
NATURE_MODIFIERS = {
    "hardy": (None, None),
    "lonely": ("atk", "def"),
    "brave": ("atk", "spe"),
    "adamant": ("atk", "spa"),
    "naughty": ("atk", "spd"),
    "bold": ("def", "atk"),
    "docile": (None, None),
    "relaxed": ("def", "spe"),
    "impish": ("def", "spa"),
    "lax": ("def", "spd"),
    "timid": ("spe", "atk"),
    "hasty": ("spe", "def"),
    "serious": (None, None),
    "jolly": ("spe", "spa"),
    "naive": ("spe", "spd"),
    "modest": ("spa", "atk"),
    "mild": ("spa", "def"),
    "quiet": ("spa", "spe"),
    "bashful": (None, None),
    "rash": ("spa", "spd"),
    "calm": ("spd", "atk"),
    "gentle": ("spd", "def"),
    "sassy": ("spd", "spe"),
    "careful": ("spd", "spa"),
    "quirky": (None, None),
}

# Item categories for embedding
ITEM_CATEGORIES = {
    "berry": 1,
    "choice": 2,
    "life orb": 3,
    "leftovers": 4,
    "rocky helmet": 5,
    "assault vest": 6,
    "focus sash": 7,
    "heavy-duty boots": 8,
    "eviolite": 9,
    "safety goggles": 10,
    "air balloon": 11,
    "covert cloak": 12,
    "protective pads": 13,
    "weakness policy": 14,
    "booster energy": 15,
    "expert belt": 16,
    "sitrus berry": 1,
    "lum berry": 1,
    "other": 0,
}

# Ability effect categories for embedding
ABILITY_EFFECT_CATEGORIES = {
    "weather": 1,  # Drought, Drizzle, Sand Stream, Snow Warning
    "terrain": 2,  # Electric/Grassy/Psychic/Misty Surge
    "stat_boost": 3,  # Intimidate, Download, Speed Boost
    "immunity": 4,  # Levitate, Water Absorb, Flash Fire
    "contact": 5,  # Rough Skin, Iron Barbs, Static
    "priority": 6,  # Prankster, Gale Wings
    "type_change": 7,  # Protean, Libero
    "field_effect": 8,  # Stealth Rock immunity (Magic Bounce), etc.
    "healing": 9,  # Regenerator, Natural Cure
    "power_boost": 10,  # Technician, Adaptability, Huge Power
    "defensive": 11,  # Multiscale, Sturdy, Unaware
    "offensive": 12,  # Mold Breaker, Teravolt
    "other": 0,
}

# Map common abilities to categories
ABILITY_TO_CATEGORY = {
    # Weather
    "drought": "weather", "drizzle": "weather", "sandstream": "weather",
    "snowwarning": "weather", "orichalcumpulse": "weather", "hadronengine": "weather",
    # Terrain
    "electricsurge": "terrain", "grassysurge": "terrain",
    "psychicsurge": "terrain", "mistysurge": "terrain",
    # Stat boost
    "intimidate": "stat_boost", "download": "stat_boost", "speedboost": "stat_boost",
    "beastboost": "stat_boost", "moody": "stat_boost", "compoundeyes": "stat_boost",
    "quarkdrive": "stat_boost", "protosynthesis": "stat_boost",
    "supremeoverlord": "stat_boost", "swordofruin": "stat_boost",
    "beadsofruin": "stat_boost", "tabletsofruin": "stat_boost", "vesselofruin": "stat_boost",
    # Immunity
    "levitate": "immunity", "waterabsorb": "immunity", "flashfire": "immunity",
    "voltabsorb": "immunity", "stormdrain": "immunity", "sapsipper": "immunity",
    "lightningrod": "immunity", "motordrive": "immunity", "eartheater": "immunity",
    # Contact
    "roughskin": "contact", "ironbarbs": "contact", "static": "contact",
    "poisonpoint": "contact", "flamebody": "contact", "effectspore": "contact",
    # Priority
    "prankster": "priority", "galewings": "priority", "triage": "priority",
    # Type change
    "protean": "type_change", "libero": "type_change", "colorchange": "type_change",
    # Field effect
    "magicbounce": "field_effect", "naturalcure": "field_effect",
    # Healing
    "regenerator": "healing", "poisonheal": "healing",
    # Power boost
    "technician": "power_boost", "adaptability": "power_boost", "hugepower": "power_boost",
    "purepower": "power_boost", "sheerforce": "power_boost", "toughclaws": "power_boost",
    "strongjaw": "power_boost", "ironfist": "power_boost", "aerilate": "power_boost",
    "pixilate": "power_boost", "refrigerate": "power_boost", "galvanize": "power_boost",
    # Defensive
    "multiscale": "defensive", "sturdy": "defensive", "unaware": "defensive",
    "furcoat": "defensive", "fluffy": "defensive", "icescales": "defensive",
    "shadowshield": "defensive", "filter": "defensive", "solidrock": "defensive",
    # Offensive
    "moldbreaker": "offensive", "teravolt": "offensive", "turboblaze": "offensive",
    "myceliummight": "offensive",
}


class TeamTensorEncoder:
    """Converts Team/PartialTeam objects to tensor format for neural networks.

    This encoder bridges the gap between the teambuilder's data structures
    and the neural network encoders. It handles:
    1. ID lookups for species, moves, items, abilities
    2. Type and stat encoding
    3. Move properties encoding
    4. EV/Nature stat modifier encoding
    """

    def __init__(self, data_loader: PokemonDataLoader):
        """Initialize the encoder.

        Args:
            data_loader: PokemonDataLoader instance for ID lookups
        """
        self.data_loader = data_loader
        self.data_loader.load()

    def _normalize_name(self, name: str | None) -> str:
        """Normalize a name for lookup."""
        if name is None:
            return ""
        return name.lower().replace(" ", "").replace("-", "")

    def _get_item_category(self, item: str | None) -> int:
        """Get item category ID for embedding."""
        if item is None:
            return 0

        item_lower = item.lower()
        # Check direct matches
        if item_lower in ITEM_CATEGORIES:
            return ITEM_CATEGORIES[item_lower]

        # Check for category keywords
        if "berry" in item_lower:
            return ITEM_CATEGORIES["berry"]
        if "choice" in item_lower:
            return ITEM_CATEGORIES["choice"]

        return ITEM_CATEGORIES["other"]

    def _get_ability_effect(self, ability: str | None) -> int:
        """Get ability effect category ID for embedding."""
        if ability is None:
            return 0

        ability_norm = self._normalize_name(ability)
        category = ABILITY_TO_CATEGORY.get(ability_norm, "other")
        return ABILITY_EFFECT_CATEGORIES.get(category, 0)

    def _encode_stat_mods(
        self,
        evs: dict[str, int] | None,
        nature: str | None,
    ) -> torch.Tensor:
        """Encode EVs and nature into a 7-dimensional tensor.

        Returns:
            (7,) tensor: [hp_ev, atk_ev, def_ev, spa_ev, spd_ev, spe_ev, nature_idx]
            EVs are normalized to [0, 1] by dividing by 252.
        """
        # Default EVs (if none provided)
        ev_values = [0.0] * 6
        if evs:
            stat_order = ["hp", "atk", "def", "spa", "spd", "spe"]
            for i, stat in enumerate(stat_order):
                ev_values[i] = evs.get(stat, 0) / 252.0  # Normalize

        # Nature modifier as a single value
        # 0 = neutral, positive = boosted stat index, negative = reduced stat index
        nature_mod = 0.0
        if nature:
            nature_lower = nature.lower()
            mods = NATURE_MODIFIERS.get(nature_lower, (None, None))
            if mods[0] is not None and mods[1] is not None:
                stat_to_idx = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}
                boost_idx = stat_to_idx.get(mods[0], 0)
                reduce_idx = stat_to_idx.get(mods[1], 0)
                # Encode as: boost_idx * 6 + reduce_idx (1-35 range for non-neutral)
                nature_mod = (boost_idx * 6 + reduce_idx + 1) / 36.0  # Normalize

        return torch.tensor(ev_values + [nature_mod], dtype=torch.float32)

    def _encode_move_properties(self, move_name: str | None) -> torch.Tensor:
        """Encode move properties into a 20-dimensional tensor.

        Format: [power, accuracy, pp, priority, category_physical, category_special,
                 category_status, is_contact, is_sound, is_bullet, is_punch,
                 is_bite, is_pulse, is_dance, is_wind, is_slicing,
                 is_priority, is_recoil, is_drain, is_setup]
        """
        properties = torch.zeros(20, dtype=torch.float32)

        if move_name is None:
            return properties

        move_data = self.data_loader.get_move(move_name)
        if move_data is None:
            return properties

        # Power (normalized to 0-1, max ~200)
        properties[0] = (move_data.power or 0) / 200.0

        # Accuracy (normalized, 100 = 1.0)
        properties[1] = (move_data.accuracy or 100) / 100.0 if move_data.accuracy else 1.0

        # PP (normalized, max ~40)
        properties[2] = move_data.pp / 40.0

        # Priority (normalized -7 to +5 -> 0 to 1)
        properties[3] = (move_data.priority + 7) / 12.0

        # Category one-hot
        category = move_data.category.lower()
        if category == "physical":
            properties[4] = 1.0
        elif category == "special":
            properties[5] = 1.0
        else:  # status
            properties[6] = 1.0

        # Flags
        flags = move_data.flags
        properties[7] = float(flags.get("contact", False))
        properties[8] = float(flags.get("sound", False))
        properties[9] = float(flags.get("bullet", False))
        properties[10] = float(flags.get("punch", False))
        properties[11] = float(flags.get("bite", False))
        properties[12] = float(flags.get("pulse", False))
        properties[13] = float(flags.get("dance", False))
        properties[14] = float(flags.get("wind", False))
        properties[15] = float(flags.get("slicing", False))

        # Additional derived properties
        properties[16] = float(move_data.priority > 0)  # is_priority
        properties[17] = float(flags.get("recoil", False))
        properties[18] = float(flags.get("heal", False))  # drain moves

        # Setup moves (boost stats) - check effect description
        effect_lower = move_data.effect.lower()
        is_setup = any(x in effect_lower for x in ["raises", "boosts", "raises the user"])
        properties[19] = float(is_setup)

        return properties

    def encode_pokemon(
        self,
        pokemon: TeamSet | PokemonSlot,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a single Pokemon into tensor format.

        Args:
            pokemon: TeamSet or PokemonSlot to encode
            device: Device for tensors

        Returns:
            Dict with all tensors needed for FullTeamEncoder.encode_pokemon()
        """
        device = device or torch.device("cpu")

        # Handle both TeamSet and PokemonSlot
        if isinstance(pokemon, PokemonSlot):
            species = pokemon.species
            moves = pokemon.moves or []
            item = pokemon.item
            ability = pokemon.ability
            nature = pokemon.nature
            evs = pokemon.evs
        else:  # TeamSet
            species = pokemon.species
            moves = pokemon.moves
            item = pokemon.item
            ability = pokemon.ability
            nature = pokemon.nature
            evs = pokemon.evs

        # Species ID and data
        species_id = self.data_loader.get_species_id(species or "")
        pokemon_data = self.data_loader.get_pokemon(species or "") if species else None

        # Type IDs
        if pokemon_data:
            type1_id = self.data_loader.get_type_id(pokemon_data.type1)
            type2_id = self.data_loader.get_type_id(pokemon_data.type2) if pokemon_data.type2 else 0
            base_stats = torch.tensor([
                pokemon_data.base_stats.get("hp", 0) / 255.0,
                pokemon_data.base_stats.get("atk", 0) / 255.0,
                pokemon_data.base_stats.get("def", 0) / 255.0,
                pokemon_data.base_stats.get("spa", 0) / 255.0,
                pokemon_data.base_stats.get("spd", 0) / 255.0,
                pokemon_data.base_stats.get("spe", 0) / 255.0,
            ], dtype=torch.float32)
        else:
            type1_id = 0
            type2_id = 0
            base_stats = torch.zeros(6, dtype=torch.float32)

        # Move IDs, types, and properties
        move_ids = []
        move_types = []
        move_properties = []
        for i in range(4):
            if i < len(moves):
                move_name = moves[i]
                move_id = self.data_loader.get_move_id(move_name)
                move_data = self.data_loader.get_move(move_name)
                move_type = self.data_loader.get_type_id(move_data.type if move_data else "normal")
                move_props = self._encode_move_properties(move_name)
            else:
                move_id = 0
                move_type = 0
                move_props = torch.zeros(20, dtype=torch.float32)

            move_ids.append(move_id)
            move_types.append(move_type)
            move_properties.append(move_props)

        # Item
        item_id = self.data_loader.get_item_id(item or "")
        item_category = self._get_item_category(item)

        # Ability
        ability_id = self.data_loader.get_ability_id(ability or "")
        ability_effect = self._get_ability_effect(ability)

        # Stat mods (EVs + nature)
        stat_mods = self._encode_stat_mods(evs, nature)

        return {
            "species_id": torch.tensor([species_id], dtype=torch.long, device=device),
            "type1_id": torch.tensor([type1_id], dtype=torch.long, device=device),
            "type2_id": torch.tensor([type2_id], dtype=torch.long, device=device),
            "base_stats": base_stats.unsqueeze(0).to(device),
            "move_ids": torch.tensor([move_ids], dtype=torch.long, device=device),
            "move_types": torch.tensor([move_types], dtype=torch.long, device=device),
            "move_properties": torch.stack(move_properties).unsqueeze(0).to(device),
            "item_id": torch.tensor([item_id], dtype=torch.long, device=device),
            "item_category": torch.tensor([item_category], dtype=torch.long, device=device),
            "ability_id": torch.tensor([ability_id], dtype=torch.long, device=device),
            "ability_effect": torch.tensor([ability_effect], dtype=torch.long, device=device),
            "stat_mods": stat_mods.unsqueeze(0).to(device),
        }

    def encode_team(
        self,
        team: Team | PartialTeam,
        device: torch.device | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Encode a complete team into list of tensor dicts.

        Args:
            team: Team or PartialTeam to encode
            device: Device for tensors

        Returns:
            List of 6 pokemon_data dicts for FullTeamEncoder.encode_team()
        """
        device = device or torch.device("cpu")

        if isinstance(team, PartialTeam):
            pokemon_list = team.slots
        else:  # Team
            pokemon_list = team.pokemon

        encoded = []
        for i in range(6):
            if i < len(pokemon_list):
                pokemon = pokemon_list[i]
                # Check if empty slot
                if isinstance(pokemon, PokemonSlot) and pokemon.is_empty():
                    encoded.append(self._empty_pokemon_encoding(device))
                else:
                    encoded.append(self.encode_pokemon(pokemon, device))
            else:
                encoded.append(self._empty_pokemon_encoding(device))

        return encoded

    def _empty_pokemon_encoding(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Create an empty Pokemon encoding (all zeros)."""
        return {
            "species_id": torch.tensor([0], dtype=torch.long, device=device),
            "type1_id": torch.tensor([0], dtype=torch.long, device=device),
            "type2_id": torch.tensor([0], dtype=torch.long, device=device),
            "base_stats": torch.zeros(1, 6, dtype=torch.float32, device=device),
            "move_ids": torch.tensor([[0, 0, 0, 0]], dtype=torch.long, device=device),
            "move_types": torch.tensor([[0, 0, 0, 0]], dtype=torch.long, device=device),
            "move_properties": torch.zeros(1, 4, 20, dtype=torch.float32, device=device),
            "item_id": torch.tensor([0], dtype=torch.long, device=device),
            "item_category": torch.tensor([0], dtype=torch.long, device=device),
            "ability_id": torch.tensor([0], dtype=torch.long, device=device),
            "ability_effect": torch.tensor([0], dtype=torch.long, device=device),
            "stat_mods": torch.zeros(1, 7, dtype=torch.float32, device=device),
        }


class TeamEvaluator(nn.Module):
    """Predicts expected performance of a team.

    Architecture:
    - Encode team using FullTeamEncoder
    - Pass through evaluation head
    - Output: Expected win rate (0-1) and uncertainty

    Can optionally condition on expected opponent distribution
    (e.g., current metagame usage stats).
    """

    def __init__(
        self,
        shared_embeddings: SharedEmbeddings,
        data_loader: PokemonDataLoader | None = None,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()

        self.embeddings = shared_embeddings
        self.hidden_dim = hidden_dim

        # Team encoder
        self.team_encoder = FullTeamEncoder(
            shared_embeddings,
            pokemon_hidden=hidden_dim,
            team_hidden=hidden_dim,
        )

        # Tensor encoder for converting Team/PartialTeam to tensors
        self._data_loader = data_loader
        self._tensor_encoder: TeamTensorEncoder | None = None

        # Evaluation head
        self.eval_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Output heads
        self.win_rate_head = nn.Linear(hidden_dim // 2, 1)  # Expected win rate
        self.uncertainty_head = nn.Linear(hidden_dim // 2, 1)  # Epistemic uncertainty

        # Auxiliary heads for interpretability
        self.archetype_head = nn.Linear(hidden_dim // 2, 10)  # Team archetype
        self.balance_head = nn.Linear(hidden_dim // 2, 3)  # Offense/Balance/Stall

    def forward(
        self,
        team_encoding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate a team.

        Args:
            team_encoding: (batch, hidden_dim) Encoded team from FullTeamEncoder

        Returns:
            Dict with:
            - win_rate: (batch, 1) Expected win rate [0, 1]
            - uncertainty: (batch, 1) Prediction uncertainty
            - archetype_logits: (batch, 10) Team archetype probabilities
            - balance_logits: (batch, 3) Offense/Balance/Stall probs
        """
        hidden = self.eval_head(team_encoding)

        win_rate = torch.sigmoid(self.win_rate_head(hidden))
        uncertainty = F.softplus(self.uncertainty_head(hidden))
        archetype_logits = self.archetype_head(hidden)
        balance_logits = self.balance_head(hidden)

        return {
            "win_rate": win_rate,
            "uncertainty": uncertainty,
            "archetype_logits": archetype_logits,
            "balance_logits": balance_logits,
        }

    @property
    def tensor_encoder(self) -> TeamTensorEncoder:
        """Get or create the tensor encoder."""
        if self._tensor_encoder is None:
            if self._data_loader is None:
                self._data_loader = PokemonDataLoader()
            self._tensor_encoder = TeamTensorEncoder(self._data_loader)
        return self._tensor_encoder

    def set_data_loader(self, data_loader: PokemonDataLoader) -> None:
        """Set the data loader for encoding teams.

        Args:
            data_loader: PokemonDataLoader instance
        """
        self._data_loader = data_loader
        self._tensor_encoder = TeamTensorEncoder(data_loader)

    def evaluate_team(
        self,
        team: Team | PartialTeam,
        device: torch.device | None = None,
    ) -> dict[str, float]:
        """Evaluate a complete team.

        Args:
            team: The team to evaluate (Team or PartialTeam)
            device: Device for computation

        Returns:
            Dict with evaluation metrics
        """
        device = device or next(self.parameters()).device

        # Encode team using TeamTensorEncoder
        team_data = self.tensor_encoder.encode_team(team, device)

        # Use FullTeamEncoder to get team encoding
        team_encoding, _ = self.team_encoder.encode_team(team_data)

        # Get evaluation
        self.eval()
        with torch.no_grad():
            output = self.forward(team_encoding)

        return {
            "win_rate": output["win_rate"].item(),
            "uncertainty": output["uncertainty"].item(),
            "archetype": output["archetype_logits"].argmax(dim=-1).item(),
            "balance": output["balance_logits"].argmax(dim=-1).item(),
        }


class MatchupPredictor(nn.Module):
    """Predicts win probability for a specific matchup.

    Given our team and opponent's team (or partial information),
    predicts win probability. Useful for:
    1. Team preview decisions
    2. Understanding team weaknesses
    3. More accurate team evaluation
    """

    def __init__(
        self,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Combine two team encodings
        self.matchup_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        our_team_encoding: torch.Tensor,
        opp_team_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict win probability for matchup.

        Args:
            our_team_encoding: (batch, hidden_dim) Our team encoding
            opp_team_encoding: (batch, hidden_dim) Opponent team encoding

        Returns:
            (batch, 1) Win probability [0, 1]
        """
        combined = torch.cat([our_team_encoding, opp_team_encoding], dim=-1)
        logit = self.matchup_encoder(combined)
        return torch.sigmoid(logit)


class CoverageAnalyzer(nn.Module):
    """Analyzes type coverage and weaknesses of a team.

    Produces interpretable metrics:
    - Types team can hit super-effectively
    - Types team resists
    - Types team is weak to
    - Overall coverage score
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_types = 18

        # Coverage heads
        self.offensive_coverage = nn.Linear(hidden_dim, self.num_types)
        self.defensive_coverage = nn.Linear(hidden_dim, self.num_types)
        self.weakness_detector = nn.Linear(hidden_dim, self.num_types)

    def forward(
        self,
        team_encoding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Analyze team coverage.

        Args:
            team_encoding: (batch, hidden_dim) Encoded team

        Returns:
            Dict with:
            - offensive: (batch, 18) Offensive coverage per type
            - defensive: (batch, 18) Defensive coverage per type
            - weaknesses: (batch, 18) Weakness severity per type
        """
        return {
            "offensive": torch.sigmoid(self.offensive_coverage(team_encoding)),
            "defensive": torch.sigmoid(self.defensive_coverage(team_encoding)),
            "weaknesses": torch.sigmoid(self.weakness_detector(team_encoding)),
        }


class TeamEvaluatorTrainer:
    """Training loop for the team evaluator.

    Trains on (team, outcome) pairs collected from battles.
    """

    def __init__(
        self,
        evaluator: TeamEvaluator,
        learning_rate: float = 1e-4,
        device: torch.device | None = None,
    ):
        self.evaluator = evaluator
        self.device = device or torch.device("cpu")

        self.optimizer = torch.optim.Adam(
            evaluator.parameters(),
            lr=learning_rate,
        )

        # Training history
        self.train_losses: list[float] = []

    def train_step(
        self,
        team_encodings: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> float:
        """Single training step.

        Args:
            team_encodings: (batch, hidden_dim) Encoded teams
            outcomes: (batch,) Win/loss outcomes (1.0 = win, 0.0 = loss)

        Returns:
            Loss value
        """
        self.evaluator.train()
        self.optimizer.zero_grad()

        # Forward pass
        output = self.evaluator(team_encodings)
        predicted_win_rate = output["win_rate"].squeeze(-1)

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(predicted_win_rate, outcomes)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.train_losses.append(loss_val)

        return loss_val

    def add_game_result(
        self,
        team: Team,
        won: bool,
    ) -> None:
        """Add a game result to the training buffer.

        Args:
            team: The team that was played
            won: Whether we won the game
        """
        # TODO: Implement experience buffer and batch training
        pass

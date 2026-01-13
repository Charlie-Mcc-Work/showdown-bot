"""Team generator for autoregressive team construction.

The generator builds teams one Pokemon at a time, conditioning each
selection on the already-chosen team members. This allows it to:
1. Ensure type coverage
2. Build around specific cores
3. Avoid redundancy
4. Respect team roles (lead, sweeper, wall, etc.)

Architecture:
- Input: Partial team encoding + generation step
- Output: Distribution over next Pokemon/moves/item choices

Two modes of operation:
1. Usage-based generation: Uses Smogon usage stats for realistic teams
2. Neural network generation: Learned policy for more diverse teams
"""

from typing import Any
import logging
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from showdown_bot.ou.shared.embeddings import SharedEmbeddings, EmbeddingConfig
from showdown_bot.ou.shared.encoders import FullTeamEncoder
from showdown_bot.ou.shared.data_loader import UsageStatsLoader, UsageStats
from showdown_bot.ou.teambuilder.team_repr import PartialTeam, PokemonSlot

logger = logging.getLogger(__name__)

# Pokemon with locked Tera types (must use specific type)
LOCKED_TERA_TYPES = {
    "Ogerpon": "Grass",
    "Ogerpon-Teal-Mask": "Grass",
    "Ogerpon-Wellspring": "Water",
    "Ogerpon-Hearthflame": "Fire",
    "Ogerpon-Cornerstone": "Rock",
    "Terapagos": "Stellar",
    "Terapagos-Terastal": "Stellar",
    "Terapagos-Stellar": "Stellar",
}


def _get_locked_tera_type(species: str) -> str | None:
    """Return locked Tera type for Pokemon that require it, or None."""
    return LOCKED_TERA_TYPES.get(species)


class TeamGenerator(nn.Module):
    """Autoregressive team generator.

    Generation proceeds in steps:
    1. Select Pokemon 1 (species + moves + item + ability)
    2. Select Pokemon 2, conditioning on Pokemon 1
    3. ... repeat until team is complete

    Each Pokemon selection is further broken down:
    1.1 Select species (from OU-viable Pokemon, ~100 options)
    1.2 Select moveset (4 moves from species' movepool)
    1.3 Select item
    1.4 Select ability
    1.5 Select nature/EVs (can use templates for simplicity)

    The generator uses a Transformer to encode the partial team and
    produce distributions over each choice. It can also fall back to
    usage-based generation when neural network isn't trained.
    """

    def __init__(
        self,
        shared_embeddings: SharedEmbeddings,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_viable_pokemon: int = 100,  # OU-viable species
        usage_loader: UsageStatsLoader | None = None,
        tier: str = "gen9ou",
    ):
        super().__init__()

        self.embeddings = shared_embeddings
        config = shared_embeddings.config
        self.hidden_dim = hidden_dim
        self.num_viable_pokemon = num_viable_pokemon
        self.tier = tier

        # Team encoder for conditioning
        self.team_encoder = FullTeamEncoder(
            shared_embeddings,
            pokemon_hidden=hidden_dim,
            team_hidden=hidden_dim,
        )

        # Generation step embedding (which slot are we filling?)
        self.step_embed = nn.Embedding(6, hidden_dim)

        # Transformer for autoregressive generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heads for each generation choice
        self.species_head = nn.Linear(hidden_dim, num_viable_pokemon)
        self.move_head = nn.Linear(hidden_dim, config.num_moves)
        self.item_head = nn.Linear(hidden_dim, config.num_items)
        self.ability_head = nn.Linear(hidden_dim, config.num_abilities)
        self.nature_head = nn.Linear(hidden_dim, 25)  # 25 natures
        self.ev_head = nn.Linear(hidden_dim, 50)  # Common EV spreads

        # Viable Pokemon mask (which species are OU-viable)
        self.register_buffer(
            "viable_pokemon_mask",
            torch.ones(num_viable_pokemon, dtype=torch.bool),
        )

        # Viable Pokemon names and IDs (populated from usage data)
        self.viable_pokemon_names: list[str] = []
        self.viable_pokemon_ids: list[int] = []
        self._name_to_idx: dict[str, int] = {}

        # Usage-based generator for fallback/bootstrapping
        self._usage_generator: UsageBasedGenerator | None = None
        if usage_loader is not None:
            self._init_usage_generator(usage_loader)

    def _init_usage_generator(self, usage_loader: UsageStatsLoader) -> None:
        """Initialize the usage-based generator."""
        self._usage_generator = UsageBasedGenerator(
            usage_loader=usage_loader,
            tier=self.tier,
            top_n=self.num_viable_pokemon,
        )

        # Populate viable Pokemon from usage data
        self.viable_pokemon_names = self._usage_generator.get_viable_pokemon_list()
        self._name_to_idx = {
            normalize_name(name): i
            for i, name in enumerate(self.viable_pokemon_names)
        }

    def set_usage_loader(self, usage_loader: UsageStatsLoader) -> None:
        """Set the usage stats loader after initialization.

        Args:
            usage_loader: Loaded UsageStatsLoader instance
        """
        self._init_usage_generator(usage_loader)

    def set_viable_pokemon(self, pokemon_ids: list[int]) -> None:
        """Set which Pokemon are viable for team building.

        Args:
            pokemon_ids: List of species IDs that are OU-viable
        """
        self.viable_pokemon_ids = pokemon_ids[:self.num_viable_pokemon]

    def forward(
        self,
        partial_team: PartialTeam,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate next Pokemon for the team.

        Args:
            partial_team: The current partial team
            device: Device for tensors

        Returns:
            Dict with logits for each choice:
            - species_logits: (num_viable_pokemon,)
            - move_logits: (num_moves,) for each of 4 move slots
            - item_logits: (num_items,)
            - ability_logits: (num_abilities,)
            - etc.
        """
        device = device or next(self.parameters()).device

        # Encode current partial team
        # TODO: Implement actual encoding
        batch_size = 1
        team_encoding = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Get current generation step
        step = partial_team.num_filled()
        step_emb = self.step_embed(torch.tensor([step], device=device))

        # Combine team encoding with step embedding
        context = team_encoding + step_emb

        # Decode to get generation distribution
        # For now, just use a simple forward pass
        # In full implementation, would use proper autoregressive decoding
        hidden = context.unsqueeze(1)  # (batch, 1, hidden)

        # Get output logits
        species_logits = self.species_head(hidden.squeeze(1))
        move_logits = self.move_head(hidden.squeeze(1))
        item_logits = self.item_head(hidden.squeeze(1))
        ability_logits = self.ability_head(hidden.squeeze(1))

        # Mask out already-selected Pokemon
        selected_species = partial_team.get_species_list()
        for species in selected_species:
            # TODO: Map species name to viable_pokemon index and mask
            pass

        return {
            "species_logits": species_logits,
            "move_logits": move_logits,
            "item_logits": item_logits,
            "ability_logits": ability_logits,
        }

    def sample_pokemon(
        self,
        partial_team: PartialTeam,
        temperature: float = 1.0,
        device: torch.device | None = None,
        use_neural: bool = False,
    ) -> PokemonSlot:
        """Sample a complete Pokemon for the next slot.

        Args:
            partial_team: Current partial team
            temperature: Sampling temperature (higher = more random)
            device: Device for tensors
            use_neural: If True, use neural network (requires training).
                       If False, use usage-based generation.

        Returns:
            A complete PokemonSlot
        """
        # Use usage-based generation if available and not using neural
        if not use_neural and self._usage_generator is not None:
            selected = set(
                normalize_name(s.species)
                for s in partial_team.slots
                if s.species is not None
            )
            return self._usage_generator._sample_pokemon(
                selected, partial_team, temperature
            )

        # Neural network based generation
        logits = self.forward(partial_team, device)

        # Sample species
        species_probs = F.softmax(logits["species_logits"] / temperature, dim=-1)

        # Mask out already-selected Pokemon
        selected_species = partial_team.get_species_list()
        for species in selected_species:
            norm_name = normalize_name(species)
            if norm_name in self._name_to_idx:
                idx = self._name_to_idx[norm_name]
                if idx < species_probs.size(-1):
                    species_probs[0, idx] = 0.0

        # Renormalize
        species_probs = species_probs / (species_probs.sum() + 1e-8)

        species_idx = torch.multinomial(species_probs.view(-1), 1).item()

        # Map index to species name
        if species_idx < len(self.viable_pokemon_names):
            species_name = self.viable_pokemon_names[species_idx]
        else:
            # Fallback to a common OU Pokemon not already on the team
            fallback_pokemon = [
                "Great Tusk", "Gholdengo", "Kingambit", "Dragapult", "Heatran",
                "Landorus-Therian", "Gliscor", "Iron Valiant", "Dragonite", "Kyurem"
            ]
            selected = partial_team.get_species_list()
            for fallback in fallback_pokemon:
                if fallback not in selected:
                    species_name = fallback
                    break
            else:
                species_name = "Corviknight"  # Last resort

        # Get usage data for this species for moves/item/ability
        if self._usage_generator is not None:
            stats = self._usage_generator.usage_loader.get_pokemon_usage(species_name)
            moves = self._usage_generator._sample_moves(species_name, stats, temperature)
            item = self._usage_generator._sample_item(species_name, stats, temperature)
            ability = self._usage_generator._sample_ability(species_name, stats, temperature)
            nature, evs = self._usage_generator._sample_spread(species_name, stats, temperature)
            tera_type = self._usage_generator._sample_tera_type(species_name, stats, moves)
        else:
            # No usage data, use generic viable defaults
            moves = ["Protect", "Substitute", "Rest", "Sleep Talk"]
            item = "Leftovers"
            ability = "Pressure"
            nature = "Serious"
            evs = {"hp": 252, "atk": 0, "def": 128, "spa": 0, "spd": 128, "spe": 0}
            # Some Pokemon have locked Tera types
            tera_type = _get_locked_tera_type(species_name) or "Normal"

        return PokemonSlot(
            species=species_name,
            moves=moves,
            item=item,
            ability=ability,
            nature=nature,
            evs=evs,
            tera_type=tera_type,
        )

    def generate_team(
        self,
        temperature: float = 1.0,
        device: torch.device | None = None,
        use_neural: bool = False,
    ) -> PartialTeam:
        """Generate a complete team from scratch.

        Args:
            temperature: Sampling temperature
            device: Device for tensors
            use_neural: If True, use neural network (requires training).
                       If False, use usage-based generation.

        Returns:
            A complete PartialTeam
        """
        # Fast path: use usage-based generator directly
        if not use_neural and self._usage_generator is not None:
            return self._usage_generator.generate_team(temperature)

        # Neural network path
        team = PartialTeam(format=self.tier)

        for i in range(6):
            slot = self.sample_pokemon(team, temperature, device, use_neural)
            team.slots[i] = slot

        return team

    def generate_team_from_usage(self, temperature: float = 1.0) -> PartialTeam:
        """Generate a team using only usage-based statistics.

        This is the recommended way to generate teams before the neural
        network has been trained.

        Args:
            temperature: Sampling temperature

        Returns:
            A complete PartialTeam
        """
        if self._usage_generator is None:
            logger.warning("No usage loader set, using neural generator fallback")
            return self.generate_team(temperature, use_neural=True)

        return self._usage_generator.generate_team(temperature)


class SpeciesSelector(nn.Module):
    """Dedicated module for species selection.

    Uses:
    - Current team composition
    - Type coverage needs
    - Role distribution
    - Usage-based priors

    To select the next species.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_species: int = 100,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_species = num_species

        # Species embeddings
        self.species_embed = nn.Embedding(num_species, hidden_dim)

        # Team context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Selection head
        self.selection_head = nn.Linear(hidden_dim, num_species)

    def forward(
        self,
        team_context: torch.Tensor,
        species_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute species selection logits.

        Args:
            team_context: (batch, hidden_dim) Encoded partial team
            species_mask: (batch, num_species) True for valid selections

        Returns:
            (batch, num_species) Selection logits
        """
        context = self.context_encoder(team_context)
        logits = self.selection_head(context)

        # Mask invalid species
        logits = logits.masked_fill(~species_mask, float("-inf"))

        return logits


class MovesetSelector(nn.Module):
    """Dedicated module for moveset selection.

    Given a species, selects 4 moves from its movepool.
    Uses:
    - Species role (sweeper, wall, support, etc.)
    - Team coverage needs
    - Common sets from usage data
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_moves: int = 800,
        max_movepool: int = 100,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_moves = num_moves
        self.max_movepool = max_movepool

        # Move embeddings
        self.move_embed = nn.Embedding(num_moves, hidden_dim // 2)

        # Context processor
        self.context_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Move selection (4 sequential selections with attention)
        self.move_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )

        self.selection_head = nn.Linear(hidden_dim, num_moves)

    def forward(
        self,
        species_context: torch.Tensor,
        movepool_ids: torch.Tensor,
        movepool_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select moves for a species.

        Args:
            species_context: (batch, hidden_dim) Species + team context
            movepool_ids: (batch, max_movepool) Move IDs in species' movepool
            movepool_mask: (batch, max_movepool) True for valid moves

        Returns:
            (batch, 4, num_moves) Move selection logits for each slot
        """
        # TODO: Implement proper autoregressive move selection
        batch_size = species_context.size(0)
        device = species_context.device

        # Placeholder: return uniform logits
        logits = torch.zeros(batch_size, 4, self.num_moves, device=device)
        return logits


# ============================================================================
# Usage-Based Team Generation
# ============================================================================


def normalize_name(name: str) -> str:
    """Normalize a Pokemon/move/item name for lookup.

    Converts to lowercase, removes spaces and hyphens.
    """
    return name.lower().replace(" ", "").replace("-", "")


# Name formatting lookup for common items, abilities, and moves
NAME_FORMAT_MAP = {
    # Items
    "leftovers": "Leftovers",
    "choicescarf": "Choice Scarf",
    "choiceband": "Choice Band",
    "choicespecs": "Choice Specs",
    "lifeorb": "Life Orb",
    "assaultvest": "Assault Vest",
    "rockyhelmet": "Rocky Helmet",
    "heavydutyboots": "Heavy-Duty Boots",
    "focussash": "Focus Sash",
    "airballoon": "Air Balloon",
    "expertbelt": "Expert Belt",
    "colburberry": "Colbur Berry",
    "sitrusberry": "Sitrus Berry",
    "lumberry": "Lum Berry",
    "shellbell": "Shell Bell",
    "eviolite": "Eviolite",
    "blacksludge": "Black Sludge",
    "loadeddice": "Loaded Dice",
    "clearamulet": "Clear Amulet",
    "covertcloak": "Covert Cloak",
    "throatspray": "Throat Spray",
    "weaknesspolicy": "Weakness Policy",
    "boosterenergy": "Booster Energy",
    "silkscarf": "Silk Scarf",
    "custapberry": "Custap Berry",
    "powerherb": "Power Herb",
    "mentalherb": "Mental Herb",
    "widelens": "Wide Lens",
    "shedshell": "Shed Shell",
    "redcard": "Red Card",
    "chestoberry": "Chesto Berry",
    # Abilities
    "vesselofruin": "Vessel of Ruin",
    "swordofruin": "Sword of Ruin",
    "tabletsofruin": "Tablets of Ruin",
    "beadsofruin": "Beads of Ruin",
    "dauntlessshield": "Dauntless Shield",
    "protosynthesis": "Protosynthesis",
    "quarkdrive": "Quark Drive",
    "multiscale": "Multiscale",
    "clearbody": "Clear Body",
    "weakarmor": "Weak Armor",
    "flashfire": "Flash Fire",
    "goodasgold": "Good as Gold",
    "supremeoverlord": "Supreme Overlord",
    "defiant": "Defiant",
    "regenerator": "Regenerator",
    "roughskin": "Rough Skin",
    "intimidate": "Intimidate",
    "waterabsorb": "Water Absorb",
    "voltabsorb": "Volt Absorb",
    "naturalcure": "Natural Cure",
    "serenegrace": "Serene Grace",
    "moldbreaker": "Mold Breaker",
    "shadowtag": "Shadow Tag",
    "magicguard": "Magic Guard",
    "magicbounce": "Magic Bounce",
    "levitate": "Levitate",
    "imposter": "Imposter",
    "drought": "Drought",
    "drizzle": "Drizzle",
    "sandstream": "Sand Stream",
    "snowwarning": "Snow Warning",
    "teravolt": "Teravolt",
    "turboblaze": "Turboblaze",
    "infiltrator": "Infiltrator",
    "poisontouch": "Poison Touch",
    "embodyaspect": "Embody Aspect",
    "embodyaspectcornerstone": "Embody Aspect (Cornerstone)",
    "embodyaspectwellspring": "Embody Aspect (Wellspring)",
    "embodyaspecthearthflame": "Embody Aspect (Hearthflame)",
    "embodyaspectteal": "Embody Aspect (Teal Mask)",
    "toxicdebris": "Toxic Debris",
    "prankster": "Prankster",
    "sandforce": "Sand Force",
    "ironbarbs": "Iron Barbs",
    "flamebody": "Flame Body",
    "static": "Static",
    # Moves
    "stealthrock": "Stealth Rock",
    "dragondance": "Dragon Dance",
    "swordsdance": "Swords Dance",
    "nastyplot": "Nasty Plot",
    "calmmind": "Calm Mind",
    "bulkup": "Bulk Up",
    "irondefense": "Iron Defense",
    "uturn": "U-turn",
    "voltswitch": "Volt Switch",
    "rapidspin": "Rapid Spin",
    "flipturn": "Flip Turn",
    "closecombat": "Close Combat",
    "bodypress": "Body Press",
    "drainpunch": "Drain Punch",
    "focusblast": "Focus Blast",
    "aurasphere": "Aura Sphere",
    "sacredsword": "Sacred Sword",
    "shadowball": "Shadow Ball",
    "shadowsneak": "Shadow Sneak",
    "shadowclaw": "Shadow Claw",
    "poltergeist": "Poltergeist",
    "bitterblade": "Bitter Blade",
    "flamethrower": "Flamethrower",
    "fireblast": "Fire Blast",
    "overheat": "Overheat",
    "heatwave": "Heat Wave",
    "flashcannon": "Flash Cannon",
    "icywind": "Icy Wind",
    "icebeam": "Ice Beam",
    "blizzard": "Blizzard",
    "iceshard": "Ice Shard",
    "iciclecrash": "Icicle Crash",
    "tripleaxel": "Triple Axel",
    "freezedry": "Freeze-Dry",
    "thunderbolt": "Thunderbolt",
    "thunder": "Thunder",
    "thunderwave": "Thunder Wave",
    "wildcharge": "Wild Charge",
    "voltswitch": "Volt Switch",
    "energyball": "Energy Ball",
    "seedbomb": "Seed Bomb",
    "leafstorm": "Leaf Storm",
    "grassknot": "Grass Knot",
    "gigadrain": "Giga Drain",
    "woodhammer": "Wood Hammer",
    "psychic": "Psychic",
    "psyshock": "Psyshock",
    "futuresight": "Future Sight",
    "expandingforce": "Expanding Force",
    "dracometeor": "Draco Meteor",
    "dragonpulse": "Dragon Pulse",
    "dragonclaw": "Dragon Claw",
    "outrage": "Outrage",
    "scaleshot": "Scale Shot",
    "dragondarts": "Dragon Darts",
    "darkpulse": "Dark Pulse",
    "crunch": "Crunch",
    "knockoff": "Knock Off",
    "suckerpunch": "Sucker Punch",
    "moonblast": "Moonblast",
    "playrough": "Play Rough",
    "dazzlinggleam": "Dazzling Gleam",
    "spiritbreak": "Spirit Break",
    "ironhead": "Iron Head",
    "heavyslam": "Heavy Slam",
    "bulletpunch": "Bullet Punch",
    "gyroball": "Gyro Ball",
    "smartstrike": "Smart Strike",
    "stoneedge": "Stone Edge",
    "rockslide": "Rock Slide",
    "earthquake": "Earthquake",
    "earthpower": "Earth Power",
    "scorchingsands": "Scorching Sands",
    "highhorsepower": "High Horsepower",
    "headlongrush": "Headlong Rush",
    "precipiceblades": "Precipice Blades",
    "sludgebomb": "Sludge Bomb",
    "sludgewave": "Sludge Wave",
    "gunkshot": "Gunk Shot",
    "poisonjab": "Poison Jab",
    "toxicspikes": "Toxic Spikes",
    "spikes": "Spikes",
    "defog": "Defog",
    "roost": "Roost",
    "recover": "Recover",
    "softboiled": "Soft-Boiled",
    "rest": "Rest",
    "sleeptalk": "Sleep Talk",
    "protect": "Protect",
    "substitute": "Substitute",
    "toxic": "Toxic",
    "willowisp": "Will-O-Wisp",
    "thunderwave": "Thunder Wave",
    "taunt": "Taunt",
    "encore": "Encore",
    "trick": "Trick",
    "switcheroo": "Switcheroo",
    "roar": "Roar",
    "whirlwind": "Whirlwind",
    "partingshot": "Parting Shot",
    "teleport": "Teleport",
    "batonpass": "Baton Pass",
    "pursuit": "Pursuit",
    "extremespeed": "Extreme Speed",
    "quickattack": "Quick Attack",
    "aquajet": "Aqua Jet",
    "machpunch": "Mach Punch",
    "jetpunch": "Jet Punch",
    "icespinner": "Ice Spinner",
    "hydrosteam": "Hydro Steam",
    "surf": "Surf",
    "scald": "Scald",
    "hydropump": "Hydro Pump",
    "liquidation": "Liquidation",
    "waterfall": "Waterfall",
    "wavecrash": "Wave Crash",
    "ruination": "Ruination",
    "makeitrain": "Make It Rain",
    "ragingbolt": "Raging Bolt",
    "thunderclap": "Thunderclap",
    "populationbomb": "Population Bomb",
    "tidyup": "Tidy Up",
    "ceaselessedge": "Ceaseless Edge",
    "kowtowcleave": "Kowtow Cleave",
    "psyblade": "Psyblade",
    "collisioncourse": "Collision Course",
    "electrodrift": "Electro Drift",
    "bloodmoon": "Blood Moon",
    "ivycudgel": "Ivy Cudgel",
    "ragefist": "Rage Fist",
    "lastrespects": "Last Respects",
    "terablast": "Tera Blast",
}


def format_name(name: str) -> str:
    """Format a Pokemon/move/item/ability name for display.

    Attempts to use the proper capitalization and spacing.
    """
    normalized = normalize_name(name)

    # Check the lookup table first
    if normalized in NAME_FORMAT_MAP:
        return NAME_FORMAT_MAP[normalized]

    # Fallback: title case each word
    # Split on common word boundaries
    result = name

    # Handle common patterns
    if result.islower() or result.isupper():
        # All lowercase or uppercase - apply title case
        result = result.title()

    # Fix some common patterns
    result = result.replace("_", " ")

    return result


def sample_weighted(items: list[tuple[str, float]], temperature: float = 1.0) -> str:
    """Sample an item from a weighted list.

    Args:
        items: List of (name, weight) tuples
        temperature: Higher = more random, lower = more greedy

    Returns:
        Selected item name
    """
    if not items:
        raise ValueError("Cannot sample from empty list")

    names, weights = zip(*items)
    weights = list(weights)

    # Apply temperature
    if temperature != 1.0:
        weights = [w ** (1.0 / temperature) for w in weights]

    total = sum(weights)
    if total == 0:
        return random.choice(names)

    weights = [w / total for w in weights]
    return random.choices(names, weights=weights, k=1)[0]


def parse_spread(spread_str: str) -> tuple[str, dict[str, int]]:
    """Parse a Smogon spread string like 'Adamant:252/0/0/0/4/252'.

    Args:
        spread_str: Format is "Nature:HP/Atk/Def/SpA/SpD/Spe"

    Returns:
        (nature, evs_dict)
    """
    try:
        nature, evs = spread_str.split(":")
        ev_values = [int(x) for x in evs.split("/")]

        if len(ev_values) != 6:
            return "Serious", {}

        ev_dict = {
            "hp": ev_values[0],
            "atk": ev_values[1],
            "def": ev_values[2],
            "spa": ev_values[3],
            "spd": ev_values[4],
            "spe": ev_values[5],
        }
        return nature, ev_dict
    except (ValueError, IndexError):
        return "Serious", {}


# Common EV spread templates for when usage data is unavailable
DEFAULT_SPREADS = {
    "physical_sweeper": ("Jolly", {"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252}),
    "special_sweeper": ("Timid", {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252}),
    "physical_tank": ("Impish", {"hp": 252, "atk": 0, "def": 252, "spa": 0, "spd": 4, "spe": 0}),
    "special_tank": ("Calm", {"hp": 252, "atk": 0, "def": 4, "spa": 0, "spd": 252, "spe": 0}),
    "balanced": ("Adamant", {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252}),
    "bulky_attacker": ("Adamant", {"hp": 252, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 0}),
}

# Default fallback items when usage data unavailable
DEFAULT_ITEMS = [
    "Leftovers", "Choice Scarf", "Choice Band", "Choice Specs",
    "Life Orb", "Assault Vest", "Rocky Helmet", "Heavy-Duty Boots",
]

# All 18 Pokemon types for tera type selection
ALL_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]


class UsageBasedGenerator:
    """Generates teams based on Smogon usage statistics.

    This provides a simple, effective baseline for team generation
    without requiring neural network training. Teams are built by:
    1. Selecting Pokemon weighted by usage rate
    2. Selecting moves/items/abilities from common sets
    3. Using teammate correlations to build coherent teams
    """

    def __init__(
        self,
        usage_loader: UsageStatsLoader | None = None,
        tier: str = "gen9ou",
        rating: int = 1695,
        top_n: int = 80,  # Consider top N Pokemon as viable
    ):
        """Initialize the usage-based generator.

        Args:
            usage_loader: Pre-loaded UsageStatsLoader, or None to create one
            tier: Format tier (e.g., "gen9ou")
            rating: Minimum rating cutoff for stats
            top_n: Number of top Pokemon to consider viable
        """
        self.tier = tier
        self.rating = rating
        self.top_n = top_n

        # Load usage stats
        if usage_loader is not None:
            self.usage_loader = usage_loader
        else:
            self.usage_loader = UsageStatsLoader()
            self.usage_loader.load(tier=tier, rating=rating)

        # Get viable Pokemon list
        self.viable_pokemon = self.usage_loader.get_top_pokemon(top_n)

        # Build usage weight lookup
        self._usage_weights: dict[str, float] = {}
        for pokemon in self.viable_pokemon:
            stats = self.usage_loader.get_pokemon_usage(pokemon)
            if stats:
                self._usage_weights[pokemon] = stats.usage_percent

    def generate_team(self, temperature: float = 1.0) -> PartialTeam:
        """Generate a complete team.

        Args:
            temperature: Sampling temperature (higher = more random)

        Returns:
            A complete PartialTeam
        """
        team = PartialTeam(format=self.tier)
        selected_species: set[str] = set()

        for i in range(6):
            slot = self._sample_pokemon(
                selected_species,
                team,
                temperature,
            )
            team.slots[i] = slot
            if slot.species:
                selected_species.add(normalize_name(slot.species))

        return team

    def _sample_pokemon(
        self,
        selected: set[str],
        partial_team: PartialTeam,
        temperature: float,
    ) -> PokemonSlot:
        """Sample a single Pokemon with moveset.

        Uses teammate correlations if we already have Pokemon selected.
        """
        # Build candidate weights
        candidates: list[tuple[str, float]] = []

        # Get teammate bonuses from already-selected Pokemon
        teammate_bonuses: dict[str, float] = {}
        for species in selected:
            stats = self.usage_loader.get_pokemon_usage(species)
            if stats and stats.common_teammates:
                for teammate, score in stats.common_teammates.items():
                    if teammate not in selected:
                        teammate_bonuses[teammate] = (
                            teammate_bonuses.get(teammate, 0) + score
                        )

        # Build weighted candidate list
        for pokemon in self.viable_pokemon:
            norm_name = normalize_name(pokemon)
            if norm_name in selected:
                continue

            # Base weight from usage
            weight = self._usage_weights.get(pokemon, 1.0)

            # Teammate bonus
            weight += teammate_bonuses.get(norm_name, 0) * 0.1

            candidates.append((pokemon, weight))

        if not candidates:
            # Fallback if somehow no candidates
            logger.warning("No candidate Pokemon available!")
            return PokemonSlot(
                species="Pikachu",
                moves=["Thunderbolt", "Volt Switch", "Grass Knot", "Surf"],
                item="Light Ball",
                ability="Static",
                nature="Timid",
                evs={"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
                tera_type="Electric",
            )

        # Sample species
        species = sample_weighted(candidates, temperature)

        # Get detailed stats for this Pokemon
        stats = self.usage_loader.get_pokemon_usage(species)

        # Sample moveset
        moves = self._sample_moves(species, stats, temperature)

        # Sample item
        item = self._sample_item(species, stats, temperature)

        # Sample ability
        ability = self._sample_ability(species, stats, temperature)

        # Sample nature/EVs from spread
        nature, evs = self._sample_spread(species, stats, temperature)

        # Select tera type
        tera_type = self._sample_tera_type(species, stats, moves)

        return PokemonSlot(
            species=species,
            moves=moves,
            item=item,
            ability=ability,
            nature=nature,
            evs=evs,
            tera_type=tera_type,
        )

    def _sample_moves(
        self,
        species: str,
        stats: UsageStats | None,
        temperature: float,
    ) -> list[str]:
        """Sample 4 moves for a Pokemon."""
        if not stats or not stats.common_moves:
            # Fallback: return placeholder moves
            return ["Tackle", "Protect", "Rest", "Sleep Talk"]

        # Get moves sorted by usage
        move_items = sorted(
            stats.common_moves.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        selected_moves: list[str] = []
        selected_normalized: set[str] = set()
        available_moves = list(move_items)

        for _ in range(4):
            if not available_moves:
                break

            # Sample a move
            move = sample_weighted(available_moves, temperature)
            formatted = format_name(move)
            selected_moves.append(formatted)
            selected_normalized.add(normalize_name(move))

            # Remove selected move from candidates
            available_moves = [
                (m, w) for m, w in available_moves
                if normalize_name(m) not in selected_normalized
            ]

        # Pad with remaining top moves if needed
        while len(selected_moves) < 4 and move_items:
            for move, _ in move_items:
                if normalize_name(move) not in selected_normalized:
                    selected_moves.append(format_name(move))
                    selected_normalized.add(normalize_name(move))
                    break
            else:
                break

        return selected_moves[:4]

    def _sample_item(
        self,
        species: str,
        stats: UsageStats | None,
        temperature: float,
    ) -> str:
        """Sample an item for a Pokemon."""
        if not stats or not stats.common_items:
            return random.choice(DEFAULT_ITEMS)

        item_items = [(k, v) for k, v in stats.common_items.items() if v > 0]
        if not item_items:
            return random.choice(DEFAULT_ITEMS)

        item = sample_weighted(item_items, temperature)
        return format_name(item)

    def _sample_ability(
        self,
        species: str,
        stats: UsageStats | None,
        temperature: float,
    ) -> str:
        """Sample an ability for a Pokemon."""
        if not stats or not stats.common_abilities:
            # Return a common generic ability instead of placeholder
            return "Pressure"

        ability_items = [(k, v) for k, v in stats.common_abilities.items() if v > 0]
        if not ability_items:
            return format_name(list(stats.common_abilities.keys())[0]) if stats.common_abilities else "Pressure"

        ability = sample_weighted(ability_items, temperature)
        return format_name(ability)

    def _sample_spread(
        self,
        species: str,
        stats: UsageStats | None,
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        """Sample a nature/EV spread for a Pokemon."""
        if not stats or not stats.common_spreads:
            # Use default based on species name heuristics
            return DEFAULT_SPREADS["balanced"]

        spread_items = [(k, v) for k, v in stats.common_spreads.items() if v > 0]
        if not spread_items:
            return DEFAULT_SPREADS["balanced"]

        spread_str = sample_weighted(spread_items, temperature)
        return parse_spread(spread_str)

    def _sample_tera_type(
        self,
        species: str,
        stats: UsageStats | None,
        moves: list[str],
    ) -> str:
        """Select a tera type for a Pokemon.

        Strategy: Often match primary STAB or coverage type.
        Some Pokemon have locked Tera types that must be used.
        """
        # Check for locked Tera types (Ogerpon forms, etc.)
        locked = _get_locked_tera_type(species)
        if locked:
            return locked

        # For now, pick a random type with bias toward common choices
        common_tera = ["Steel", "Fairy", "Water", "Ground", "Ghost", "Flying"]

        # 60% chance to pick from common types, 40% random
        if random.random() < 0.6:
            return random.choice(common_tera)
        return random.choice(ALL_TYPES)

    def get_viable_pokemon_list(self) -> list[str]:
        """Get the list of OU-viable Pokemon."""
        return self.viable_pokemon.copy()

    def get_pokemon_info(self, species: str) -> dict[str, Any]:
        """Get usage info for a specific Pokemon."""
        stats = self.usage_loader.get_pokemon_usage(species)
        if not stats:
            return {}

        return {
            "usage_percent": stats.usage_percent,
            "common_moves": self.usage_loader.get_common_moves(species, 10),
            "common_items": self.usage_loader.get_common_items(species, 5),
            "common_teammates": self.usage_loader.get_common_teammates(species, 5),
        }

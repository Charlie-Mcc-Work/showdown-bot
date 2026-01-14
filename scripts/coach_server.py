#!/usr/bin/env python3
"""
Coaching server that provides move recommendations for Pokemon Showdown.

This server receives battle state from the browser extension and returns
move recommendations from the trained model.

Usage:
    python scripts/coach_server.py

Then install the browser extension from the extension/ folder.
"""

import sys
import warnings
from pathlib import Path

# Suppress ROCm warnings
warnings.filterwarnings("ignore", message=".*hipBLASLt.*")
warnings.filterwarnings("ignore", message=".*Flash attention.*experimental.*")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS

from showdown_bot.models.network import PolicyValueNetwork
from showdown_bot.config import training_config
from showdown_bot.environment.state_encoder import (
    StateEncoder,
    TYPE_TO_IDX,
    STATUS_TO_IDX,
    WEATHER_TO_IDX,
    TERRAIN_TO_IDX,
    SIDE_CONDITION_TO_IDX,
    NUM_TYPES,
    NUM_STATUSES,
    NUM_WEATHERS,
    NUM_TERRAINS,
    NUM_SIDE_CONDITIONS,
    CATEGORY_TO_IDX,
)

app = Flask(__name__)
CORS(app)  # Allow requests from browser extension

# Global model and encoder
model = None
device = None


def load_model():
    """Load the trained model."""
    global model, device

    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Find best checkpoint
    checkpoint_dir = Path("data/checkpoints")
    checkpoint_path = None

    for name in ["best_model.pt", "latest.pt"]:
        path = checkpoint_dir / name
        if path.exists():
            checkpoint_path = path
            break

    if not checkpoint_path:
        print("WARNING: No checkpoint found. Using random model.")
        print("Train a model first: python scripts/train.py -t 50000")
        model = PolicyValueNetwork.from_config().to(device)
        model.eval()
        return

    print(f"Loading model: {checkpoint_path}")
    model = PolicyValueNetwork.from_config().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        stats = checkpoint.get("stats", {})
        print(f"  Trained for: {stats.get('total_timesteps', '?'):,} steps")
        print(f"  Best win rate: {stats.get('best_win_rate', 0) * 100:.1f}%")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully!")


class BrowserStateEncoder:
    """Converts browser extension battle state JSON to model tensors."""

    def __init__(self, device: torch.device):
        self.device = device

    def encode(self, state: dict) -> dict:
        """
        Convert browser battle state to model input tensors.

        Args:
            state: Battle state JSON from browser extension

        Returns:
            Dictionary of tensors ready for the model
        """
        # Encode player team
        player_pokemon = self._encode_team(
            state.get("myTeam", []),
            state.get("myActive"),
            revealed=True
        )

        # Encode opponent team
        opponent_pokemon = self._encode_team(
            state.get("opponentTeam", []),
            state.get("opponentActive"),
            revealed=False
        )

        # Find active indices
        player_active_idx = self._find_active_idx(state.get("myTeam", []))
        opponent_active_idx = self._find_active_idx(state.get("opponentTeam", []))

        # Encode field state
        field_state = self._encode_field(state)

        # Create action mask from available moves/switches
        action_mask = self._create_action_mask(state)

        return {
            "player_pokemon": torch.tensor(player_pokemon, dtype=torch.float32, device=self.device).unsqueeze(0),
            "opponent_pokemon": torch.tensor(opponent_pokemon, dtype=torch.float32, device=self.device).unsqueeze(0),
            "player_active_idx": torch.tensor([player_active_idx], dtype=torch.long, device=self.device),
            "opponent_active_idx": torch.tensor([opponent_active_idx], dtype=torch.long, device=self.device),
            "field_state": torch.tensor(field_state, dtype=torch.float32, device=self.device).unsqueeze(0),
            "action_mask": torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0),
        }

    def _encode_team(self, team: list, active_info: dict | None, revealed: bool) -> np.ndarray:
        """Encode a team of up to 6 Pokemon."""
        encoded = np.zeros((6, StateEncoder.POKEMON_FEATURES), dtype=np.float32)

        for i, pokemon in enumerate(team[:6]):
            is_active = pokemon.get("active", False)
            encoded[i] = self._encode_pokemon(pokemon, is_active, revealed)

        return encoded

    def _encode_pokemon(self, pokemon: dict, is_active: bool, is_revealed: bool) -> np.ndarray:
        """Encode a single Pokemon's features."""
        features: list[float] = []

        # HP fraction
        hp = pokemon.get("hp", 100)
        max_hp = pokemon.get("maxHp", 100)
        hp_fraction = hp / max(max_hp, 1) if hp is not None else 0.0
        features.append(hp_fraction)

        # Parse types from species if not provided
        type1, type2 = self._parse_types(pokemon)

        # Type 1 one-hot
        type1_onehot = [0.0] * NUM_TYPES
        if type1 and type1.lower() in TYPE_TO_IDX:
            type1_onehot[TYPE_TO_IDX[type1.lower()]] = 1.0
        features.extend(type1_onehot)

        # Type 2 one-hot
        type2_onehot = [0.0] * NUM_TYPES
        if type2 and type2.lower() in TYPE_TO_IDX:
            type2_onehot[TYPE_TO_IDX[type2.lower()]] = 1.0
        features.extend(type2_onehot)

        # Status one-hot (+ none)
        status_onehot = [0.0] * (NUM_STATUSES + 1)
        status = pokemon.get("status")
        if status and status.lower() in STATUS_TO_IDX:
            status_onehot[STATUS_TO_IDX[status.lower()]] = 1.0
        else:
            status_onehot[-1] = 1.0  # No status
        features.extend(status_onehot)

        # Stat boosts
        boosts = pokemon.get("boosts", {}) if isinstance(pokemon.get("boosts"), dict) else {}
        for stat in ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]:
            boost = boosts.get(stat, 0)
            features.append(boost / 6.0)

        # Is active
        features.append(1.0 if is_active else 0.0)

        # Is fainted
        fainted = pokemon.get("fainted", False) or hp_fraction == 0
        features.append(1.0 if fainted else 0.0)

        # Is revealed
        features.append(1.0 if is_revealed else 0.0)

        # Opponent modeling features
        moves = pokemon.get("moves", [])
        num_moves_revealed = len(moves)

        # Num moves revealed (normalized 0-1, max 4 moves)
        features.append(num_moves_revealed / 4.0)

        # Could have more moves (1 if <4 revealed)
        features.append(1.0 if num_moves_revealed < 4 else 0.0)

        # Ability revealed (browser may not track this)
        ability_revealed = pokemon.get("ability") is not None
        features.append(1.0 if ability_revealed else 0.0)

        # Item revealed
        item = pokemon.get("item")
        item_revealed = item is not None and item != ""
        features.append(1.0 if item_revealed else 0.0)

        # Item consumed/knocked off
        item_consumed = item == ""
        features.append(1.0 if item_consumed else 0.0)

        # Encode moves (4 slots) with is_revealed flag
        for i in range(4):
            if i < len(moves):
                features.extend(self._encode_move(moves[i], is_revealed=True))
            else:
                # Empty move slot - all zeros including is_revealed=0
                features.extend([0.0] * StateEncoder.MOVE_FEATURES)

        return np.array(features, dtype=np.float32)

    def _encode_move(self, move: str | dict, is_revealed: bool = True) -> list[float]:
        """Encode a single move (simplified - uses name heuristics).

        Args:
            move: Move data (string name or dict with details)
            is_revealed: Whether this move has been revealed to us
        """
        features: list[float] = []

        # Type one-hot - we'd need a move database for accuracy
        # For now, use a placeholder
        type_onehot = [0.0] * NUM_TYPES
        features.extend(type_onehot)

        # Category one-hot (placeholder)
        category_onehot = [0.0] * 3
        category_onehot[0] = 1.0  # Default to physical
        features.extend(category_onehot)

        # Base power (placeholder)
        features.append(0.4)  # Assume medium power

        # PP fraction
        if isinstance(move, dict):
            pp = move.get("pp", 10)
            max_pp = move.get("maxPp", 10)
            pp_fraction = pp / max(max_pp, 1) if pp is not None else 1.0
        else:
            pp_fraction = 1.0
        features.append(pp_fraction)

        # Is revealed flag (for opponent modeling)
        features.append(1.0 if is_revealed else 0.0)

        return features

    def _parse_types(self, pokemon: dict) -> tuple[str | None, str | None]:
        """Parse Pokemon types (would need a pokedex for accuracy)."""
        # The browser extension doesn't always provide types
        # In a full implementation, we'd look up types from species
        return None, None

    def _encode_field(self, state: dict) -> np.ndarray:
        """Encode field conditions."""
        features: list[float] = []

        # Weather one-hot (+ none)
        weather_onehot = [0.0] * (NUM_WEATHERS + 1)
        weather = state.get("weather")
        if weather and weather.lower() in WEATHER_TO_IDX:
            weather_onehot[WEATHER_TO_IDX[weather.lower()]] = 1.0
        else:
            weather_onehot[-1] = 1.0
        features.extend(weather_onehot)

        # Terrain one-hot (+ none)
        terrain_onehot = [0.0] * (NUM_TERRAINS + 1)
        terrain = state.get("terrain")
        if terrain and terrain.lower() in TERRAIN_TO_IDX:
            terrain_onehot[TERRAIN_TO_IDX[terrain.lower()]] = 1.0
        else:
            terrain_onehot[-1] = 1.0
        features.extend(terrain_onehot)

        # Player side conditions (placeholder - browser doesn't track well)
        features.extend([0.0] * NUM_SIDE_CONDITIONS)

        # Opponent side conditions
        features.extend([0.0] * NUM_SIDE_CONDITIONS)

        # Trick room
        features.append(0.0)

        # Turn count normalized
        turn = state.get("turn", 1)
        features.append(min(turn / 100.0, 1.0))

        # Opponent modeling - how many opponent Pokemon have we seen?
        opponent_team = state.get("opponentTeam", [])
        num_opponent_revealed = len(opponent_team)
        max_opponent_pokemon = 6

        # Num opponent Pokemon revealed (normalized 0-1)
        features.append(num_opponent_revealed / max_opponent_pokemon)

        # Num opponent Pokemon unrevealed (normalized 0-1)
        num_unrevealed = max_opponent_pokemon - num_opponent_revealed
        features.append(num_unrevealed / max_opponent_pokemon)

        return np.array(features, dtype=np.float32)

    def _find_active_idx(self, team: list) -> int:
        """Find the index of the active Pokemon."""
        for i, pokemon in enumerate(team):
            if pokemon.get("active", False):
                return i
        return 0

    def _create_action_mask(self, state: dict) -> np.ndarray:
        """Create action mask from available moves and switches."""
        mask = np.zeros(StateEncoder.NUM_ACTIONS, dtype=np.float32)

        # Available moves
        available_moves = state.get("availableMoves", [])
        for i, move in enumerate(available_moves[:4]):
            if not move.get("disabled", False):
                mask[i] = 1.0

        # Available switches (indices 4-8)
        available_switches = state.get("availableSwitches", [])
        for i, switch in enumerate(available_switches[:5]):
            mask[4 + i] = 1.0

        # Ensure at least one action is available
        if mask.sum() == 0:
            mask[0] = 1.0

        return mask


def get_move_recommendations(state: dict) -> list:
    """
    Get move recommendations from the trained model.

    Returns a list of moves with their scores.
    """
    global model, device

    if model is None:
        return _get_heuristic_recommendations(state)

    encoder = BrowserStateEncoder(device)

    try:
        # Encode state to tensors
        tensors = encoder.encode(state)

        # Run model inference
        with torch.no_grad():
            logits, value = model.forward(
                tensors["player_pokemon"],
                tensors["opponent_pokemon"],
                tensors["player_active_idx"],
                tensors["opponent_active_idx"],
                tensors["field_state"],
                tensors["action_mask"],
            )

            # Get probabilities
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            value_estimate = value.cpu().item()

        # Build recommendations
        recommendations = []
        available_moves = state.get("availableMoves", [])
        available_switches = state.get("availableSwitches", [])

        # Move recommendations (indices 0-3)
        for i, move in enumerate(available_moves[:4]):
            if not move.get("disabled", False):
                recommendations.append({
                    "name": move.get("name", f"Move {i+1}"),
                    "id": move.get("id", f"move{i}"),
                    "score": float(probs[i]),
                    "index": i,
                    "type": "move"
                })

        # Switch recommendations (indices 4-8)
        for i, switch in enumerate(available_switches[:5]):
            recommendations.append({
                "name": f"Switch to {switch.get('species', 'Pokemon')}",
                "id": f"switch{i}",
                "score": float(probs[4 + i]),
                "index": 4 + i,
                "type": "switch"
            })

        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        # Add value estimate
        if recommendations:
            recommendations[0]["value"] = value_estimate

        return recommendations[:4]

    except Exception as e:
        print(f"Model inference error: {e}")
        import traceback
        traceback.print_exc()
        return _get_heuristic_recommendations(state)


def _get_heuristic_recommendations(state: dict) -> list:
    """Fallback heuristic recommendations when model fails."""
    recommendations = []

    my_active = state.get("myActive") or {}
    opp_active = state.get("opponentActive") or {}
    moves = state.get("availableMoves", [])
    switches = state.get("availableSwitches", [])

    # Score each move
    for move in moves:
        if move.get("disabled"):
            continue

        score = 0.5  # Base score
        move_name = move.get("name", "").lower()

        # Prefer attacking moves
        if move.get("pp", 0) > 0:
            score += 0.1

        # Low HP? Prioritize priority moves
        my_hp = my_active.get("hp", 100)
        my_max_hp = my_active.get("maxHp", 100)
        my_hp_pct = my_hp / max(my_max_hp, 1)

        if my_hp_pct < 0.3:
            priority_moves = ["quick", "mach", "bullet", "aqua jet", "sucker", "shadow sneak", "extreme", "ice shard"]
            if any(p in move_name for p in priority_moves):
                score += 0.2

        # Status moves
        status_moves = ["thunder wave", "toxic", "will-o-wisp", "spore", "sleep powder"]
        if any(s in move_name for s in status_moves):
            if not opp_active.get("status"):
                score += 0.15

        # Setup moves when healthy
        if my_hp_pct > 0.7:
            setup_moves = ["sword", "dance", "nasty plot", "calm mind", "dragon dance", "quiver", "bulk up"]
            if any(s in move_name for s in setup_moves):
                score += 0.15

        recommendations.append({
            "name": move.get("name", "Unknown"),
            "id": move.get("id"),
            "score": min(score, 1.0),
            "index": move.get("index", 0),
            "type": "move"
        })

    # Sort by score
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    # Add switch options
    for switch in switches[:2]:
        recommendations.append({
            "name": f"Switch to {switch.get('species', 'Pokemon')}",
            "id": f"switch{switch.get('index', 0)}",
            "score": 0.3,
            "index": switch.get("index", 0) + 4,
            "type": "switch"
        })

    return recommendations[:4]


@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint for the browser extension to get move recommendations."""
    try:
        state = request.json

        if not state:
            return jsonify({"error": "No state provided"}), 400

        if not state.get("waitingForChoice"):
            return jsonify({"moves": [], "message": "Not your turn"})

        recommendations = get_move_recommendations(state)

        return jsonify({
            "moves": recommendations,
            "turn": state.get("turn", 0),
            "model_active": model is not None
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device else "none"
    })


def main():
    print("=" * 60)
    print("Pokemon Showdown Coach Server")
    print("=" * 60)

    load_model()

    print()
    print("Starting server on http://localhost:5000")
    print()
    print("To use the coach:")
    print("1. Install the browser extension from extension/")
    print("2. Go to https://play.pokemonshowdown.com")
    print("3. Start a battle - recommendations will appear!")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host='localhost', port=5000, debug=False)


if __name__ == "__main__":
    main()

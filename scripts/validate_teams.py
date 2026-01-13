#!/usr/bin/env python3
"""Validate teams against Pokemon Showdown's team validator.

Usage:
    python scripts/validate_teams.py [team_file] [format]
    python scripts/validate_teams.py  # validates default gen9ou teams
    python scripts/validate_teams.py data/sample_teams/gen9ou.txt gen9ou
"""

import subprocess
import sys
from pathlib import Path

# Pokemon Showdown installation path
PS_PATH = Path.home() / "pokemon-showdown"


def parse_teams_file(filepath: Path) -> list[tuple[str, str]]:
    """Parse a teams file into list of (team_name, team_text) tuples."""
    content = filepath.read_text()
    teams = []

    current_name = None
    current_lines = []

    for line in content.split("\n"):
        if line.startswith("==="):
            # Save previous team
            if current_name and current_lines:
                teams.append((current_name, "\n".join(current_lines)))

            # Start new team
            current_name = line.strip("= ")
            current_lines = []
        else:
            current_lines.append(line)

    # Save last team
    if current_name and current_lines:
        teams.append((current_name, "\n".join(current_lines)))

    return teams


def validate_team(team_text: str, format_id: str = "gen9ou") -> tuple[bool, str]:
    """Validate a team using Pokemon Showdown's validator.

    Args:
        team_text: Team in Pokemon Showdown paste format
        format_id: Battle format (e.g., 'gen9ou', 'gen9randombattle')

    Returns:
        (is_valid, error_message) tuple
    """
    if not PS_PATH.exists():
        return False, f"Pokemon Showdown not found at {PS_PATH}"

    # Use Pokemon Showdown's validate-team command
    cmd = [str(PS_PATH / "pokemon-showdown"), "validate-team", format_id]

    try:
        result = subprocess.run(
            cmd,
            input=team_text,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PS_PATH,
        )

        # Empty output means valid
        if not result.stdout.strip() and result.returncode == 0:
            return True, ""

        # Output contains validation errors
        return False, result.stdout.strip() or result.stderr.strip()

    except subprocess.TimeoutExpired:
        return False, "Validation timed out"
    except FileNotFoundError:
        return False, "Node.js or pokemon-showdown not found"
    except Exception as e:
        return False, f"Validation error: {e}"


def validate_teams_file(filepath: Path, format_id: str = "gen9ou") -> dict:
    """Validate all teams in a file.

    Returns:
        Dict with 'valid', 'invalid', and 'errors' keys
    """
    teams = parse_teams_file(filepath)
    results = {
        "valid": [],
        "invalid": [],
        "errors": {},
    }

    for name, team_text in teams:
        is_valid, error = validate_team(team_text, format_id)
        if is_valid:
            results["valid"].append(name)
        else:
            results["invalid"].append(name)
            results["errors"][name] = error

    return results


def main():
    # Default paths
    default_file = Path(__file__).parent.parent / "data" / "sample_teams" / "gen9ou.txt"
    default_format = "gen9ou"

    # Parse args
    if len(sys.argv) >= 2:
        team_file = Path(sys.argv[1])
    else:
        team_file = default_file

    if len(sys.argv) >= 3:
        format_id = sys.argv[2]
    else:
        format_id = default_format

    if not team_file.exists():
        print(f"Error: Team file not found: {team_file}")
        sys.exit(1)

    print(f"Validating teams from: {team_file}")
    print(f"Format: {format_id}")
    print("-" * 50)

    results = validate_teams_file(team_file, format_id)

    # Print results
    print(f"\nValid teams ({len(results['valid'])}):")
    for name in results["valid"]:
        print(f"  ✓ {name}")

    if results["invalid"]:
        print(f"\nInvalid teams ({len(results['invalid'])}):")
        for name in results["invalid"]:
            print(f"  ✗ {name}")
            error = results["errors"][name]
            for line in error.split("\n"):
                if line.strip():
                    print(f"    - {line.strip()}")

    print("-" * 50)
    print(f"Total: {len(results['valid'])}/{len(results['valid']) + len(results['invalid'])} valid")

    # Exit with error if any invalid
    if results["invalid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

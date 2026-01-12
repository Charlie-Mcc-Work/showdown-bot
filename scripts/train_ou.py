#!/usr/bin/env python3
"""Training script for Pokemon Showdown Gen 9 OU RL bot.

This script supports three training modes:
1. Player mode: Train the battle decision network using self-play
2. Teambuilder mode: Train the team generator using battle outcomes
3. Joint mode: Alternating player/teambuilder training (future)

Unlike random battles, OU requires:
- Pre-built teams (loaded from data/sample_teams/)
- Team preview / lead selection
- Terastallization decisions

Usage:
    # Player training (default)
    python scripts/train_ou.py                      # Start fresh training
    python scripts/train_ou.py --resume             # Resume from checkpoint
    python scripts/train_ou.py -t 100000            # Train for 100k steps
    python scripts/train_ou.py --num-envs 4         # Use 4 parallel envs

    # Teambuilder training (requires trained player)
    python scripts/train_ou.py --mode teambuilder --player-checkpoint best_player.pt
    python scripts/train_ou.py -m teambuilder --total-teams 500

    # Joint training (not fully implemented)
    python scripts/train_ou.py --mode joint
"""

import argparse
import asyncio
import gc
import logging
import signal
import sys
import warnings
from pathlib import Path
from datetime import datetime

# Suppress ROCm/PyTorch experimental feature warnings
warnings.filterwarnings("ignore", message=".*hipBLASLt.*")
warnings.filterwarnings("ignore", message=".*Flash attention.*experimental.*")
warnings.filterwarnings("ignore", message=".*Memory Efficient attention.*experimental.*")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from poke_env.player import RandomPlayer
from torch.utils.tensorboard import SummaryWriter

from showdown_bot.ou.player.network import OUPlayerNetwork
from showdown_bot.ou.player.state_encoder import OUStateEncoder
from showdown_bot.ou.player.ou_player import (
    OUTrainablePlayer,
    OUNeuralNetworkPlayer,
    OUMaxDamagePlayer,
)
from showdown_bot.ou.training.config import OUTrainingConfig
from showdown_bot.ou.training.player_trainer import OUPlayerTrainer
from showdown_bot.ou.training.self_play import OUSelfPlayManager
from showdown_bot.ou.training.teambuilder_trainer import TeambuilderTrainer
from showdown_bot.ou.training.joint_trainer import JointTrainer, create_joint_trainer
from showdown_bot.ou.training.curriculum import (
    CurriculumConfig,
    OpponentType,
    create_curriculum,
)
from showdown_bot.ou.teambuilder.generator import TeamGenerator
from showdown_bot.ou.teambuilder.evaluator import TeamEvaluator
from showdown_bot.ou.shared.data_loader import TeamLoader
from showdown_bot.ou.shared.embeddings import SharedEmbeddings, EmbeddingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class OUTrainingManager:
    """Manages the OU training loop."""

    def __init__(
        self,
        config: OUTrainingConfig,
        device: torch.device,
        teams_dir: str = "data/sample_teams",
        checkpoint_dir: str = "data/checkpoints/ou",
        log_dir: str = "runs/ou",
        num_envs: int = 1,
        use_self_play: bool = True,
    ):
        self.config = config
        self.device = device
        self.num_envs = num_envs
        self.use_self_play = use_self_play
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load teams
        logger.info(f"Loading teams from {teams_dir}...")
        self.team_loader = TeamLoader(teams_dir)
        self.teams = self.team_loader.load_teams("gen9ou")
        logger.info(f"Loaded {len(self.teams)} teams")

        if not self.teams:
            raise ValueError(f"No teams found in {teams_dir}. Add teams to data/sample_teams/gen9ou.txt")

        # Create network and encoder
        logger.info("Initializing neural network...")
        self.encoder = OUStateEncoder(device=device)
        self.network = OUPlayerNetwork(
            pokemon_dim=self.encoder.POKEMON_FEATURES,
            field_dim=self.encoder.FIELD_FEATURES,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            num_actions=self.encoder.NUM_ACTIONS,
        ).to(device)

        num_params = sum(p.numel() for p in self.network.parameters())
        logger.info(f"Model parameters: {num_params:,}")

        # Create trainer
        self.trainer = OUPlayerTrainer(
            network=self.network,
            encoder=self.encoder,
            config=config,
            device=device,
        )
        self.trainer.setup_logging(log_dir)

        # Self-play manager
        self.self_play_manager: OUSelfPlayManager | None = None
        if use_self_play:
            self.self_play_manager = OUSelfPlayManager(
                pool_dir=self.checkpoint_dir / "opponent_pool",
                teams=self.teams,
                max_pool_size=10,
                self_play_ratio=0.7,
                checkpoint_interval=50000,
                sampling_strategy="skill_matched",
                device=device,
            )
            logger.info("Self-play enabled with opponent pool")

        # Training state
        self.total_timesteps = 0
        self.total_episodes = 0
        self.should_stop = False

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("\nShutdown requested, finishing current episode...")
        self.should_stop = True

    async def train(
        self,
        total_timesteps: int,
        eval_interval: int = 10000,
        save_interval: int = 50000,
    ) -> None:
        """Main training loop.

        Args:
            total_timesteps: Total timesteps to train
            eval_interval: Steps between evaluations
            save_interval: Steps between checkpoint saves
        """
        logger.info(f"Starting OU training for {total_timesteps:,} timesteps")
        logger.info(f"Using {self.num_envs} parallel environment(s)")
        if self.self_play_manager:
            logger.info(f"Self-play ratio: {self.self_play_manager.self_play_ratio:.0%}")

        start_time = datetime.now()
        last_eval = self.total_timesteps
        last_save = self.total_timesteps

        while self.total_timesteps < total_timesteps and not self.should_stop:
            # Add checkpoint to opponent pool if needed
            if self.self_play_manager and self.self_play_manager.should_add_checkpoint(self.total_timesteps):
                self.self_play_manager.add_checkpoint(self.network, self.total_timesteps)

            # Collect experience from battles
            steps, episodes = await self._collect_rollout()
            self.total_timesteps += steps
            self.total_episodes += episodes

            # PPO update
            if len(self.trainer.buffer) >= self.config.player_batch_size:
                metrics = self.trainer.update()

                # Log progress
                elapsed = (datetime.now() - start_time).total_seconds()
                steps_per_sec = self.total_timesteps / max(elapsed, 1)

                # Build log message
                log_parts = [
                    f"Step {self.total_timesteps:,}",
                    f"Episodes: {self.total_episodes:,}",
                    f"Win rate: {self.trainer.win_rate:.1%}",
                ]

                if self.self_play_manager:
                    stats = self.self_play_manager.get_stats()
                    log_parts.append(f"Skill: {stats['agent_skill']:.0f}")
                    log_parts.append(f"Pool: {stats['pool_size']}")
                else:
                    log_parts.append(f"Skill: {self.trainer.skill_rating:.0f}")

                log_parts.append(f"Speed: {steps_per_sec:.1f} steps/s")
                logger.info(" | ".join(log_parts))

                # Log self-play stats to TensorBoard
                if self.self_play_manager and self.trainer.writer:
                    stats = self.self_play_manager.get_stats()
                    self.trainer.writer.add_scalar(
                        "self_play/agent_skill", stats["agent_skill"], self.total_timesteps
                    )
                    self.trainer.writer.add_scalar(
                        "self_play/pool_size", stats["pool_size"], self.total_timesteps
                    )
                    self.trainer.writer.add_scalar(
                        "self_play/ratio_actual", stats["self_play_ratio_actual"], self.total_timesteps
                    )

            # Periodic evaluation
            if self.total_timesteps - last_eval >= eval_interval:
                await self._evaluate()
                last_eval = self.total_timesteps

            # Periodic checkpoint
            if self.total_timesteps - last_save >= save_interval:
                self._save_checkpoint()
                last_save = self.total_timesteps

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final save
        self._save_checkpoint(final=True)
        logger.info(f"Training complete! Total episodes: {self.total_episodes:,}")

    async def _collect_rollout(self) -> tuple[int, int]:
        """Collect experience from battles.

        Returns:
            (steps_collected, episodes_completed)
        """
        steps = 0
        episodes = 0

        # Create players for this rollout
        players: list[OUTrainablePlayer] = []
        opponents = []
        opponent_infos = []  # Track self-play opponent info for rating updates

        import random
        for i in range(self.num_envs):
            # Pick a random team for this player
            team = random.choice(self.teams) if self.teams else None

            player = OUTrainablePlayer(
                network=self.network,
                state_encoder=self.encoder,
                device=self.device,
                teams=self.teams,
                current_team=team,
                battle_format="gen9ou",
                max_concurrent_battles=1,
            )
            players.append(player)

            # Get opponent from self-play manager or baseline
            if self.self_play_manager:
                opponent, opponent_info, opponent_type = self.self_play_manager.get_opponent(
                    battle_format="gen9ou"
                )
                opponents.append(opponent)
                opponent_infos.append((opponent_info, opponent_type))
            else:
                # Fallback: Random opponent selection (mix of strategies)
                if random.random() < 0.5:
                    opponent = RandomPlayer(
                        battle_format="gen9ou",
                        max_concurrent_battles=1,
                    )
                    opponent_infos.append((None, "random"))
                else:
                    opponent = OUMaxDamagePlayer(
                        teams=self.teams,
                        battle_format="gen9ou",
                        max_concurrent_battles=1,
                    )
                    opponent_infos.append((None, "maxdamage"))
                opponents.append(opponent)

        try:
            # Run battles concurrently
            battle_tasks = []
            for player, opponent in zip(players, opponents):
                task = asyncio.create_task(
                    self._run_battle(player, opponent)
                )
                battle_tasks.append(task)

            results = await asyncio.gather(*battle_tasks, return_exceptions=True)

            # Process results and collect experiences
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Battle error: {result}")
                    continue
                battle_steps, won = result
                steps += battle_steps
                episodes += 1

                # Update self-play stats
                opponent_info, opponent_type = opponent_infos[i]
                if self.self_play_manager:
                    self.self_play_manager.update_after_game(
                        opponent_info, won, opponent_type
                    )

                # Collect experiences from this player
                player = players[i]
                experiences = player.get_experiences()

                for j, exp in enumerate(experiences):
                    # Get next state (None if last in episode)
                    next_state = None
                    if j < len(experiences) - 1:
                        next_state = experiences[j + 1]["state"]

                    self.trainer.add_transition(
                        state=exp["state"],
                        action=exp["action"],
                        reward=exp["reward"],
                        next_state=next_state,
                        done=exp["done"],
                        log_prob=exp["log_prob"],
                        value=exp["value"],
                        team=exp.get("team"),
                    )

        finally:
            # Cleanup players
            await self._cleanup_players(players + opponents)

        return steps, episodes

    async def _run_battle(
        self,
        player: OUTrainablePlayer,
        opponent,
    ) -> tuple[int, bool]:
        """Run a single battle and collect experience.

        The OUTrainablePlayer automatically collects experiences during
        choose_move() and handles terminal rewards in _battle_finished_callback().

        Returns:
            (steps, won)
        """
        # Start battle
        await player.battle_against(opponent, n_battles=1)

        # Get final battle for result
        if not player.battles:
            return 0, False

        battle = list(player.battles.values())[-1]
        won = battle.won if battle.won is not None else False

        # Update trainer stats
        self.trainer.update_skill_rating(won, 1000.0)

        return max(battle.turn, 1), won

    async def _cleanup_players(self, players: list) -> None:
        """Clean up player connections."""
        cleanup_tasks = []
        for player in players:
            try:
                if hasattr(player, 'ps_client') and player.ps_client is not None:
                    cleanup_tasks.append(player.ps_client.stop_listening())
            except Exception:
                pass

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            await asyncio.sleep(0.5)

        # Free memory
        for player in players:
            if hasattr(player, 'network'):
                player.network = None
            if hasattr(player, 'state_encoder'):
                player.state_encoder = None

    async def _evaluate(self, n_games: int = 20) -> dict:
        """Evaluate current policy against baselines.

        Args:
            n_games: Number of games per opponent type

        Returns:
            Evaluation metrics
        """
        logger.info("Running evaluation...")

        results = {"vs_random": 0, "vs_maxdamage": 0}

        # Create evaluation player
        eval_player = OUNeuralNetworkPlayer(
            network=self.network,
            state_encoder=self.encoder,
            device=self.device,
            deterministic=True,
            teams=self.teams,
            battle_format="gen9ou",
            max_concurrent_battles=1,
        )

        try:
            # vs Random
            random_opponent = RandomPlayer(
                battle_format="gen9ou",
                max_concurrent_battles=1,
            )
            await eval_player.battle_against(random_opponent, n_battles=n_games)

            wins = sum(1 for b in eval_player.battles.values() if b.won)
            results["vs_random"] = wins / n_games
            logger.info(f"  vs Random: {wins}/{n_games} ({results['vs_random']:.1%})")

            # Clear battles for next evaluation
            eval_player._battles = {}

            # vs MaxDamage
            maxdmg_opponent = OUMaxDamagePlayer(
                teams=self.teams,
                battle_format="gen9ou",
                max_concurrent_battles=1,
            )
            await eval_player.battle_against(maxdmg_opponent, n_battles=n_games)

            wins = sum(1 for b in eval_player.battles.values() if b.won)
            results["vs_maxdamage"] = wins / n_games
            logger.info(f"  vs MaxDamage: {wins}/{n_games} ({results['vs_maxdamage']:.1%})")

        finally:
            await self._cleanup_players([eval_player, random_opponent, maxdmg_opponent])

        # Log to TensorBoard
        if self.trainer.writer:
            self.trainer.writer.add_scalar(
                "eval/vs_random", results["vs_random"], self.total_timesteps
            )
            self.trainer.writer.add_scalar(
                "eval/vs_maxdamage", results["vs_maxdamage"], self.total_timesteps
            )

        return results

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save a training checkpoint."""
        checkpoint_path = self.checkpoint_dir / "latest.pt"

        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "total_wins": self.trainer.total_wins,
            "total_losses": self.trainer.total_losses,
            "skill_rating": self.trainer.skill_rating,
            "config": self.config.to_dict(),
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Also save periodic checkpoint
        if not final:
            periodic_path = self.checkpoint_dir / f"checkpoint_{self.total_timesteps}.pt"
            torch.save(checkpoint, periodic_path)

        # Save best model based on win rate
        if self.trainer.win_rate >= 0.5:
            best_path = self.checkpoint_dir / "best_model.pt"
            if not best_path.exists():
                torch.save(checkpoint, best_path)
                logger.info(f"Saved new best model with {self.trainer.win_rate:.1%} win rate")

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        self.total_episodes = checkpoint.get("total_episodes", 0)
        self.trainer.total_wins = checkpoint.get("total_wins", 0)
        self.trainer.total_losses = checkpoint.get("total_losses", 0)
        self.trainer.skill_rating = checkpoint.get("skill_rating", 1000.0)

        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"  Timesteps: {self.total_timesteps:,}")
        logger.info(f"  Episodes: {self.total_episodes:,}")
        logger.info(f"  Win rate: {self.trainer.win_rate:.1%}")


class TeambuilderTrainingManager:
    """Manages teambuilder training with battle evaluation.

    This trains the teambuilder by:
    1. Generating teams using the TeamGenerator
    2. Playing battles with generated teams using a frozen player
    3. Using win rates to update the teambuilder via policy gradient
    """

    def __init__(
        self,
        config: OUTrainingConfig,
        device: torch.device,
        player_checkpoint: str | None = None,
        teams_dir: str = "data/sample_teams",
        checkpoint_dir: str = "data/checkpoints/ou/teambuilder",
        log_dir: str = "runs/ou/teambuilder",
    ):
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load sample teams for initial training
        logger.info(f"Loading sample teams from {teams_dir}...")
        self.team_loader = TeamLoader(teams_dir)
        self.sample_teams = self.team_loader.load_teams("gen9ou")
        logger.info(f"Loaded {len(self.sample_teams)} sample teams for supervised learning")

        # Create shared embeddings
        logger.info("Initializing embeddings and teambuilder...")
        emb_config = EmbeddingConfig()
        self.embeddings = SharedEmbeddings(emb_config)

        # Create teambuilder components
        self.generator = TeamGenerator(
            shared_embeddings=self.embeddings,
            hidden_dim=512,
            num_heads=8,
            num_layers=4,
        ).to(device)

        self.evaluator = TeamEvaluator(
            shared_embeddings=self.embeddings,
            hidden_dim=512,
        ).to(device)

        # Create trainer
        self.trainer = TeambuilderTrainer(
            generator=self.generator,
            evaluator=self.evaluator,
            config=config,
            device=device,
        )
        self.trainer.setup_logging(log_dir)
        self.trainer.add_sample_teams(self.sample_teams)

        # Load player for evaluation (if checkpoint provided)
        self.player_network: OUPlayerNetwork | None = None
        self.player_encoder: OUStateEncoder | None = None
        if player_checkpoint:
            self._load_player(player_checkpoint)

        # Training state
        self.total_teams_generated = 0
        self.total_battles = 0
        self.should_stop = False

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info("\nShutdown requested...")
        self.should_stop = True

    def _load_player(self, checkpoint_path: str) -> None:
        """Load player network for team evaluation."""
        logger.info(f"Loading player from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.player_encoder = OUStateEncoder(device=self.device)
        self.player_network = OUPlayerNetwork(
            pokemon_dim=self.player_encoder.POKEMON_FEATURES,
            field_dim=self.player_encoder.FIELD_FEATURES,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            num_actions=self.player_encoder.NUM_ACTIONS,
        ).to(self.device)

        self.player_network.load_state_dict(checkpoint["model_state_dict"])
        self.player_network.eval()
        logger.info("Player loaded successfully")

    async def train(
        self,
        total_teams: int = 1000,
        games_per_team: int = 10,
        save_interval: int = 100,
    ) -> None:
        """Train the teambuilder.

        Args:
            total_teams: Total teams to generate and evaluate
            games_per_team: Games to play per generated team
            save_interval: Save checkpoint every N teams
        """
        logger.info(f"Starting teambuilder training for {total_teams} teams")
        logger.info(f"Playing {games_per_team} games per team")

        # NOTE: The teambuilder components are currently placeholders.
        # This training loop is ready for when the generator is fully implemented.
        if not self.player_network:
            logger.warning("No player checkpoint provided - using random evaluation")
            logger.warning("Teambuilder training will use sample teams only")

        for team_num in range(total_teams):
            if self.should_stop:
                break

            # Generate a team
            generated_team = self.trainer.generate_team()
            if generated_team is None:
                logger.warning("Failed to generate team, using sample team")
                import random
                if self.sample_teams:
                    generated_team = random.choice(self.sample_teams)
                else:
                    continue

            self.total_teams_generated += 1

            # Evaluate team by playing games
            if self.player_network and self.player_encoder:
                wins, losses = await self._evaluate_team(
                    generated_team,
                    games_per_team,
                )
                self.total_battles += wins + losses

                # Record outcome
                for _ in range(wins):
                    self.trainer.add_battle_outcome(
                        team=generated_team,
                        won=True,
                        turns=30,  # Placeholder
                        opponent_revealed=[],
                        elo_delta=0.0,
                    )
                for _ in range(losses):
                    self.trainer.add_battle_outcome(
                        team=generated_team,
                        won=False,
                        turns=30,
                        opponent_revealed=[],
                        elo_delta=0.0,
                    )

            # Update teambuilder
            metrics = self.trainer.update()

            # Log progress
            if team_num % 10 == 0:
                best_teams = self.trainer.outcome_buffer.get_best_teams(n=3)
                best_wr = best_teams[0][1] if best_teams else 0.0
                logger.info(
                    f"Team {team_num + 1}/{total_teams} | "
                    f"Total battles: {self.total_battles} | "
                    f"Best win rate: {best_wr:.1%}"
                )

            # Save checkpoint
            if (team_num + 1) % save_interval == 0:
                self._save_checkpoint()

            gc.collect()

        self._save_checkpoint(final=True)
        logger.info(f"Teambuilder training complete! Generated {self.total_teams_generated} teams")

    async def _evaluate_team(
        self,
        team,
        n_games: int,
    ) -> tuple[int, int]:
        """Evaluate a team by playing games.

        Args:
            team: Team to evaluate
            n_games: Number of games to play

        Returns:
            (wins, losses)
        """
        if not self.player_network or not self.player_encoder:
            return 0, 0

        wins = 0
        losses = 0

        # Create player with this specific team
        player = OUNeuralNetworkPlayer(
            network=self.player_network,
            state_encoder=self.player_encoder,
            device=self.device,
            deterministic=True,
            teams=[team],
            battle_format="gen9ou",
            max_concurrent_battles=1,
        )

        try:
            # Play against random opponent
            opponent = RandomPlayer(
                battle_format="gen9ou",
                max_concurrent_battles=1,
            )

            await player.battle_against(opponent, n_battles=n_games)

            for battle in player.battles.values():
                if battle.won:
                    wins += 1
                else:
                    losses += 1

        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
        finally:
            # Cleanup
            if hasattr(player, 'ps_client') and player.ps_client:
                try:
                    await player.ps_client.stop_listening()
                except Exception:
                    pass

        return wins, losses

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save teambuilder checkpoint."""
        checkpoint_path = self.checkpoint_dir / "latest.pt"
        self.trainer.save_checkpoint(checkpoint_path)

        if not final:
            periodic_path = self.checkpoint_dir / f"teambuilder_{self.total_teams_generated}.pt"
            self.trainer.save_checkpoint(periodic_path)


class JointTrainingManager:
    """Manages joint player/teambuilder training with feedback loop.

    This coordinates:
    1. Generating teams using the teambuilder
    2. Playing battles with generated teams
    3. Updating player based on battle experiences
    4. Updating teambuilder based on team win rates
    5. Rotating teams based on performance
    """

    def __init__(
        self,
        config: OUTrainingConfig,
        device: torch.device,
        teams_dir: str = "data/sample_teams",
        checkpoint_dir: str = "data/checkpoints/ou/joint",
        log_dir: str = "runs/ou/joint",
        num_envs: int = 1,
        curriculum_strategy: str = "adaptive",
    ):
        self.config = config
        self.device = device
        self.num_envs = num_envs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.curriculum_strategy = curriculum_strategy

        # Load sample teams for bootstrapping
        logger.info(f"Loading sample teams from {teams_dir}...")
        self.team_loader = TeamLoader(teams_dir)
        self.sample_teams = self.team_loader.load_teams("gen9ou")
        logger.info(f"Loaded {len(self.sample_teams)} sample teams")

        # Create shared embeddings
        logger.info("Initializing shared embeddings...")
        emb_config = EmbeddingConfig()
        self.embeddings = SharedEmbeddings(emb_config)

        # Create player network and encoder
        logger.info("Initializing player network...")
        self.encoder = OUStateEncoder(device=device)
        self.network = OUPlayerNetwork(
            pokemon_dim=self.encoder.POKEMON_FEATURES,
            field_dim=self.encoder.FIELD_FEATURES,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            num_actions=self.encoder.NUM_ACTIONS,
        ).to(device)

        num_params = sum(p.numel() for p in self.network.parameters())
        logger.info(f"Player model parameters: {num_params:,}")

        # Create teambuilder components
        logger.info("Initializing teambuilder...")
        self.generator = TeamGenerator(
            shared_embeddings=self.embeddings,
            hidden_dim=512,
            num_heads=8,
            num_layers=4,
        ).to(device)

        self.evaluator = TeamEvaluator(
            shared_embeddings=self.embeddings,
            hidden_dim=512,
        ).to(device)

        tb_params = sum(p.numel() for p in self.generator.parameters())
        tb_params += sum(p.numel() for p in self.evaluator.parameters())
        logger.info(f"Teambuilder model parameters: {tb_params:,}")

        # Create joint trainer using factory function
        logger.info(f"Using curriculum strategy: {curriculum_strategy}")
        self.joint_trainer = create_joint_trainer(
            config=config,
            player_network=self.network,
            state_encoder=self.encoder,
            team_generator=self.generator,
            team_evaluator=self.evaluator,
            device=device,
            curriculum_strategy=curriculum_strategy,
        )
        self.joint_trainer.setup_logging(log_dir)

        # Initialize team pool with sample teams
        self.joint_trainer.initialize_teams(
            sample_teams=self.sample_teams[:10],
            num_generated=5,
        )

        # Training state
        self.total_timesteps = 0
        self.total_episodes = 0
        self.should_stop = False

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info("\nShutdown requested, finishing current episode...")
        self.should_stop = True

    async def train(
        self,
        total_timesteps: int,
        eval_interval: int = 10000,
        save_interval: int = 50000,
    ) -> None:
        """Main joint training loop.

        Args:
            total_timesteps: Total timesteps to train
            eval_interval: Steps between evaluations
            save_interval: Steps between checkpoint saves
        """
        logger.info(f"Starting joint training for {total_timesteps:,} timesteps")
        logger.info(f"Using {self.num_envs} parallel environment(s)")
        logger.info(f"Active team pool: {len(self.joint_trainer.active_teams)} teams")

        start_time = datetime.now()
        last_eval = self.total_timesteps
        last_save = self.total_timesteps

        while self.total_timesteps < total_timesteps and not self.should_stop:
            # Collect experience from battles
            steps, episodes = await self._collect_rollout()
            self.total_timesteps += steps
            self.total_episodes += episodes

            # Joint training step (updates player and teambuilder)
            buffer = self.joint_trainer.player_trainer.buffer
            if len(buffer) >= self.config.player_batch_size:
                metrics = self.joint_trainer.training_step()

                # Log progress
                elapsed = (datetime.now() - start_time).total_seconds()
                steps_per_sec = self.total_timesteps / max(elapsed, 1)

                summary = self.joint_trainer.get_summary()
                logger.info(
                    f"Step {self.total_timesteps:,} | "
                    f"Episodes: {self.total_episodes:,} | "
                    f"Win rate: {summary['player_win_rate']:.1%} | "
                    f"Skill: {summary['player_skill_rating']:.0f} | "
                    f"Best team: {summary['best_team_win_rate']:.1%} | "
                    f"Teams: {summary['active_teams']} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )

            # Periodic evaluation
            if self.total_timesteps - last_eval >= eval_interval:
                await self._evaluate()
                last_eval = self.total_timesteps

            # Periodic checkpoint
            if self.total_timesteps - last_save >= save_interval:
                self._save_checkpoint()
                last_save = self.total_timesteps

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final save
        self._save_checkpoint(final=True)
        summary = self.joint_trainer.get_summary()
        logger.info(f"Joint training complete!")
        logger.info(f"  Total episodes: {self.total_episodes:,}")
        logger.info(f"  Final win rate: {summary['player_win_rate']:.1%}")
        logger.info(f"  Teams generated: {summary['teams_generated']}")

    async def _collect_rollout(self) -> tuple[int, int]:
        """Collect experience from battles using teams from the joint trainer.

        Returns:
            (steps_collected, episodes_completed)
        """
        steps = 0
        episodes = 0

        # Create players for this rollout
        players: list[OUTrainablePlayer] = []
        opponents = []
        teams_used = []
        opponent_types_used = []

        import random
        for i in range(self.num_envs):
            # Get team from joint trainer's pool (uses curriculum)
            team = self.joint_trainer.get_training_team()
            if team is None:
                # Fallback to sample teams
                if self.sample_teams:
                    team = random.choice(self.sample_teams)
                else:
                    continue

            teams_used.append(team)

            player = OUTrainablePlayer(
                network=self.network,
                state_encoder=self.encoder,
                device=self.device,
                teams=[team],
                current_team=team,
                battle_format="gen9ou",
                max_concurrent_battles=1,
            )
            players.append(player)

            # Use curriculum to select opponent type
            opponent_type = self.joint_trainer.get_opponent_type()
            opponent_types_used.append(opponent_type.value)

            if opponent_type == OpponentType.RANDOM:
                opponent = RandomPlayer(
                    battle_format="gen9ou",
                    max_concurrent_battles=1,
                )
            elif opponent_type == OpponentType.MAXDAMAGE:
                opponent = OUMaxDamagePlayer(
                    teams=self.sample_teams,
                    battle_format="gen9ou",
                    max_concurrent_battles=1,
                )
            else:
                # SELF_PLAY and others: use maxdamage as placeholder
                # TODO: Add self-play support for joint training
                opponent = OUMaxDamagePlayer(
                    teams=self.sample_teams,
                    battle_format="gen9ou",
                    max_concurrent_battles=1,
                )
            opponents.append(opponent)

        if not players:
            return 0, 0

        try:
            # Run battles concurrently
            battle_tasks = []
            for player, opponent in zip(players, opponents):
                task = asyncio.create_task(self._run_battle(player, opponent))
                battle_tasks.append(task)

            results = await asyncio.gather(*battle_tasks, return_exceptions=True)

            # Process results and collect experiences
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Battle error: {result}")
                    continue

                battle_steps, won, turns, opponent_revealed = result
                steps += battle_steps
                episodes += 1

                # Record battle outcome in joint trainer
                team = teams_used[i]
                opponent_type = opponent_types_used[i]
                self.joint_trainer.record_battle_outcome(
                    team=team,
                    won=won,
                    turns=turns,
                    opponent_revealed=opponent_revealed,
                    opponent_rating=1000.0,  # Baseline opponents
                    opponent_type=opponent_type,
                )

                # Collect experiences from player
                player = players[i]
                experiences = player.get_experiences()

                for j, exp in enumerate(experiences):
                    next_state = None
                    if j < len(experiences) - 1:
                        next_state = experiences[j + 1]["state"]

                    self.joint_trainer.player_trainer.add_transition(
                        state=exp["state"],
                        action=exp["action"],
                        reward=exp["reward"],
                        next_state=next_state,
                        done=exp["done"],
                        log_prob=exp["log_prob"],
                        value=exp["value"],
                        team=exp.get("team"),
                    )

        finally:
            # Cleanup players
            await self._cleanup_players(players + opponents)

        return steps, episodes

    async def _run_battle(
        self,
        player: OUTrainablePlayer,
        opponent,
    ) -> tuple[int, bool, int, list[str]]:
        """Run a single battle.

        Returns:
            (steps, won, turns, opponent_revealed)
        """
        await player.battle_against(opponent, n_battles=1)

        if not player.battles:
            return 0, False, 0, []

        battle = list(player.battles.values())[-1]
        won = battle.won if battle.won is not None else False
        turns = max(battle.turn, 1)

        # Extract revealed opponent Pokemon
        opponent_revealed = []
        if hasattr(battle, 'opponent_team') and battle.opponent_team:
            opponent_revealed = [
                mon.species for mon in battle.opponent_team.values()
            ]

        return turns, won, turns, opponent_revealed

    async def _cleanup_players(self, players: list) -> None:
        """Clean up player connections."""
        cleanup_tasks = []
        for player in players:
            try:
                if hasattr(player, 'ps_client') and player.ps_client is not None:
                    cleanup_tasks.append(player.ps_client.stop_listening())
            except Exception:
                pass

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            await asyncio.sleep(0.5)

        for player in players:
            if hasattr(player, 'network'):
                player.network = None
            if hasattr(player, 'state_encoder'):
                player.state_encoder = None

    async def _evaluate(self, n_games: int = 20) -> dict:
        """Evaluate current policy against baselines."""
        logger.info("Running evaluation...")

        results = {"vs_random": 0, "vs_maxdamage": 0}

        # Use best performing team for evaluation
        best_teams = self.joint_trainer.teambuilder_trainer.outcome_buffer.get_best_teams(n=1)
        if best_teams:
            eval_teams = [best_teams[0][0]]
        else:
            eval_teams = self.sample_teams[:1]

        eval_player = OUNeuralNetworkPlayer(
            network=self.network,
            state_encoder=self.encoder,
            device=self.device,
            deterministic=True,
            teams=eval_teams,
            battle_format="gen9ou",
            max_concurrent_battles=1,
        )

        try:
            # vs Random
            random_opponent = RandomPlayer(
                battle_format="gen9ou",
                max_concurrent_battles=1,
            )
            await eval_player.battle_against(random_opponent, n_battles=n_games)

            wins = sum(1 for b in eval_player.battles.values() if b.won)
            results["vs_random"] = wins / n_games
            logger.info(f"  vs Random: {wins}/{n_games} ({results['vs_random']:.1%})")

            eval_player._battles = {}

            # vs MaxDamage
            maxdmg_opponent = OUMaxDamagePlayer(
                teams=self.sample_teams,
                battle_format="gen9ou",
                max_concurrent_battles=1,
            )
            await eval_player.battle_against(maxdmg_opponent, n_battles=n_games)

            wins = sum(1 for b in eval_player.battles.values() if b.won)
            results["vs_maxdamage"] = wins / n_games
            logger.info(f"  vs MaxDamage: {wins}/{n_games} ({results['vs_maxdamage']:.1%})")

        finally:
            await self._cleanup_players([eval_player, random_opponent, maxdmg_opponent])

        # Log to TensorBoard
        if self.joint_trainer.writer:
            self.joint_trainer.writer.add_scalar(
                "eval/vs_random", results["vs_random"], self.total_timesteps
            )
            self.joint_trainer.writer.add_scalar(
                "eval/vs_maxdamage", results["vs_maxdamage"], self.total_timesteps
            )

        return results

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save joint training checkpoint."""
        self.joint_trainer.save_checkpoint(self.checkpoint_dir)

        # Also save just the player model for easier loading
        player_checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "config": self.config.to_dict(),
        }
        torch.save(player_checkpoint, self.checkpoint_dir / "player_model.pt")
        logger.info(f"Saved joint checkpoint to {self.checkpoint_dir}")

    def load_checkpoint(self, path: str) -> None:
        """Load a joint training checkpoint."""
        checkpoint_path = Path(path)

        if checkpoint_path.is_dir():
            # Load full joint checkpoint
            self.joint_trainer.load_checkpoint(checkpoint_path)
            summary = self.joint_trainer.get_summary()
            self.total_timesteps = summary.get("total_battles", 0)
        else:
            # Try loading as player-only checkpoint
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.network.load_state_dict(checkpoint["model_state_dict"])
            self.total_timesteps = checkpoint.get("total_timesteps", 0)
            self.total_episodes = checkpoint.get("total_episodes", 0)

        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"  Timesteps: {self.total_timesteps:,}")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def find_latest_checkpoint(checkpoint_dir: str) -> Path | None:
    """Find the latest checkpoint."""
    path = Path(checkpoint_dir)
    if not path.exists():
        return None

    latest = path / "latest.pt"
    if latest.exists():
        return latest

    checkpoints = list(path.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    def get_timestep(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    return max(checkpoints, key=get_timestep)


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    device = get_device()
    config = OUTrainingConfig()

    # Print server reminder
    print("\n" + "=" * 60)
    print("NOTE: Training requires a Pokemon Showdown server running locally.")
    print("If not running, start it with:")
    print("  cd ~/pokemon-showdown && node pokemon-showdown start --no-security")
    print("=" * 60 + "\n")

    if args.mode == "player":
        # Player training mode
        manager = OUTrainingManager(
            config=config,
            device=device,
            teams_dir=args.teams_dir,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            num_envs=args.num_envs,
            use_self_play=not args.no_self_play,
        )

        # Load checkpoint if resuming
        if args.resume:
            if args.resume == "auto":
                checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
                if checkpoint_path:
                    logger.info(f"Auto-resuming from: {checkpoint_path}")
                    manager.load_checkpoint(str(checkpoint_path))
                else:
                    logger.info("No checkpoint found, starting fresh")
            else:
                manager.load_checkpoint(args.resume)

        try:
            await manager.train(
                total_timesteps=args.timesteps,
                eval_interval=args.eval_interval,
                save_interval=args.save_interval,
            )
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

    elif args.mode == "teambuilder":
        # Teambuilder training mode
        logger.info("=" * 60)
        logger.info("TEAMBUILDER TRAINING MODE")
        logger.info("Note: Teambuilder components are partially implemented.")
        logger.info("The generator produces placeholder teams for now.")
        logger.info("=" * 60)

        manager = TeambuilderTrainingManager(
            config=config,
            device=device,
            player_checkpoint=args.player_checkpoint,
            teams_dir=args.teams_dir,
            checkpoint_dir=args.checkpoint_dir + "/teambuilder",
            log_dir=args.log_dir + "/teambuilder",
        )

        try:
            await manager.train(
                total_teams=args.total_teams,
                games_per_team=args.games_per_team,
                save_interval=args.save_interval // 1000 or 100,
            )
        except Exception as e:
            logger.error(f"Teambuilder training error: {e}")
            raise

    elif args.mode == "joint":
        # Joint training mode (alternating player/teambuilder)
        logger.info("=" * 60)
        logger.info("JOINT TRAINING MODE")
        logger.info("This mode alternates between player and teambuilder training.")
        logger.info("Teams are rotated based on performance, and the teambuilder")
        logger.info("learns from battle outcomes via the evaluator network.")
        logger.info("=" * 60)

        joint_checkpoint_dir = args.checkpoint_dir + "/joint"
        joint_log_dir = args.log_dir + "/joint"

        manager = JointTrainingManager(
            config=config,
            device=device,
            teams_dir=args.teams_dir,
            checkpoint_dir=joint_checkpoint_dir,
            log_dir=joint_log_dir,
            num_envs=args.num_envs,
            curriculum_strategy=args.curriculum if args.curriculum != "none" else "adaptive",
        )

        if args.resume:
            if args.resume == "auto":
                checkpoint_path = Path(joint_checkpoint_dir)
                if checkpoint_path.exists():
                    logger.info(f"Auto-resuming from: {checkpoint_path}")
                    manager.load_checkpoint(str(checkpoint_path))
                else:
                    logger.info("No checkpoint found, starting fresh")
            else:
                manager.load_checkpoint(args.resume)

        try:
            await manager.train(
                total_timesteps=args.timesteps,
                eval_interval=args.eval_interval,
                save_interval=args.save_interval,
            )
        except Exception as e:
            logger.error(f"Joint training error: {e}")
            raise


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Train the Pokemon Showdown Gen 9 OU RL bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["player", "teambuilder", "joint"],
        default="player",
        help="Training mode: player (battle decisions), teambuilder (team construction), joint (both)",
    )

    # General training arguments
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=1_000_000,
        help="Total timesteps to train (player mode)",
    )
    parser.add_argument(
        "--teams-dir",
        type=str,
        default="data/sample_teams",
        help="Directory containing team files",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/checkpoints/ou",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/ou",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Resume from checkpoint (auto or path)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10000,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50000,
        help="Steps between checkpoint saves",
    )
    parser.add_argument(
        "--num-envs", "-e",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--no-self-play",
        action="store_true",
        help="Disable self-play (use only random/maxdamage opponents)",
    )

    # Teambuilder-specific arguments
    parser.add_argument(
        "--player-checkpoint",
        type=str,
        default=None,
        help="Player checkpoint for team evaluation (teambuilder mode)",
    )
    parser.add_argument(
        "--total-teams",
        type=int,
        default=1000,
        help="Total teams to generate (teambuilder mode)",
    )
    parser.add_argument(
        "--games-per-team",
        type=int,
        default=10,
        help="Games to play per team for evaluation (teambuilder mode)",
    )

    # Curriculum learning arguments (joint mode)
    parser.add_argument(
        "--curriculum",
        type=str,
        choices=["adaptive", "progressive", "matchup", "complexity", "none"],
        default="adaptive",
        help="Curriculum learning strategy (joint mode). 'none' disables curriculum.",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

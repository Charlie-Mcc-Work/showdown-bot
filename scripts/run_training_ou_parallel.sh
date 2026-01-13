#!/bin/bash
# Parallel OU self-play training - runs multiple training processes
# All workers share the same opponent pool, so they learn from each other
#
# This bypasses Python's GIL while keeping full self-play training.
# Throughput increases ~linearly with workers.

set -e

# Configuration
NUM_WORKERS=3          # Number of parallel training processes
ENVS_PER_WORKER=4      # Environments per worker
SERVERS_PER_WORKER=1   # Servers per worker
BASE_PORT=8000
TIMESTEPS=5000000
MODE="joint"           # joint, player, or teambuilder
CURRICULUM="adaptive"  # adaptive, progressive, matchup, complexity, none

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --envs-per-worker)
            ENVS_PER_WORKER="$2"
            shift 2
            ;;
        --servers-per-worker)
            SERVERS_PER_WORKER="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --curriculum)
            CURRICULUM="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Runs multiple OU self-play training processes in parallel."
            echo "All workers share the opponent pool - they learn from each other!"
            echo ""
            echo "Options:"
            echo "  --workers N           Number of parallel workers (default: 3)"
            echo "  --envs-per-worker N   Environments per worker (default: 4)"
            echo "  --servers-per-worker N Servers per worker (default: 1)"
            echo "  --timesteps N         Steps per run per worker (default: 5000000)"
            echo "  --mode MODE           Training mode: joint, player (default: joint)"
            echo "  --curriculum STR      Curriculum strategy (default: adaptive)"
            echo "  -h, --help            Show this help"
            echo ""
            echo "Example:"
            echo "  $0 --workers 4 --envs-per-worker 4  # 16 total envs across 4 processes"
            echo ""
            echo "Features:"
            echo "  - Full self-play training (not RandomPlayer)"
            echo "  - Shared opponent pool across all workers"
            echo "  - Auto-resumes from best checkpoint"
            echo "  - Ctrl+C gracefully saves all workers"
            echo "  - Best model merged at end"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PS_DIR="$HOME/pokemon-showdown"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

TOTAL_SERVERS=$((NUM_WORKERS * SERVERS_PER_WORKER))
TOTAL_ENVS=$((NUM_WORKERS * ENVS_PER_WORKER))

# Shared directories based on mode
if [ "$MODE" = "joint" ]; then
    CHECKPOINT_BASE="data/checkpoints/ou/joint"
    SHARED_OPPONENT_DIR="data/checkpoints/ou/joint/opponent_pool"
    LOG_BASE="runs/ou/joint"
elif [ "$MODE" = "teambuilder" ]; then
    CHECKPOINT_BASE="data/checkpoints/ou/teambuilder"
    SHARED_OPPONENT_DIR="data/checkpoints/ou/teambuilder/opponent_pool"
    LOG_BASE="runs/ou/teambuilder"
else
    CHECKPOINT_BASE="data/checkpoints/ou"
    SHARED_OPPONENT_DIR="data/checkpoints/ou/opponent_pool"
    LOG_BASE="runs/ou"
fi
BEST_MODEL="$CHECKPOINT_BASE/best_model.pt"

# Track PIDs
SERVER_PIDS=()
WORKER_PIDS=()
USER_EXIT_REQUESTED=false

cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"

    # Kill workers first (they'll save checkpoints on SIGINT)
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${BLUE}Stopping worker (PID: $pid)${NC}"
            kill -INT "$pid" 2>/dev/null || true
        fi
    done

    # Wait for workers to save checkpoints
    echo -e "${BLUE}Waiting for workers to save checkpoints...${NC}"
    sleep 5

    # Force kill any remaining workers
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # Kill servers
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${BLUE}Stopping server (PID: $pid)${NC}"
            kill "$pid" 2>/dev/null || true
        fi
    done

    # Kill any node processes on our ports
    for i in $(seq 0 $((TOTAL_SERVERS - 1))); do
        port=$((BASE_PORT + i))
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    # Merge best checkpoint from workers
    merge_checkpoints

    echo -e "${GREEN}Cleanup complete${NC}"
}

handle_sigint() {
    echo -e "\n${YELLOW}Ctrl+C pressed - stopping all workers...${NC}"
    USER_EXIT_REQUESTED=true
    cleanup
    exit 0
}

trap handle_sigint SIGINT SIGTERM
trap cleanup EXIT

start_servers() {
    echo -e "${BLUE}Starting $TOTAL_SERVERS Pokemon Showdown servers...${NC}"

    if [ ! -d "$PS_DIR" ]; then
        echo -e "${RED}Error: Pokemon Showdown not found at $PS_DIR${NC}"
        exit 1
    fi

    cd "$PS_DIR"

    for i in $(seq 0 $((TOTAL_SERVERS - 1))); do
        port=$((BASE_PORT + i))

        if lsof -ti:$port >/dev/null 2>&1; then
            echo -e "${YELLOW}Port $port in use, killing existing process${NC}"
            kill $(lsof -ti:$port) 2>/dev/null || true
            sleep 1
        fi

        echo -e "${GREEN}Starting server on port $port${NC}"
        node pokemon-showdown start --no-security --port $port > /dev/null 2>&1 &
        SERVER_PIDS+=($!)
        sleep 0.3
    done

    echo -e "${BLUE}Waiting for servers to initialize...${NC}"
    sleep 3
    echo -e "${GREEN}All servers started${NC}"
}

start_workers() {
    cd "$PROJECT_DIR"

    # Activate virtual environment
    if [ -f ".venv-rocm/bin/activate" ]; then
        source .venv-rocm/bin/activate
    elif [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo -e "${RED}No virtual environment found${NC}"
        exit 1
    fi

    # Create shared directories
    mkdir -p "$SHARED_OPPONENT_DIR"
    mkdir -p logs

    # Check for existing best model to resume from
    RESUME_ARG=""
    if [ -f "$BEST_MODEL" ]; then
        CHECKPOINT_INFO=$(python -c "
import torch
cp = torch.load('$BEST_MODEL', map_location='cpu', weights_only=False)
steps = cp.get('total_timesteps', 0)
print(f'Steps: {steps:,}')
" 2>/dev/null || echo "Unable to read")
        echo -e "${BLUE}Resuming all workers from: $BEST_MODEL${NC}"
        echo -e "${BLUE}  $CHECKPOINT_INFO${NC}"
        RESUME_ARG="--resume $BEST_MODEL"
    else
        echo -e "${YELLOW}No checkpoint found, starting fresh${NC}"
    fi

    echo ""
    echo -e "${BLUE}Starting $NUM_WORKERS parallel OU training workers...${NC}"
    echo -e "${CYAN}Mode: $MODE | Curriculum: $CURRICULUM${NC}"
    echo ""

    for worker in $(seq 0 $((NUM_WORKERS - 1))); do
        # Calculate port range for this worker
        start_port=$((BASE_PORT + worker * SERVERS_PER_WORKER))
        ports=""
        for s in $(seq 0 $((SERVERS_PER_WORKER - 1))); do
            port=$((start_port + s))
            ports="$ports $port"
        done

        # Each worker saves to its own directory (for safety)
        save_dir="$CHECKPOINT_BASE/worker_${worker}"
        log_dir="$LOG_BASE/worker_${worker}"

        echo -e "${CYAN}Worker $worker: ports$ports, envs=$ENVS_PER_WORKER${NC}"

        # Build training command
        TRAIN_CMD="python scripts/train_ou.py \
            --mode $MODE \
            --num-envs $ENVS_PER_WORKER \
            --server-ports $ports \
            --timesteps $TIMESTEPS \
            --checkpoint-dir $save_dir \
            --log-dir $log_dir"

        # Add curriculum for joint mode
        if [ "$MODE" = "joint" ]; then
            TRAIN_CMD="$TRAIN_CMD --curriculum $CURRICULUM"
        fi

        # Add resume if checkpoint exists
        if [ -n "$RESUME_ARG" ]; then
            TRAIN_CMD="$TRAIN_CMD $RESUME_ARG"
        fi

        # Start worker in background
        eval "$TRAIN_CMD" > "logs/ou_worker_${worker}.log" 2>&1 &

        WORKER_PIDS+=($!)
        sleep 1
    done

    echo ""
    echo -e "${GREEN}All workers started!${NC}"
}

monitor_workers() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Parallel OU Self-Play Training${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Mode: ${BLUE}$MODE${NC}"
    echo -e "Total environments: ${BLUE}$TOTAL_ENVS${NC}"
    echo -e "Total servers: ${BLUE}$TOTAL_SERVERS${NC}"
    echo ""
    echo "Worker logs: logs/ou_worker_*.log"
    echo "Live view:   tail -f logs/ou_worker_0.log"
    echo ""
    echo "Press Ctrl+C to stop all workers gracefully."
    echo ""

    # Monitor until all workers complete or user interrupts
    while true; do
        alive=0
        for pid in "${WORKER_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=$((alive + 1))
            fi
        done

        if [ $alive -eq 0 ]; then
            echo -e "\n${GREEN}All workers completed${NC}"
            break
        fi

        echo -ne "\r${BLUE}Workers: $alive/$NUM_WORKERS running | Use: tail -f logs/ou_worker_0.log${NC}     "
        sleep 5
    done
}

merge_checkpoints() {
    echo ""
    echo -e "${BLUE}Finding best checkpoint across workers...${NC}"

    best_skill=0
    best_checkpoint=""

    for worker in $(seq 0 $((NUM_WORKERS - 1))); do
        # Check both best_model.pt and latest.pt
        for ckpt_name in best_model.pt latest.pt; do
            checkpoint="$CHECKPOINT_BASE/worker_${worker}/$ckpt_name"
            if [ -f "$checkpoint" ]; then
                skill=$(python -c "
import torch
cp = torch.load('$checkpoint', map_location='cpu', weights_only=False)
# Try different locations for skill rating
skill = cp.get('skill_rating', 0)
if skill == 0:
    skill = cp.get('self_play', {}).get('agent_skill', 0)
print(skill)
" 2>/dev/null || echo "0")
                echo -e "Worker $worker ($ckpt_name): Skill = $skill"

                if (( $(echo "$skill > $best_skill" | bc -l) )); then
                    best_skill=$skill
                    best_checkpoint=$checkpoint
                fi
                break  # Only check best_model.pt if it exists
            fi
        done
    done

    if [ -n "$best_checkpoint" ]; then
        echo ""
        echo -e "${GREEN}Best model: $best_checkpoint (Skill: $best_skill)${NC}"
        echo -e "Copying to $CHECKPOINT_BASE/best_model.pt..."
        mkdir -p "$CHECKPOINT_BASE"
        cp "$best_checkpoint" "$CHECKPOINT_BASE/best_model.pt"
        echo -e "${GREEN}Done! Best model saved to $CHECKPOINT_BASE/best_model.pt${NC}"
    fi
}

main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Parallel OU Self-Play Training${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Mode: ${BLUE}$MODE${NC}"
    echo -e "Workers: ${BLUE}$NUM_WORKERS${NC}"
    echo -e "Envs per worker: ${BLUE}$ENVS_PER_WORKER${NC}"
    echo -e "Total environments: ${BLUE}$TOTAL_ENVS${NC}"
    echo -e "Servers per worker: ${BLUE}$SERVERS_PER_WORKER${NC}"
    echo -e "Total servers: ${BLUE}$TOTAL_SERVERS${NC} (ports $BASE_PORT-$((BASE_PORT + TOTAL_SERVERS - 1)))"
    if [ "$MODE" = "joint" ]; then
        echo -e "Curriculum: ${BLUE}$CURRICULUM${NC}"
    fi
    echo ""

    start_servers

    RUN_COUNT=0

    # Training loop - continues until Ctrl+C
    while true; do
        if [ "$USER_EXIT_REQUESTED" = true ]; then
            echo -e "${GREEN}User requested stop. Exiting...${NC}"
            break
        fi

        RUN_COUNT=$((RUN_COUNT + 1))
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  Starting OU training run #$RUN_COUNT${NC}"
        echo -e "${GREEN}========================================${NC}"

        # Reset worker PIDs for this run
        WORKER_PIDS=()

        start_workers
        monitor_workers
        merge_checkpoints

        if [ "$USER_EXIT_REQUESTED" = true ]; then
            break
        fi

        echo -e "${BLUE}Run $RUN_COUNT completed. Starting next run in 3 seconds...${NC}"
        sleep 3
    done

    echo ""
    echo -e "${GREEN}OU Training session ended after $RUN_COUNT run(s)${NC}"
}

main "$@"

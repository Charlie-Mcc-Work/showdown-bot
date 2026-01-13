#!/bin/bash
# Multi-process training script that bypasses Python's GIL
#
# Main process owns the model and does PPO updates, worker processes
# collect experiences in parallel. This achieves true CPU parallelism.
#
# Example:
#   ./scripts/run_training_multiproc.sh --workers 3 --envs-per-worker 4
#   # = 12 total envs across 3 parallel Python processes

set -e

# Configuration (defaults)
NUM_WORKERS=3          # Number of parallel training processes
ENVS_PER_WORKER=4      # Environments per worker
NUM_SERVERS=3          # Pokemon Showdown servers to start
BASE_PORT=8000
TIMESTEPS=5000000

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
        --servers)
            NUM_SERVERS="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Runs multi-process PPO training to bypass Python's GIL."
            echo "Main process owns model, workers collect experiences in parallel."
            echo ""
            echo "Options:"
            echo "  --workers N           Number of parallel workers (default: 3)"
            echo "  --envs-per-worker N   Environments per worker (default: 4)"
            echo "  --servers N           Number of PS servers (default: 3)"
            echo "  --timesteps N         Training steps (default: 5000000)"
            echo "  --resume PATH         Resume from checkpoint"
            echo "  -h, --help            Show this help"
            echo ""
            echo "Example:"
            echo "  $0 --workers 4 --envs-per-worker 4  # 16 total envs across 4 processes"
            echo ""
            echo "Resource usage with defaults (3 workers):"
            echo "  - 3 Pokemon Showdown servers (ports 8000-8002)"
            echo "  - 12 total environments (4 per worker)"
            echo "  - ~3 CPU cores for Python processes"
            echo "  - GPU shared across all workers"
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
NC='\033[0m'

TOTAL_ENVS=$((NUM_WORKERS * ENVS_PER_WORKER))

# Track PIDs
SERVER_PIDS=()
TRAINER_PID=""

cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"

    # Send SIGINT to trainer first (it will save checkpoint)
    if [ -n "$TRAINER_PID" ] && kill -0 "$TRAINER_PID" 2>/dev/null; then
        echo -e "${BLUE}Stopping trainer (PID: $TRAINER_PID)${NC}"
        kill -INT "$TRAINER_PID" 2>/dev/null || true
        # Wait for trainer to save
        sleep 5
        if kill -0 "$TRAINER_PID" 2>/dev/null; then
            kill -9 "$TRAINER_PID" 2>/dev/null || true
        fi
    fi

    # Kill servers
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${BLUE}Stopping server (PID: $pid)${NC}"
            kill "$pid" 2>/dev/null || true
        fi
    done

    # Kill any node processes on our ports
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        port=$((BASE_PORT + i))
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}Cleanup complete${NC}"
}

trap cleanup EXIT

start_servers() {
    echo -e "${BLUE}Starting $NUM_SERVERS Pokemon Showdown servers...${NC}"

    if [ ! -d "$PS_DIR" ]; then
        echo -e "${RED}Error: Pokemon Showdown not found at $PS_DIR${NC}"
        echo "Clone it with: git clone https://github.com/smogon/pokemon-showdown.git ~/pokemon-showdown"
        exit 1
    fi

    cd "$PS_DIR"

    for i in $(seq 0 $((NUM_SERVERS - 1))); do
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

build_ports_arg() {
    local ports=""
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        port=$((BASE_PORT + i))
        ports="$ports $port"
    done
    echo $ports
}

main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Multi-Process Training Launcher${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Workers: ${BLUE}$NUM_WORKERS${NC}"
    echo -e "Envs per worker: ${BLUE}$ENVS_PER_WORKER${NC}"
    echo -e "Total environments: ${BLUE}$TOTAL_ENVS${NC}"
    echo -e "Servers: ${BLUE}$NUM_SERVERS${NC} (ports $BASE_PORT-$((BASE_PORT + NUM_SERVERS - 1)))"
    echo -e "Timesteps: ${BLUE}$TIMESTEPS${NC}"
    echo ""

    start_servers

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

    PORTS=$(build_ports_arg)

    # Build command
    CMD="python scripts/train_multiproc.py \
        --workers $NUM_WORKERS \
        --envs-per-worker $ENVS_PER_WORKER \
        --server-ports $PORTS \
        --timesteps $TIMESTEPS"

    if [ -n "$RESUME" ]; then
        CMD="$CMD --resume $RESUME"
    fi

    echo ""
    echo -e "${BLUE}Starting multi-process training...${NC}"
    echo "Press Ctrl+C to stop gracefully."
    echo ""

    # Run and capture PID
    eval $CMD &
    TRAINER_PID=$!

    # Wait for trainer
    wait $TRAINER_PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}Training completed successfully${NC}"
    elif [ $EXIT_CODE -eq 130 ]; then
        echo -e "${YELLOW}Training stopped by user${NC}"
    else
        echo -e "${RED}Training exited with code $EXIT_CODE${NC}"
    fi
}

main "$@"

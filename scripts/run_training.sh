#!/bin/bash
# Automated training script with multi-server support
# - Starts Pokemon Showdown servers automatically
# - Resumes from checkpoint or best model
# - Restarts on memory issues or run completion
# - Ctrl+C gracefully exits everything

set -e

# Configuration (defaults)
NUM_SERVERS=1
BASE_PORT=8000
NUM_ENVS=6  # Sweet spot for single-process training
TIMESTEPS=5000000  # 5M steps per run
DEVICE="cuda"  # Use GPU by default
MULTIPROC=false  # Use multi-process training (bypasses GIL)
NUM_WORKERS=1  # Only used in multiproc mode

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --num-servers)
            NUM_SERVERS="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --multiproc|-m)
            MULTIPROC=true
            shift
            ;;
        --workers|-w)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --num-envs N      Environments per worker (default: 6)"
            echo "  --num-servers N   Number of PS servers to start (default: 1)"
            echo "  --multiproc, -m   Use multi-process training (bypasses GIL)"
            echo "  --workers N       Number of workers for multiproc (default: 2)"
            echo "  --timesteps N     Steps per training run (default: 5000000)"
            echo "  --device DEV      Device: cuda, cpu (default: cuda)"
            echo "  -h, --help        Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                    # Single-process, 1 server, 6 envs (~290/s)"
            echo "  $0 --multiproc        # Multi-process, 2 workers x 6 envs (~450/s)"
            echo "  $0 -m --workers 3     # Multi-process with 3 workers (~550/s?)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-scale servers based on mode
if [ "$MULTIPROC" = true ]; then
    # Multi-process: default to 2 workers if not specified
    [ "$NUM_WORKERS" -eq 1 ] && NUM_WORKERS=2
    # Auto-scale servers to match workers
    [ "$NUM_SERVERS" -eq 1 ] && NUM_SERVERS=$NUM_WORKERS
fi
TOTAL_ENVS=$((NUM_WORKERS * NUM_ENVS))

PS_DIR="$HOME/pokemon-showdown"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track server PIDs
SERVER_PIDS=()

# Track if user requested exit
USER_EXIT_REQUESTED=false

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"

    # Kill all Pokemon Showdown servers we started
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${BLUE}Stopping server (PID: $pid)${NC}"
            kill "$pid" 2>/dev/null || true
        fi
    done

    # Also kill any node processes on our ports (in case PIDs were lost)
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        port=$((BASE_PORT + i))
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            echo -e "${BLUE}Killing process on port $port (PID: $pid)${NC}"
            kill "$pid" 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}Cleanup complete${NC}"
}

# Handle Ctrl+C - set flag and let main loop handle it
handle_sigint() {
    echo -e "\n${YELLOW}Ctrl+C pressed - saving checkpoint and exiting...${NC}"
    USER_EXIT_REQUESTED=true
}

# Set up trap for SIGINT (Ctrl+C) - don't cleanup yet, just set flag
trap handle_sigint SIGINT

# Cleanup on actual exit
trap cleanup EXIT

# Function to start Pokemon Showdown servers
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

        # Check if port is already in use
        if lsof -ti:$port >/dev/null 2>&1; then
            echo -e "${YELLOW}Port $port already in use, killing existing process${NC}"
            kill $(lsof -ti:$port) 2>/dev/null || true
            sleep 1
        fi

        echo -e "${GREEN}Starting server on port $port${NC}"
        node pokemon-showdown start --no-security --port $port > /dev/null 2>&1 &
        SERVER_PIDS+=($!)
        sleep 0.5  # Brief pause between server starts
    done

    # Wait for servers to be ready
    echo -e "${BLUE}Waiting for servers to initialize...${NC}"
    sleep 3

    # Verify servers are running
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        port=$((BASE_PORT + i))
        if ! lsof -ti:$port >/dev/null 2>&1; then
            echo -e "${RED}Warning: Server on port $port may not have started${NC}"
        fi
    done

    echo -e "${GREEN}All servers started${NC}"
}

# Build server ports argument
build_ports_arg() {
    local ports=""
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        port=$((BASE_PORT + i))
        ports="$ports $port"
    done
    echo $ports
}

# Main training loop
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Pokemon Showdown RL Training Script${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    if [ "$MULTIPROC" = true ]; then
        echo -e "Mode: ${BLUE}Multi-Process${NC} (bypasses GIL)"
        echo -e "Workers: ${BLUE}$NUM_WORKERS${NC} (separate processes)"
    else
        echo -e "Mode: ${BLUE}Single-Process${NC}"
    fi
    echo -e "Servers: ${BLUE}$NUM_SERVERS${NC} (ports $BASE_PORT-$((BASE_PORT + NUM_SERVERS - 1)))"
    echo -e "Envs/worker: ${BLUE}$NUM_ENVS${NC}"
    echo -e "Total battles: ${BLUE}$((NUM_WORKERS * NUM_ENVS))${NC}"
    echo -e "Device: ${BLUE}$DEVICE${NC}"
    echo -e "Steps per run: ${BLUE}$TIMESTEPS${NC}"
    echo ""

    # Start servers
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
    RUN_COUNT=0
    CHECKPOINT_DIR="data/checkpoints"
    LATEST_MODEL="$CHECKPOINT_DIR/latest.pt"

    # Training loop - continues until Ctrl+C
    while true; do
        # Check if user requested exit before starting new run
        if [ "$USER_EXIT_REQUESTED" = true ]; then
            echo -e "${GREEN}User requested stop. Exiting...${NC}"
            break
        fi

        RUN_COUNT=$((RUN_COUNT + 1))
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  Starting training run #$RUN_COUNT${NC}"
        echo -e "${GREEN}========================================${NC}"

        RESUME_ARG=""
        if [ -f "$LATEST_MODEL" ]; then
            STEPS=$(python -c "
import torch
cp = torch.load('$LATEST_MODEL', map_location='cpu', weights_only=False)
print(cp.get('stats', {}).get('total_timesteps', 0))
" 2>/dev/null || echo "0")
            echo -e "${BLUE}Resuming from: latest.pt (${STEPS} steps)${NC}"
            RESUME_ARG="--resume $LATEST_MODEL"
        else
            echo -e "${YELLOW}No checkpoint found, starting fresh${NC}"
        fi
        echo ""

        # Run training
        # Exit code 0 = normal completion (including memory soft limit)
        # Exit code 130 = SIGINT (Ctrl+C)
        mkdir -p logs
        set +e  # Don't exit on error
        if [ "$MULTIPROC" = true ]; then
            # Multi-process training (bypasses GIL, uses separate processes)
            stdbuf -oL python -u scripts/train_multiproc.py \
                $RESUME_ARG \
                --workers $NUM_WORKERS \
                --envs-per-worker $NUM_ENVS \
                --server-ports $PORTS \
                --timesteps $TIMESTEPS \
                --device $DEVICE 2>&1 | tee logs/worker_0.log
            EXIT_CODE=${PIPESTATUS[0]}
        else
            # Single-process training
            stdbuf -oL python -u scripts/train.py \
                $RESUME_ARG \
                --num-envs $NUM_ENVS \
                --server-ports $PORTS \
                --timesteps $TIMESTEPS \
                --device $DEVICE 2>&1 | tee logs/worker_0.log
            EXIT_CODE=${PIPESTATUS[0]}
        fi
        set -e

        echo ""
        echo -e "${YELLOW}Training exited with code: $EXIT_CODE${NC}"

        # Check if user requested exit via Ctrl+C (either to script or passed through)
        if [ "$USER_EXIT_REQUESTED" = true ] || [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 2 ]; then
            echo -e "${GREEN}User requested stop. Exiting...${NC}"
            break
        elif [ $EXIT_CODE -eq 0 ]; then
            # Normal completion or memory soft limit
            echo -e "${BLUE}Run completed. Starting next run...${NC}"
            sleep 2
        else
            # Some other error
            echo -e "${YELLOW}Training exited unexpectedly. Restarting in 5 seconds...${NC}"
            sleep 5
        fi
    done

    echo ""
    echo -e "${GREEN}Training session ended after $RUN_COUNT run(s)${NC}"
}

# Run main
main "$@"

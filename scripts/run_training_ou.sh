#!/bin/bash
# Simple OU training script - handles everything automatically
#
# Usage:
#   ./scripts/run_training_ou.sh                    # 8 envs (optimal for single GPU)
#   ./scripts/run_training_ou.sh --num-envs 6       # 6 envs (lighter memory usage)
#   ./scripts/run_training_ou.sh --mode player      # Player-only training (faster)

set -e

# Defaults
NUM_WORKERS=1
ENVS_PER_WORKER=8  # Sweet spot for single GPU - more envs adds overhead
MODE="joint"
CURRICULUM="adaptive"
DEVICE="cuda"  # GPU works for multiple workers now that orphan cleanup is fixed
TIMESTEPS=5000000
BASE_PORT=8000

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers|-w)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --num-envs|--envs|-e)
            ENVS_PER_WORKER="$2"
            shift 2
            ;;
        --mode|-m)
            MODE="$2"
            shift 2
            ;;
        --device|-d)
            DEVICE="$2"
            shift 2
            ;;
        --timesteps|-t)
            TIMESTEPS="$2"
            shift 2
            ;;
        --curriculum)
            CURRICULUM="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --num-envs, -e N   Number of parallel environments (default: 8)"
            echo "  --mode, -m MODE    Training mode: joint, player (default: joint)"
            echo "  --device, -d DEV   Device: cuda, cpu (default: cuda)"
            echo "  --timesteps, -t N  Steps to train (default: 5000000)"
            echo "  --curriculum STR   Curriculum: adaptive, progressive, none (default: adaptive)"
            echo ""
            echo "Examples:"
            echo "  $0                      # 8 parallel battles on GPU (optimal)"
            echo "  $0 --num-envs 6         # 6 parallel battles (lighter memory)"
            echo "  $0 --mode player        # Faster player-only training"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto device selection
if [ "$DEVICE" = "auto" ]; then
    DEVICE="cuda"
fi

# Calculate servers needed (1 per 3-4 envs, minimum 2 for good throughput)
SERVERS_PER_WORKER=$(( (ENVS_PER_WORKER + 2) / 3 ))
[ "$SERVERS_PER_WORKER" -lt 2 ] && SERVERS_PER_WORKER=2
TOTAL_SERVERS=$((NUM_WORKERS * SERVERS_PER_WORKER))
TOTAL_ENVS=$((NUM_WORKERS * ENVS_PER_WORKER))

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

# Checkpoint paths based on mode
# Note: For joint mode, train_ou.py appends /joint to checkpoint_dir internally
# So we pass the base dir and let Python handle the mode-specific subdirectory
if [ "$MODE" = "joint" ]; then
    CHECKPOINT_DIR="data/checkpoints/ou"
    CHECKPOINT_RESUME_DIR="data/checkpoints/ou/joint"  # Where Python actually saves
    LOG_DIR="runs/ou/joint"
else
    CHECKPOINT_DIR="data/checkpoints/ou"
    CHECKPOINT_RESUME_DIR="data/checkpoints/ou"
    LOG_DIR="runs/ou"
fi

SERVER_PIDS=()
WORKER_PIDS=()

cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"

    # Kill workers gracefully first
    for pid in "${WORKER_PIDS[@]}"; do
        kill -INT "$pid" 2>/dev/null || true
    done
    sleep 3

    # Force kill any remaining train_ou processes
    pkill -9 -f "train_ou.py" 2>/dev/null || true

    # Kill servers
    for pid in "${SERVER_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done

    # Kill any pokemon-showdown on our ports
    for port in $(seq $BASE_PORT $((BASE_PORT + TOTAL_SERVERS - 1))); do
        pid=$(lsof -ti:$port 2>/dev/null || true)
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done

    echo -e "${GREEN}Cleanup complete${NC}"
}

trap cleanup EXIT INT TERM

# ============================================
# Pre-cleanup: kill any orphaned processes
# ============================================
echo -e "${BLUE}Checking for orphaned processes...${NC}"
orphans=$(pgrep -f "train_ou.py" 2>/dev/null | wc -l)
if [ "$orphans" -gt 0 ]; then
    echo -e "${YELLOW}Killing $orphans orphaned training processes...${NC}"
    pkill -9 -f "train_ou.py" 2>/dev/null || true
    sleep 2
fi

# Kill any processes on our ports
for port in $(seq $BASE_PORT $((BASE_PORT + TOTAL_SERVERS - 1))); do
    pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Killing process on port $port${NC}"
        kill -9 "$pid" 2>/dev/null || true
    fi
done
sleep 1

# ============================================
# Migrate old checkpoints if needed
# Old path: data/checkpoints/ou/joint/worker_X/joint/ or data/checkpoints/ou/joint/joint/
# New path: data/checkpoints/ou/worker_X/joint/ or data/checkpoints/ou/joint/
# ============================================
OLD_CHECKPOINT_BASE="data/checkpoints/ou/joint"
if [ -d "$OLD_CHECKPOINT_BASE" ]; then
    # Check for single-worker old path (double nested joint)
    if [ -d "$OLD_CHECKPOINT_BASE/joint" ] && [ ! -L "$OLD_CHECKPOINT_BASE/joint" ]; then
        # Old single worker checkpoints at data/checkpoints/ou/joint/joint/ - nothing to do
        # These will be found by the fallback logic
        :
    fi

    # Check for multi-worker old paths
    for old_worker_dir in "$OLD_CHECKPOINT_BASE"/worker_*/joint; do
        if [ -d "$old_worker_dir" ]; then
            # Extract worker number
            worker_num=$(basename "$(dirname "$old_worker_dir")" | sed 's/worker_//')
            new_worker_dir="data/checkpoints/ou/worker_$worker_num/joint"

            if [ ! -d "$new_worker_dir" ]; then
                echo -e "${YELLOW}Migrating old checkpoints: $old_worker_dir -> $new_worker_dir${NC}"
                mkdir -p "$(dirname "$new_worker_dir")"
                mv "$old_worker_dir" "$new_worker_dir"
            fi
        fi
    done
fi

# ============================================
# Print config
# ============================================
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  OU Training${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "Mode:         ${BLUE}$MODE${NC}"
echo -e "Workers:      ${BLUE}$NUM_WORKERS${NC}"
echo -e "Envs/worker:  ${BLUE}$ENVS_PER_WORKER${NC}"
echo -e "Total envs:   ${BLUE}$TOTAL_ENVS${NC}"
echo -e "Device:       ${BLUE}$DEVICE${NC}"
echo -e "Servers:      ${BLUE}$TOTAL_SERVERS${NC} (ports $BASE_PORT-$((BASE_PORT + TOTAL_SERVERS - 1)))"
if [ "$MODE" = "joint" ]; then
    echo -e "Curriculum:   ${BLUE}$CURRICULUM${NC}"
fi
echo -e "${GREEN}======================================${NC}"
echo ""

# ============================================
# Check Pokemon Showdown exists
# ============================================
if [ ! -d "$PS_DIR" ]; then
    echo -e "${RED}Error: Pokemon Showdown not found at $PS_DIR${NC}"
    echo "Install with:"
    echo "  git clone https://github.com/smogon/pokemon-showdown.git ~/pokemon-showdown"
    echo "  cd ~/pokemon-showdown && npm install"
    exit 1
fi

# ============================================
# Start Pokemon Showdown servers
# ============================================
echo -e "${BLUE}Starting $TOTAL_SERVERS Pokemon Showdown servers...${NC}"
cd "$PS_DIR"
for i in $(seq 0 $((TOTAL_SERVERS - 1))); do
    port=$((BASE_PORT + i))
    node pokemon-showdown start --no-security --port $port > /dev/null 2>&1 &
    SERVER_PIDS+=($!)
    sleep 0.5  # Stagger server starts
done
cd "$PROJECT_DIR"

# Wait for all servers to be ready
echo -e "${BLUE}Waiting for servers to initialize...${NC}"
for attempt in $(seq 1 30); do
    ready=0
    for i in $(seq 0 $((TOTAL_SERVERS - 1))); do
        port=$((BASE_PORT + i))
        if lsof -ti:$port >/dev/null 2>&1; then
            ready=$((ready + 1))
        fi
    done
    if [ "$ready" -eq "$TOTAL_SERVERS" ]; then
        echo -e "${GREEN}All $TOTAL_SERVERS servers ready${NC}"
        break
    fi
    if [ "$attempt" -eq 30 ]; then
        echo -e "${RED}Error: Only $ready/$TOTAL_SERVERS servers started${NC}"
        echo -e "${RED}Check if Pokemon Showdown is installed correctly${NC}"
        exit 1
    fi
    sleep 1
done

# ============================================
# Activate virtual environment
# ============================================
if [ -f ".venv-rocm/bin/activate" ]; then
    source .venv-rocm/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Create directories
mkdir -p "$CHECKPOINT_RESUME_DIR" logs

# ============================================
# Check for checkpoint to resume from (prefer best_model.pt)
# Also check old nested path for backward compatibility
# ============================================
RESUME_ARG=""
OLD_NESTED_DIR="$CHECKPOINT_RESUME_DIR/joint"  # Old buggy path had double /joint

if [ -f "$CHECKPOINT_RESUME_DIR/best_model.pt" ]; then
    echo -e "${BLUE}Resuming from: $CHECKPOINT_RESUME_DIR/best_model.pt${NC}"
    RESUME_ARG="--resume $CHECKPOINT_RESUME_DIR/best_model.pt"
elif [ -f "$CHECKPOINT_RESUME_DIR/player_model.pt" ]; then
    echo -e "${BLUE}Resuming from: $CHECKPOINT_RESUME_DIR/player_model.pt${NC}"
    RESUME_ARG="--resume $CHECKPOINT_RESUME_DIR/player_model.pt"
elif [ -f "$OLD_NESTED_DIR/best_model.pt" ]; then
    echo -e "${BLUE}Resuming from (old path): $OLD_NESTED_DIR/best_model.pt${NC}"
    RESUME_ARG="--resume $OLD_NESTED_DIR/best_model.pt"
elif [ -f "$OLD_NESTED_DIR/player_model.pt" ]; then
    echo -e "${BLUE}Resuming from (old path): $OLD_NESTED_DIR/player_model.pt${NC}"
    RESUME_ARG="--resume $OLD_NESTED_DIR/player_model.pt"
elif [ -d "$OLD_NESTED_DIR" ] && [ "$(ls -A $OLD_NESTED_DIR 2>/dev/null)" ]; then
    echo -e "${BLUE}Resuming from (old path): $OLD_NESTED_DIR${NC}"
    RESUME_ARG="--resume $OLD_NESTED_DIR"
elif [ -d "$CHECKPOINT_RESUME_DIR" ] && [ "$(ls -A $CHECKPOINT_RESUME_DIR 2>/dev/null)" ]; then
    echo -e "${BLUE}Resuming from: $CHECKPOINT_RESUME_DIR (directory)${NC}"
    RESUME_ARG="--resume $CHECKPOINT_RESUME_DIR"
else
    echo -e "${YELLOW}No checkpoint found, starting fresh${NC}"
fi
echo ""

# ============================================
# Start training worker(s)
# ============================================
echo -e "${BLUE}Starting $NUM_WORKERS training worker(s)...${NC}"
echo ""

for worker in $(seq 0 $((NUM_WORKERS - 1))); do
    # Calculate ports for this worker
    start_port=$((BASE_PORT + worker * SERVERS_PER_WORKER))
    ports=""
    for s in $(seq 0 $((SERVERS_PER_WORKER - 1))); do
        ports="$ports $((start_port + s))"
    done

    # Worker-specific paths for multi-worker
    if [ "$NUM_WORKERS" -gt 1 ]; then
        worker_checkpoint="$CHECKPOINT_DIR/worker_$worker"
        worker_log="$LOG_DIR/worker_$worker"
        mkdir -p "$worker_checkpoint"
    else
        worker_checkpoint="$CHECKPOINT_DIR"
        worker_log="$LOG_DIR"
    fi

    # Build command
    CMD="python scripts/train_ou.py \
        --mode $MODE \
        --num-envs $ENVS_PER_WORKER \
        --server-ports $ports \
        --timesteps $TIMESTEPS \
        --checkpoint-dir $worker_checkpoint \
        --log-dir $worker_log \
        --device $DEVICE"

    if [ "$MODE" = "joint" ]; then
        CMD="$CMD --curriculum $CURRICULUM"
    fi

    # Resume logic: single worker uses global RESUME_ARG, multi-worker auto-resumes
    if [ "$NUM_WORKERS" -eq 1 ] && [ -n "$RESUME_ARG" ]; then
        CMD="$CMD $RESUME_ARG"
    elif [ "$NUM_WORKERS" -gt 1 ]; then
        # Each worker auto-resumes from its own checkpoint directory
        CMD="$CMD --resume auto"
    fi

    # Start worker
    if [ "$NUM_WORKERS" -eq 1 ]; then
        # Single worker: run in foreground so Ctrl+C works properly
        # Tee to log file for monitoring (unbuffered for real-time updates)
        echo -e "${GREEN}Training started. Press Ctrl+C to stop.${NC}"
        echo -e "Monitor in another terminal: ${BLUE}tail -f logs/ou_worker_0.log${NC}"
        echo ""
        set +e
        eval "stdbuf -oL $CMD" 2>&1 | tee logs/ou_worker_0.log
        EXIT_CODE=${PIPESTATUS[0]}
        set -e
        echo -e "${YELLOW}Training exited with code: $EXIT_CODE${NC}"
    else
        # Multiple workers: run in background
        eval "$CMD" > "logs/ou_worker_$worker.log" 2>&1 &
        WORKER_PIDS+=($!)
        echo -e "${CYAN}Worker $worker started (PID: ${WORKER_PIDS[-1]}, ports:$ports)${NC}"
    fi
done

# ============================================
# For multi-worker, monitor and wait
# ============================================
if [ "$NUM_WORKERS" -gt 1 ]; then
    echo ""
    echo -e "${GREEN}All workers started!${NC}"
    echo -e "Logs: ${BLUE}logs/ou_worker_*.log${NC}"
    echo -e "Monitor: ${BLUE}tail -f logs/ou_worker_0.log${NC}"
    echo ""
    echo "Press Ctrl+C to stop all workers."
    echo ""

    # Wait for workers
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

        # Show brief status
        echo -ne "\r${BLUE}Workers running: $alive/$NUM_WORKERS${NC}  "
        sleep 5
    done
fi

echo -e "${GREEN}Training complete${NC}"

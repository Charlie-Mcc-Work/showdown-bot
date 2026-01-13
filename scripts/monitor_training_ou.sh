#!/bin/bash
# Monitor all parallel OU training workers and show combined stats
# Works with logs from run_training_ou_parallel.sh

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Parallel OU Training Monitor${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Press Ctrl+C to exit monitor (training continues)"
echo ""

while true; do
    # Count workers
    num_workers=$(ls logs/ou_worker_*.log 2>/dev/null | wc -l)

    if [ "$num_workers" -eq 0 ]; then
        echo -e "\r${YELLOW}No OU worker logs found. Is training running?${NC}    "
        sleep 2
        continue
    fi

    # Aggregate stats from all workers
    total_steps=0
    total_episodes=0
    total_speed=0
    total_win=0
    best_skill=0
    active_workers=0

    for log in logs/ou_worker_*.log; do
        # Get recent lines to find the latest progress update
        # OU logs use format: "Step 1,234 | Episodes: 56 | Win rate: 75.0% | Skill: 1200 | ..."
        line=$(grep -E "Step [0-9,]+ \|" "$log" 2>/dev/null | tail -1)

        if [ -n "$line" ]; then
            active_workers=$((active_workers + 1))

            # Extract step count (e.g., "Step 1,234,567")
            steps=$(echo "$line" | grep -oP 'Step [0-9,]+' | head -1 | grep -oP '[0-9,]+' | tr -d ',')
            if [ -n "$steps" ]; then
                total_steps=$((total_steps + steps))
            fi

            # Extract episodes (e.g., "Episodes: 123")
            episodes=$(echo "$line" | grep -oP 'Episodes: [0-9,]+' | head -1 | grep -oP '[0-9,]+' | tr -d ',')
            if [ -n "$episodes" ]; then
                total_episodes=$((total_episodes + episodes))
            fi

            # Extract win rate (e.g., "Win rate: 75.0%")
            win=$(echo "$line" | grep -oP 'Win rate: [0-9.]+' | head -1 | grep -oP '[0-9.]+')
            if [ -n "$win" ]; then
                # Convert to integer percentage
                win_int=$(echo "$win" | cut -d. -f1)
                total_win=$((total_win + win_int))
            fi

            # Extract skill (e.g., "Skill: 1200")
            skill=$(echo "$line" | grep -oP 'Skill: [0-9]+' | head -1 | grep -oP '[0-9]+')
            if [ -n "$skill" ] && [ "$skill" -gt "$best_skill" ]; then
                best_skill=$skill
            fi

            # Extract speed (e.g., "Speed: 45.2 steps/s")
            speed=$(echo "$line" | grep -oP 'Speed: [0-9.]+' | head -1 | grep -oP '[0-9.]+' | cut -d. -f1)
            if [ -n "$speed" ]; then
                total_speed=$((total_speed + speed))
            fi
        fi
    done

    # Calculate averages
    if [ $active_workers -gt 0 ]; then
        avg_win=$((total_win / active_workers))
    else
        avg_win=0
    fi

    # Format total steps
    if [ $total_steps -gt 1000000 ]; then
        total_fmt=$(echo "scale=2; $total_steps / 1000000" | bc)M
    elif [ $total_steps -gt 1000 ]; then
        total_fmt=$(echo "scale=1; $total_steps / 1000" | bc)K
    else
        total_fmt=$total_steps
    fi

    # Build progress bar (assume 10M total target for OU)
    target=10000000
    if [ $total_steps -gt 0 ]; then
        pct=$((total_steps * 100 / target))
        if [ $pct -gt 100 ]; then pct=100; fi
        filled=$((pct / 5))
        empty=$((20 - filled))
        bar=$(printf "%${filled}s" | tr ' ' '=')
        bar_empty=$(printf "%${empty}s" | tr ' ' '-')
    else
        bar=""
        bar_empty="--------------------"
        pct=0
    fi

    # Calculate ETA
    if [ $total_speed -gt 0 ]; then
        remaining=$((target - total_steps))
        if [ $remaining -lt 0 ]; then remaining=0; fi
        eta_seconds=$((remaining / total_speed))
        eta_hours=$((eta_seconds / 3600))
        eta_mins=$(((eta_seconds % 3600) / 60))
        eta="${eta_hours}h${eta_mins}m"
    else
        eta="--"
    fi

    # Display combined stats (percentage and ETA right after progress bar)
    echo -ne "\r[${bar}${bar_empty}] ${pct}% | ETA:${eta} | ${total_fmt} | ${CYAN}${total_speed}/s${NC} | Workers:${active_workers}/${num_workers} | Win:${avg_win}% | Skill:${best_skill} | Ep:${total_episodes}    "

    sleep 2
done

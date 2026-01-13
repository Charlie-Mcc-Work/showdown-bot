#!/bin/bash
# Monitor all parallel training workers and show combined stats

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Parallel Training Monitor${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Press Ctrl+C to exit monitor (training continues)"
echo ""

while true; do
    # Count workers
    num_workers=$(ls logs/worker_*.log 2>/dev/null | wc -l)

    if [ "$num_workers" -eq 0 ]; then
        echo -e "\r${YELLOW}No worker logs found. Is training running?${NC}    "
        sleep 2
        continue
    fi

    # Aggregate stats from all workers
    total_steps=0
    total_speed=0
    total_wins=0
    total_games=0
    best_skill=0
    active_workers=0

    for log in logs/worker_*.log; do
        # Get the last line (progress update)
        # Use tr to handle carriage returns - get last segment after any \r
        line=$(tail -1 "$log" 2>/dev/null | tr '\r' '\n' | tail -1)

        # Check if it's a progress line (contains M/ pattern)
        if echo "$line" | grep -q "M/.*M"; then
            active_workers=$((active_workers + 1))

            # Extract current steps (e.g., "7.24M" before the /)
            steps_m=$(echo "$line" | grep -oP '\d+\.\d+M/' | head -1 | sed 's/M\///')
            if [ -n "$steps_m" ]; then
                steps=$(echo "$steps_m * 1000000" | bc | cut -d. -f1)
                total_steps=$((total_steps + steps))
            fi

            # Extract speed (e.g., "115/s")
            speed=$(echo "$line" | grep -oP '\|\s+\d+/s' | grep -oP '\d+' | head -1)
            if [ -n "$speed" ]; then
                total_speed=$((total_speed + speed))
            fi

            # Extract win rate (e.g., "Win:74%" or "Win: 9%" with space padding)
            win=$(echo "$line" | grep -oP 'Win:\s*\d+' | head -1 | grep -oP '\d+')
            if [ -n "$win" ]; then
                total_wins=$((total_wins + win))
                total_games=$((total_games + 1))
            fi

            # Extract skill (e.g., "Skill:31428")
            skill=$(echo "$line" | grep -oP 'Skill:\d+' | head -1 | grep -oP '\d+')
            if [ -n "$skill" ] && [ "$skill" -gt "$best_skill" ]; then
                best_skill=$skill
            fi
        fi
    done

    # Calculate averages
    if [ $total_games -gt 0 ]; then
        avg_win=$((total_wins / total_games))
    else
        avg_win=0
    fi

    # Format total steps
    if [ $total_steps -gt 0 ]; then
        total_m=$(echo "scale=2; $total_steps / 1000000" | bc)
    else
        total_m="0.00"
    fi

    # Build progress bar (based on rough estimate - assume 50M total target)
    target=50000000
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
    echo -ne "\r[${bar}${bar_empty}] ${pct}% | ETA:${eta} | ${total_m}M | ${CYAN}${total_speed}/s${NC} | Workers:${active_workers}/${num_workers} | Win:${avg_win}% | Skill:${best_skill}    "

    sleep 2
done

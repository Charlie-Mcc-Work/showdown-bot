#!/bin/bash
# Monitor all parallel OU training workers and show combined stats
# Works with logs from run_training_ou.sh

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
    total_speed=0
    total_wins=0
    total_games=0
    best_skill=0
    active_workers=0

    for log in logs/ou_worker_*.log; do
        # Get the last line, strip ANSI codes and carriage returns
        line=$(tail -1 "$log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g; s/\][^]]*\\//g; s/\r//g')

        # Check if it's a progress line (contains [--- or [=== pattern and /s)
        if echo "$line" | grep -qE '\[[-=]+\].*[0-9]+/s'; then
            active_workers=$((active_workers + 1))

            # Extract current steps (e.g., "200" or "1.5K" or "7.5M" before the /)
            # Format: 200/5.00M or 1.5K/5.00M or 7.5M/5.00M
            steps_str=$(echo "$line" | sed -n 's/.*\] *\([0-9.]*[KM]*\)\/.*/\1/p')
            if [ -n "$steps_str" ]; then
                if [[ "$steps_str" == *M ]]; then
                    steps=$(echo "${steps_str%M} * 1000000" | bc 2>/dev/null | cut -d. -f1)
                elif [[ "$steps_str" == *K ]]; then
                    steps=$(echo "${steps_str%K} * 1000" | bc 2>/dev/null | cut -d. -f1)
                else
                    steps="$steps_str"
                fi
                [ -n "$steps" ] && [ "$steps" -eq "$steps" ] 2>/dev/null && total_steps=$((total_steps + steps))
            fi

            # Extract speed (e.g., "22/s")
            speed=$(echo "$line" | sed -n 's/.*| *\([0-9]*\)\/s.*/\1/p')
            if [ -n "$speed" ] && [ "$speed" -eq "$speed" ] 2>/dev/null; then
                total_speed=$((total_speed + speed))
            fi

            # Extract win rate (e.g., "Win: 0%" or "Win:85%")
            win=$(echo "$line" | sed -n 's/.*Win: *\([0-9]*\)%.*/\1/p')
            if [ -n "$win" ] && [ "$win" -eq "$win" ] 2>/dev/null; then
                total_wins=$((total_wins + win))
                total_games=$((total_games + 1))
            fi

            # Extract skill (e.g., "Skill:  914" or "Skill:30066")
            skill=$(echo "$line" | sed -n 's/.*Skill: *\([0-9]*\).*/\1/p')
            if [ -n "$skill" ] && [ "$skill" -eq "$skill" ] 2>/dev/null && [ "$skill" -gt "$best_skill" ]; then
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
    if [ $total_steps -gt 1000000 ]; then
        total_fmt=$(echo "scale=2; $total_steps / 1000000" | bc)M
    elif [ $total_steps -gt 1000 ]; then
        total_fmt=$(echo "scale=1; $total_steps / 1000" | bc)K
    else
        total_fmt=$total_steps
    fi

    # Build progress bar (assume 20M total target for OU with multiple workers)
    target=20000000
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

    # Display combined stats
    printf "\r[%s%s] %d%% | ETA:%s | %s | ${CYAN}%d/s${NC} | Workers:%d/%d | Win:%d%% | Skill:%d    " \
        "$bar" "$bar_empty" "$pct" "$eta" "$total_fmt" "$total_speed" "$active_workers" "$num_workers" "$avg_win" "$best_skill"

    sleep 2
done

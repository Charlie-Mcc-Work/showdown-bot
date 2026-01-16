# Bench-Gated Promotion System

Plan for implementing promotion criteria that requires maintaining MaxDamage benchmark performance.

## Status: IMPLEMENTED

## Problem

Pure self-play can cause "strategy drift" where models evolve to beat each other but forget Pokemon fundamentals. Need an anchor to reality.

## Solution: Option 4 (Floor + Non-Regression)

### Promotion Criteria

```
Promote if:
  (vs_best >= 60%) AND (bench% >= max(floor, prev_best_bench% - tolerance))
```

### Parameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Floor | 20% | Just above random (~15-18%). Minimum competency check. |
| Tolerance | 8% | With 50 games, std error is ~7%. Allows ~1 std dev variance. |
| vs_best threshold | 60% | Unchanged from current. |

## MaxDamage Games per Checkpoint

- **50 games** at each checkpoint
- Fast (~2-3 min), accurate enough (±7% std error)
- **Add experiences to training buffer** (don't waste them)
- Checkpoint frequency already varies (more frequent early), so total MaxDamage exposure is naturally higher during early training

### Checkpoint Schedule (already implemented)

| Period | Checkpoints | MaxDamage Games |
|--------|-------------|-----------------|
| 0-200k | 25k, 50k, 100k, 150k, 200k | 250 total |
| 200k+ | Every 100k | 50 per checkpoint |

## Implementation Changes Needed

### 1. Modify `run_benchmark_evaluation()` in `train_multiproc.py`

- Return experiences along with win rate
- Add experiences to training buffer

### 2. Add promotion tracking

- Track `prev_best_bench%` (the bench% when current best was promoted)
- Save/load this in checkpoint

### 3. Update promotion logic

```python
# Current:
if eval_vs_best >= 0.60:
    promote()

# New:
bench_threshold = max(0.20, prev_best_bench - 0.08)
if eval_vs_best >= 0.60 and bench_rate >= bench_threshold:
    promote()
    prev_best_bench = bench_rate
```

### 4. Run benchmark at every checkpoint

- Currently benchmark runs every 500k
- Change to run at every checkpoint (uses early checkpoint schedule)

## Example Scenarios

| Scenario | vs_best | bench% | prev_bench% | Promotes? |
|----------|---------|--------|-------------|-----------|
| Early success | 65% | 22% | N/A | Yes (above 20% floor) |
| Forgot fundamentals | 70% | 15% | 30% | No (below 20% floor) |
| Slight regression | 62% | 28% | 32% | Yes (28% >= 24%) |
| Major regression | 65% | 20% | 35% | No (20% < 27%) |
| Steady improvement | 60% | 40% | 35% | Yes |

## Dependencies

- Waiting on self-play experiment results to decide if this is needed
- If pure self-play works fine, may not need this complexity
- If pure self-play drifts, this provides guardrails

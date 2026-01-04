# Router Hysteresis Test: Time Crystallization

## Description

This test checks the hypothesis of "time crystallization" — a phenomenon where the router demonstrates inertia when switching between domains and retains memory of the traversed trajectory.

## Test Sequences

1. **A → B → A**: Pure switching between domains
   - Segment A (Shakespeare)
   - Segment B (Python)
   - Segment A (Shakespeare, return)

2. **A → (A/B mix) → A**: Switching through a mixed domain
   - Segment A (Shakespeare)
   - Segment Mix (mixed A/B)
   - Segment A (Shakespeare, return)

## Metrics

**⚠️ IMPORTANT:** See formal metric definitions in [METRICS_DEFINITIONS.md](METRICS_DEFINITIONS.md)

### Switch-lag (switching inertia)

**Formal definition:** Number of tokens after segment boundary until condition `w(domain) > 0.9` is met for **K=3 consecutive tokens** (excludes single peaks/spikes).

- **High switch-lag (>5 tokens)**: There is inertia, router "crystallizes"
- **Low switch-lag (≤2 tokens)**: Fast switching, no inertia

### Return-gap (trajectory memory)

**Formal definition:** Measure of difference between first segment "A" (A1) and second segment "A" (A2) by weight trajectory w(t).

**Computation methods:**
- **Cosine distance** (`1 - cosine_similarity`): 0.0 = identical, 1.0 = opposite
  - ⚠️ **Do NOT confuse with cosine_similarity!** This is distance, not similarity
- **Euclidean distance**: Euclidean distance between weight vectors
- **DTW (Dynamic Time Warping)**: Dynamic alignment of time series

**Interpretation:**
- **High return-gap (>0.1)**: Second "A" differs from first → trajectory memory exists
- **Low return-gap (<0.05)**: Second "A" similar to first → no trajectory memory

## Running

```bash
# First train the model
python temporal_lora.py

# Then run the hysteresis test
python test_hysteresis.py
```

## Result Interpretation

If time crystallization exists:
- ✓ Switch-lag will be high
- ✓ Return-gap will be high
- ✓ This confirms "epoch inertia" and "trajectory memory"

## Output Files

- `hysteresis_analysis.png`: Visualization of weight trajectories and segment comparison

## Dependencies

In addition to main project dependencies:
- `scipy` (for distance computation)

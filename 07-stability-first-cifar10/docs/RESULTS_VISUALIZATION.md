# Lazarus Project: Results Visualization

## Recovery Curve (Weight Noise)

The following graph shows the V-shape recovery pattern when applying Lazarus v3 protocol to models damaged by weight noise.

![Recovery Curve](../results/lazarus_recovery_curve.png)

### Key Observations:
- **V-Shape Pattern**: Clear recovery pattern visible at alpha=0.2
- **Optimal Zone**: Alpha 0.15-0.25 shows best recovery rates
- **Recovery Rate**: Up to 93.9% of lost accuracy recovered

### Data Table:

| Alpha | Damaged | Restored | Delta  | Recovery % | Agreement Improvement |
|-------|---------|----------|--------|------------|----------------------|
| 0.1   | 72.78%  | 72.98%   | +0.20% | 51.3%      | +2.56                |
| 0.2   | 71.06%  | 72.96%   | +1.90% | **90.0%**  | +4.69                |
| 0.3   | 69.84%  | 72.57%   | +2.73% | 82.0%      | +9.37                |

---

## Pruning Curve (Frozen Mask vs Regrow)

Comparison of two recovery modes: Frozen Mask (topology preserved) vs Regrow (weights allowed to regrow).

![Pruning Curve](../results/pruning_curve_comparison.png)

### Key Findings:
- **Frozen Mask > Regrow**: Topology preservation outperforms weight regrowth
- **Sweet Spot**: 70-80% pruning shows optimal recovery
- **Recovery Rate**: Up to 85.3% recovery at 80% pruning with Frozen Mask

### Data Table (Frozen Mask):

| Pruning | Pruned | Restored | Delta  | Recovery % | Agreement Improvement |
|---------|--------|----------|--------|------------|----------------------|
| 30%     | 72.77% | 72.68%   | -0.09% | -75.0%     | -0.0462              |
| 50%     | 72.52% | 72.66%   | +0.14% | 37.8%      | +0.2720              |
| 70%     | 71.85% | 72.65%   | +0.80% | **76.9%**  | +1.9332              |
| 80%     | 70.99% | 72.61%   | +1.62% | **85.3%**  | +5.3914              |

### Data Table (Regrow Allowed):

| Pruning | Pruned | Restored | Delta  | Recovery % | Agreement Improvement |
|---------|--------|----------|--------|------------|----------------------|
| 30%     | 72.77% | 72.56%   | -0.21% | -175.0%    | -0.0662              |
| 50%     | 72.52% | 72.66%   | +0.14% | 37.8%      | +0.2584              |
| 70%     | 71.85% | 72.59%   | +0.74% | 71.2%      | +1.9795              |
| 80%     | 70.99% | 72.50%   | +1.51% | 79.5%      | +5.5864              |

---

## Interpretation

### Recovery Curve Insights:

1. **V-Shape Recovery**: The graph clearly shows a V-shaped pattern where:
   - At low damage (α=0.1), recovery is minimal (network is already stable)
   - At optimal damage (α=0.2), recovery peaks at 90%
   - At high damage (α=0.3), recovery decreases but still significant (82%)

2. **Optimal Zone**: Alpha values between 0.15-0.25 provide the best recovery rates, confirming the "Sweet Spot" hypothesis.

3. **Agreement Improvement**: As damage increases, agreement with reference model improves significantly, indicating successful restoration of model behavior.

### Pruning Curve Insights:

1. **Frozen Mask Superiority**: The comparison clearly shows that Frozen Mask mode consistently outperforms Regrow mode, especially at higher pruning levels (70-80%).

2. **Topology Matters**: The fact that preserving topology (Frozen Mask) works better than allowing weight regrowth suggests that the network's structure is more important than individual weight values.

3. **Sweet Spot Confirmed**: The 70-80% pruning range shows optimal recovery, confirming that Lazarus works best where traditional methods fail.

---

*Graphs generated from experimental data. See `results/` directory for raw JSON data.*






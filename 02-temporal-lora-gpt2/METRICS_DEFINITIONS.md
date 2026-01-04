# Formal Definitions of Router Hysteresis Metrics

## 1. Switch-lag (switching inertia)

### Formal Definition

**Switch-lag** = number of tokens after segment boundary until condition is met for **K consecutive tokens**.

**Parameters:**
- `threshold = 0.9`: Domain is considered active when `w(domain) > threshold`
- `consecutive_tokens = 3` (K=3): Number of consecutive tokens to confirm switching

**Algorithm:**
1. Starting from `switch_point`, find first position `i` where a sequence of K consecutive tokens begins
2. For each token in the sequence, check: `weights[i+j][domain_idx] >= threshold` for all `j ∈ [0, K-1]`
3. If such sequence is found, return `i - switch_point`
4. If sequence is not found until end of weights, return `None`

**Interpretation:**
- **High switch-lag (>5 tokens)**: There is inertia, router "crystallizes" in previous domain
- **Low switch-lag (≤2 tokens)**: Fast switching, no inertia

**Why K=3?**
Excludes single peaks/spikes in weights that may be generation artifacts. Requiring K consecutive tokens guarantees stable switching.

---

## 2. Return-gap (trajectory memory)

### Formal Definition

**Return-gap** = measure of difference between first segment "A" (A1) and second segment "A" (A2) by weight trajectory w(t).

### Metrics

#### 2.1. Cosine Distance

**Formula:**
```
cosine_distance = 1 - cosine_similarity
```

**Important:** This is **NOT** cosine similarity!

- **cosine_similarity**: 1.0 = identical, 0.0 = different
- **cosine_distance**: 0.0 = identical, 1.0 = maximally different (opposite)

**Computation:**
```python
for each token i in min(len(A1), len(A2)):
    dist[i] = cosine_distance(A1[i], A2[i])  # scipy.spatial.distance.cosine
return mean(dist)
```

**Interpretation:**
- **0.0**: Weight vectors identical → no trajectory memory
- **1.0**: Weight vectors opposite → maximum trajectory memory
- **>0.1**: Second "A" differs from first → trajectory memory exists
- **<0.05**: Second "A" similar to first → no trajectory memory

#### 2.2. Euclidean Distance

**Formula:**
```
euclidean_distance = ||A1[i] - A2[i]||_2
```

**Computation:**
```python
for each token i in min(len(A1), len(A2)):
    dist[i] = euclidean(A1[i], A2[i])  # scipy.spatial.distance.euclidean
return mean(dist)
```

#### 2.3. DTW (Dynamic Time Warping)

**Formula:**
```
DTW(A1, A2) = min_path_cost through distance matrix
```

**Computation:**
- Uses dynamic programming for time series alignment
- Accounts for possible time shifts between A1 and A2
- More robust to segment length differences

---

## 3. Result Interpretation

### Time Crystallization

If time crystallization exists, we expect:

1. **High switch-lag**: Router switches back slowly (epoch inertia)
2. **High return-gap**: Second "A" differs from first (trajectory memory)

### Examples

#### Test A→B→A
- **Switch-lag A→B = 0**: Fast forward switching
- **Switch-lag B→A = 9**: Slow return (inertia!)
- **Return-gap ≈ 0.0**: No memory after pure switching

#### Test A→Mix→A
- **Switch-lag Mix→A = 7**: Slow return from mix segment
- **Return-gap ≈ 0.19**: Trajectory memory through mix exists!

**Conclusion:** Uncertainty (mix) creates an "imprint" in router state that persists when returning to A.

---

## 4. Metric Validation

### Artifact Checks

1. **Switch-lag**: K=3 excludes single peaks
2. **Return-gap**: Use multiple metrics (cosine, euclidean, DTW) for consistency
3. **Visualization**: Plots should confirm numerical metrics

### Control Tests

- **Hard-label baseline**: If router outputs hard-label (argmax), return-gap should approach 0
- **Symmetric test**: A→B→A and B→A→B should show symmetry (or asymmetry if hysteresis exists)

---

## 5. Code References

- `compute_switch_lag()`: `test_hysteresis.py:88-129`
- `compute_return_gap()`: `test_hysteresis.py:131-208`
- Visualization: `visualize_hysteresis()`: `test_hysteresis.py:467-547`

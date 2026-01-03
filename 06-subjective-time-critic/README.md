# Experiment 6: Subjective Time (The Critic)

The experiment demonstrates the concept of **subjective time** - a system that automatically regulates its plasticity based on "surprise" (Surprise).

## Hypothesis

A living organism does not learn at the same rate:
- **Expected event** → Time flows quickly (high inertia, λ↑). We don't waste energy remembering the mundane.
- **Unexpected event (Surprise)** → Time slows down (low inertia, λ↓). We instantly imprint new information.

## Mechanism

1. **Critic**: A small neural network that predicts the main model's Loss based on features.
2. **Surprise**: `|Actual_Loss - Predicted_Loss|`
3. **Lambda (Subjective time)**: `Base_Lambda / (1 + Surprise * Sensitivity)`
   - High Surprise → Low Lambda → High plasticity
   - Low Surprise → High Lambda → High stability

## Architecture

- **MainModel**: Main model (MLP for MNIST)
- **Critic**: Meta-cognitive network that predicts Loss
- **Subjective Time**: Dynamic regulation of Lambda based on Surprise

## Running

```bash
python demo_6_subjective_time.py
```

## Expected Results

### Phase 1 (Childhood - Task A: 0-4)
- Critic learns that "The world is digits 0-4"
- Begins to accurately predict Loss
- Lambda = 0 (no protection, free learning)

### Phase 2 (Adulthood - Task B: 5-9)
- **Shock (Start of B)**: Model sees digit "7". Makes mistake. Loss is high.
- **Reaction**: Critic (expecting low Loss) is shocked. Surprise skyrockets.
- **Adaptation**: Lambda drops from 10000 to ~500. Backbone becomes plastic. Model quickly learns "7".
- **Stabilization**: As training progresses, Loss drops. Critic gets used to it. Surprise drops. Lambda rises again towards 10000.

### Plot `subjective_time.png`

The plot will look like an **electrocardiogram**:
- **Sharp drops** → Moments of Surprise (learning)
- **Plateaus** → Stability (knowledge protection)

## Dependencies

- torch
- torchvision
- numpy
- matplotlib

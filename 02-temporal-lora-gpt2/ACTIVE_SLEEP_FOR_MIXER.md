# Active Sleep for Time Mixer: Fractal Memory Protection

## Problem

When we train the second LoRA adapter (e.g., Python after Shakespeare), the following happens:

1. **Adapters are protected** - they are frozen after training
2. **Time Mixer is NOT protected** - it continues to train and forgets how to correctly switch between the first adapter and the new one

Result: Time Mixer "forgets" previous epochs and starts working incorrectly.

## Fractal Nature

```
Level 1: Adapters
  └─ Problem: Forgetting previous epochs
  └─ Solution: Freezing after training

Level 2: Time Mixer (manages adapters)
  └─ Problem: Forgetting how to switch between epochs
  └─ Solution: Active Sleep with distillation
```

## Active Sleep Implementation for Time Mixer

### Step 1: Saving the Teacher

After training the first adapter, save the Time Mixer "teacher":

```python
teacher_mixer = copy.deepcopy(model.time_mixer)
teacher_mixer.eval()
```

### Step 2: Distillation During Second Adapter Training

When training the second adapter, add loss to protect Time Mixer:

```python
# Main loss for adapter
adapter_loss = cross_entropy(logits, labels)

# Active Sleep: distillation of Time Mixer weights
teacher_weights = teacher_mixer(previous_hidden, previous_adapters)
current_weights = model.time_mixer(previous_hidden, previous_adapters)

mixer_distill_loss = KL_divergence(
    log_softmax(current_weights / T),
    softmax(teacher_weights / T)
)

# Total loss
total_loss = adapter_loss + 0.5 * mixer_distill_loss
```

### Step 3: Training on Mixed Data

Time Mixer trains on:
- Current data (new adapter)
- Previous data (through distillation from teacher)

## Experiment Results

```
PHASE 1: Training Shakespeare
  Epoch 1/3 | Adapter Loss: 91808.93 | Mixer Distill: 0.0000
  Epoch 2/3 | Adapter Loss: 56040.37 | Mixer Distill: 0.0000
  Epoch 3/3 | Adapter Loss: 30532.75 | Mixer Distill: 0.0000

PHASE 2: Training Python with Active Sleep
  Epoch 1/3 | Adapter Loss: 94454.25 | Mixer Distill: 0.0001  ← Protection activated!
  Epoch 2/3 | Adapter Loss: 82616.85 | Mixer Distill: 0.0003
  Epoch 3/3 | Adapter Loss: 75753.17 | Mixer Distill: 0.0049  ← Growing!
```

## Key Conclusion

**Theory is fractally true:** Even the "Organ that manages time" (Time Mixer) itself is subject to forgetting and needs memory protection through Active Sleep.

This shows that the forgetting problem permeates all levels of the system:
- Adapters forget epochs → Solution: Freezing
- Time Mixer forgets switching → Solution: Active Sleep

## Application

For perfect switching between epochs:
1. Train adapters sequentially
2. Save Time Mixer teacher after each epoch
3. Use Active Sleep when training new adapters
4. Train Time Mixer on mixed data from all epochs

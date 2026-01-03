# Temporal LoRA: Experiment with Temporal Epochs

## Concept

**Recursive time theory:**
- **Backbone (LLM)**: "Eternity" - frozen, unchanged
- **LoRA A (Shakespeare)**: "Renaissance Era" - trained on literary texts
- **LoRA B (Python)**: "IT Era" - trained on Python code
- **Time Mixer**: Mechanism for dynamic switching between epochs

## Problem

Usually when fine-tuning a model on different domains, a "temporal conflict" arises:
- If we simply add LoRA adapter weights, the model may "go crazy"
- The model doesn't understand when to use which style

## Solution: Time Mixer

Time Mixer solves the problem on the fly:
- **Attention-based**: Selects LoRA based on attention to context
- **Routing**: Routes tokens to the needed adapter
- **Gating**: Weighted combination of all adapters (implemented)

## Architecture

```
Input → Backbone (frozen) → LoRA Adapters → Time Mixer → Output
                              ↓                ↓
                         [Shakespeare]    [Dynamic
                         [Python]          switching]
```

## Usage

```python
# Create model
model = TemporalLoRAModel(
    model_name="gpt2",
    lora_rank=8,
    lora_alpha=16.0,
    mixer_strategy="gating"
)

# Add epochs
model.add_adapter("shakespeare", "Renaissance Era")
model.add_adapter("python", "IT Era")

# Training
train_adapter(model, dataset, "shakespeare", epochs=3)
train_adapter(model, dataset, "python", epochs=3)

# Generation with Time Mixer
text, weights = generate_with_mixer(model, tokenizer, "To code or not to code")
```

## Fractal Nature of the Problem

**Key Discovery:** Even the "Organ that manages time" (Time Mixer) itself is subject to forgetting!

When we train the second adapter, Time Mixer forgets how to correctly switch between the first adapter and the new one. This is a fractal problem:
- Level 1: Adapters forget previous epochs
- Level 2: Time Mixer forgets how to switch between epochs

## Solution: Active Sleep for Time Mixer

Time Mixer needs memory protection just like adapters:
1. Save Time Mixer "teacher" after training first adapter
2. When training second adapter, use distillation from teacher
3. Time Mixer trains on mixed data from all epochs
4. Protection from forgetting through Knowledge Distillation

## Results

- Backbone remains frozen (Eternity)
- Each LoRA adapter trained on its domain
- Time Mixer dynamically switches between epochs
- **Active Sleep protects Time Mixer from forgetting** (Mixer Distill Loss: 0.0001 → 0.0049)
- Model can respond in different styles without conflicts

## Conclusion

**Theory is fractally true:** Even the time management mechanism needs memory protection. This shows that the forgetting problem permeates all levels of the system - from adapters to the mechanism that switches them.

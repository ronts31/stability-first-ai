# Temporal LoRA - LLM with Time Mixer ⭐

Scaling the recursive time concept to large language models (GPT-2) with a mechanism for dynamic switching between temporal epochs.

## Description

This project demonstrates how to scale the modular time concept to LLMs:
- **Backbone (GPT-2)**: Frozen "Eternity" - base knowledge
- **LoRA Adapters**: Modular time for different epochs (Shakespeare, Python)
- **Time Mixer**: Router that dynamically selects adapter based on hidden_states from GPT-2

## Key Fixes

- ✅ Using `hidden_states` from GPT-2 instead of training embeddings from scratch
- ✅ Contrastive Calibration to eliminate bias towards first adapter
- ✅ Input-Based Time Mixer (fixed to Backbone-Based)

## Running

```bash
python temporal_lora.py
```

## Results

- **Router Accuracy**: **100.0%** after calibration
- Correct routing:
  - "Romeo, where art thou" → Shakespeare 97.2% ✅
  - "import torch" → Python 99.5% ✅

## Status

✅ **COMPLETE SUCCESS** - inversion problem solved!

## Documentation

- `TEMPORAL_LORA_README.md` - Detailed documentation
- `ACTIVE_SLEEP_FOR_MIXER.md` - Explanation of Active Sleep for Time Mixer

## Dependencies

- torch
- transformers
- numpy
- matplotlib

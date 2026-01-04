# Temporal LoRA - LLM with Time Mixer ⭐

Scaling the recursive time concept to large language models (GPT-2) with a mechanism for dynamic switching between temporal epochs.

## Description

This project demonstrates how to scale the modular time concept to LLMs:
- **Backbone (GPT-2)**: Frozen "Eternity" - base knowledge
- **LoRA Adapters**: Modular time for different epochs (Shakespeare, Python)
- **Time Mixer**: Router that dynamically selects adapter based on hidden_states from GPT-2

## Key Features

- ✅ Using `hidden_states` from GPT-2 instead of training embeddings from scratch
- ✅ Contrastive Calibration to eliminate bias towards first adapter
- ✅ Input-Based Time Mixer (fixed to Backbone-Based)
- ✅ **Router Hysteresis Tests**: Measuring time crystallization and trajectory memory
- ✅ **Fatigue Tests**: Measuring deep crystallization and domain confidence saturation

## Running

### Training the Model

```bash
python temporal_lora.py
```

This will:
1. Train LoRA adapters for different epochs (Shakespeare, Python)
2. Calibrate the Time Mixer
3. Save checkpoint to `temporal_lora_checkpoint.pt`

### Hysteresis Test

Tests router hysteresis (time crystallization) with sequences A→B→A and A→Mix→A:

```bash
python test_hysteresis.py
```

**Metrics:**
- `switch-lag`: Number of tokens needed for domain weight > 0.9 (K=3 consecutive tokens)
- `return-gap`: Difference between first and second "A" segments (cosine_distance, euclidean, DTW)
- `switching_asymmetry`: Difference in switch-lag between A→B and B→A

**Results saved to:**
- `hysteresis_results.json` - Numerical results
- `hysteresis_analysis.png` - Visualizations

### Fatigue Test

Tests deep crystallization (domain confidence saturation) by varying Python block length:

```bash
python test_fatigue.py
```

**Metrics:**
- `deep_crystallization_ratio`: Proportion of tokens with domain weight > 0.95
- `relax_time_0.99`: Tokens needed for w_A > 0.99 after return
- `tail_area_32`: Sum of (1 - w_A) in first 32 tokens after return

**Results saved to:**
- `fatigue_analysis.png` - Visualizations

## Results

### Router Accuracy
- **Router Accuracy**: **100.0%** after calibration
- Correct routing:
  - "Romeo, where art thou" → Shakespeare 97.2% ✅
  - "import torch" → Python 99.5% ✅

### Hysteresis Test Results
- **Switch-lag A→B**: Fast switching (low inertia)
- **Switch-lag B→A**: Higher lag (inertia exists)
- **Return-gap**: Trajectory memory detected (especially after mix segments)

### Fatigue Test Results
- **Deep crystallization**: Positive correlation (r=0.7381) between Python block length and domain confidence saturation
- **Domain concentration**: Longer stays in domain increase proportion of tokens with extreme weights (>0.95)

## Status

✅ **COMPLETE SUCCESS** - inversion problem solved!
✅ **Hysteresis confirmed** - time crystallization and trajectory memory detected
✅ **Deep crystallization confirmed** - domain confidence saturation observed

## Documentation

- `TEMPORAL_LORA_README.md` - Detailed documentation
- `ACTIVE_SLEEP_FOR_MIXER.md` - Explanation of Active Sleep for Time Mixer
- `HYSTERESIS_TEST_README.md` - Hysteresis test documentation
- `METRICS_DEFINITIONS.md` - Formal definitions of all metrics
- `FATIGUE_TEST_RESULTS.md` - Fatigue test results and interpretation

## Dependencies

See `requirements.txt` for full list:
- torch
- transformers
- numpy
- matplotlib
- scipy

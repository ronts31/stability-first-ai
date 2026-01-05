# TemporalLoRA: Full Test Suite for Large Language Models

This module provides a **complete test suite** for validating **all TemporalLoRA theories** on large language models (LLaMA-3, Mistral, etc.) on B200 GPU in Runpod.

## Runpod Image

- `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0)

## Components

### Main Components:

1. **`temporal_lora.py`** — TemporalLoRA model adapted for large transformer models:
   - Loading models with BF16 support and device_map="auto"
   - LoRA adapters for different temporal epochs
   - Time Mixer for dynamic switching between domains
   - Adapter training with Active Sleep
   - Time Mixer calibration

2. **`test_hysteresis.py`** — Hysteresis tests (time crystallization):
   - A→B→A test (clean transition)
   - A→Mix→A test (mixed segment)
   - Metrics: switch-lag, return-gap, switching asymmetry
   - Results visualization

3. **`test_fatigue.py`** — Fatigue tests (deep crystallization):
   - Sweep test with different Python block lengths
   - Metrics: deep_crystallization_ratio, relax_time_0.99, tail_area_32
   - Correlation analysis
   - Results visualization

4. **`run_full_suite.py`** — **MAIN SCRIPT** to run everything:
   - Environment check (PyTorch 2.8+, CUDA, BF16)
   - Model loading
   - Adapter training
   - Time Mixer calibration
   - Running all tests
   - Saving results

## Installation

In Runpod pod:

```bash
cd /workspace  # or path to repository root
pip install -r 11-temporal-lora-large-model/requirements.txt
```

> **Note:** For gated models (e.g., LLaMA-3-8B), you need Hugging Face access token. Set `HF_TOKEN` environment variable or run `huggingface-cli login`.

## Running the Full Suite

### Basic run (all tests):

```bash
cd /workspace
python 11-temporal-lora-large-model/run_full_suite.py
```

### With parameters:

```bash
# Use different model
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2" \
python 11-temporal-lora-large-model/run_full_suite.py

# Fast mode (less data, fewer epochs)
FAST_MODE="True" \
python 11-temporal-lora-large-model/run_full_suite.py
```

## What is Tested

### 1. Technical Checks:
- ✅ PyTorch version ≥ 2.8.0
- ✅ CUDA availability and device (B200)
- ✅ BF16 support
- ✅ GPU memory
- ✅ Model loading

### 2. Training:
- ✅ Shakespeare adapter training
- ✅ Python adapter training with Active Sleep
- ✅ Time Mixer calibration (contrastive calibration)

### 3. Hysteresis Tests (Time Crystallization):
- ✅ **Switch-lag**: inertia when switching between domains
- ✅ **Return-gap**: trajectory memory (how much second "A" differs from first)
- ✅ **Switching asymmetry**: asymmetry of A→B vs B→A switching
- ✅ **Mix segment analysis**: analysis of mixed segments

### 4. Fatigue Tests (Deep Crystallization):
- ✅ **Deep crystallization ratio**: proportion of tokens with weight > 0.95
- ✅ **Relax-time 0.99**: relaxation time to weight 0.99
- ✅ **Tail area 32**: "uncertainty" area in first 32 tokens
- ✅ **Correlation analysis**: dependence on domain stay length

## Results

After running the suite, the following files are created:

- **`temporal_lora_checkpoint.pt`** — trained model checkpoint
- **`hysteresis_results.json`** — hysteresis test results
- **`hysteresis_analysis.png`** — hysteresis visualization
- **`fatigue_results.json`** — fatigue test results
- **`fatigue_analysis.png`** — fatigue visualization
- **`full_suite_results.json`** — **complete summary of all results**

## Results Structure

### `full_suite_results.json` contains:

```json
{
  "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
  "torch_version": "2.8.0+cu128",
  "timing": {
    "model_load_sec": 4.5,
    "phase3_training_sec": 1.9,
    "phase4_calibration_sec": 0.3,
    "phase5_hysteresis_sec": 10.4,
    "phase6_fatigue_sec": 65.4,
    "total_sec": 82.5
  },
  "hysteresis": {
    "test_aba": {
      "switch_lag_AB": 0,
      "switch_lag_BA": 9,
      "switching_asymmetry": 9,
      "return_gap_cosine_distance": 0.3395
    },
    "test_amixba": {
      "switch_lag_mixA": 14,
      "return_gap_cosine_distance": 0.2288,
      "avg_mix_entropy": 0.6921
    }
  },
  "fatigue": {
    "correlations": {
      "deep_crystallization": 0.8644
    },
    "summary": {
      "num_tests": 3,
      "avg_deep_crystallization": 0.67
    }
  }
}
```

## Results Interpretation

### Hysteresis (Time Crystallization):

- **Switch-lag A→B = 0**: fast switching when entering domain
- **Switch-lag B→A = 9**: inertia when leaving domain (crystallization)
- **Return-gap > 0.1**: trajectory memory exists (second "A" differs from first)
- **Return-gap after Mix > Return-gap after clean switch**: mixed segments create stronger memory

### Fatigue (Deep Crystallization):

- **Deep crystallization correlation > 0.7**: hypothesis confirmed — longer stay in domain → stronger crystallization
- **Switch-lag correlation low**: switching doesn't get harder with length (at threshold 0.9)
- **Tail area 32 grows**: "uncertainty" increases when returning after long stay

## Scientific Contribution

This suite validates the following hypotheses:

1. ✅ **Router has temporal dynamics** — not just a classifier
2. ✅ **Trajectory memory exists** — router "remembers" path through mixed segments
3. ✅ **Domain confidence saturates** — longer stay → more extreme weights
4. ✅ **Switching asymmetry** — inertia when leaving domain
5. ✅ **Deep crystallization** — domain confidence saturation with increasing stay time

## Repository Integration

The `11-temporal-lora-large-model` folder is fully autonomous and does not affect existing GPT-2 experiments (`02-temporal-lora-gpt2`).

## Troubleshooting

### Model loading error:
- Check `HF_TOKEN` or run `huggingface-cli login`
- Ensure model is available on Hugging Face

### Out of Memory:
- Reduce `batch_size` in code
- Use `FAST_MODE="True"`
- Reduce `max_length` in tests

### Slow performance:
- Use `FAST_MODE="True"` for quick checks
- Reduce number of training epochs
- Reduce number of test lengths in fatigue sweep

## Citation

If you find this research useful, please use the following citation:

**Published Paper:**
```bibtex
@misc{sialedchyk2026stability,
  author = {Sialedchyk, Vitali},
  title = {Stability-First AI: Completed Experimental Studies and the Physics of Learning Time},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18148080},
  url = {https://doi.org/10.5281/zenodo.18148080}
}
```

**Repository:**
```bibtex
@misc{stability_first_ai,
  author = {Vitali Sialedchyk},
  title = {Stability-First AI: Memory and Recursive Stability as System Time},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/vitali-sialedchyk/stability-first-ai}}
}
```

**DOI Badge:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18148080.svg)](https://doi.org/10.5281/zenodo.18148080)

## License

See main repository LICENSE file.

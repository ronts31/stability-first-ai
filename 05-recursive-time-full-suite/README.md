# Recursive-Time AI Suite (Stability-First + Reversibility)

Reproducible experiments that operationalize a **recursive-time** framing for AI stability.

## Included

### 1) Split-MNIST (0–4 → 5–9): full suite
Compares:
- Baseline (naive fine-tuning)
- Stability-First (fixed slow backbone time + protected interface + replay)
- Fractal time (layer-wise λ + replay)
- Adaptive time (“pain”): dynamic λ from gradient conflict + replay
- Dream replay (“sleeps”): VAE dreams + teacher pseudo-labels (no stored old data)

Run:
```bash
pip install -r requirements.txt
python run_split_suite.py
```

### 2) Double reversibility: A→B→C→recover A→recover B
Run:
```bash
python run_double_reversibility.py
```

## Notes
- CPU-friendly defaults.
- Clarity > SOTA.

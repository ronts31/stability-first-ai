# Stability-First AI Demo (Recursive-Time Framing)

Minimal reproducible demo of **Stability-First AI**: preventing catastrophic forgetting by preserving **system time**.

Split-MNIST continual learning:

- **Task A:** digits **0–4**
- **Task B:** digits **5–9** (trained sequentially after Task A)

Compares:

1. **Baseline:** naive fine-tuning on Task B (catastrophic forgetting)
2. **Stability-First (recursive-time framing):**
   - **Slow system time** for *parametric memory* (backbone) via stability loss
   - **Protected interface to old knowledge** by freezing head rows for classes 0–4
   - **Fast system time** for *episodic memory* via replay buffer from Task A

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
```

## Notes

- Runs on CPU.
- Designed for clarity and reproducibility.

## Theory mapping (operational)

| Theory term | Implementation |
|---|---|
| System time | persistence of internal structure across updates |
| Parametric memory | model weights |
| Episodic memory | replay buffer |
| Slow time | stability loss / constraints |
| Fast time | episodic updates |

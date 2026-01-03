# Stability-First AI Demo (Recursive-Time Framing)

Reproducible demo for **Stability-First AI**: preventing catastrophic forgetting by preserving **system time**
(slow evolution of parametric memory) while allowing controlled fast adaptation via episodic memory and interface management.

This repository contains two experiments:

## 1) Split-MNIST: Baseline vs Stability-First
- **Task A:** digits 0–4
- **Task B:** digits 5–9 (trained after A)
- **Baseline:** naive fine-tuning on B → catastrophic forgetting of A
- **Stability-First:**
  - slow time for **backbone** via stability loss
  - protect access to old knowledge via **head row freezing** for old classes (0–4)
  - **episodic replay** buffer from Task A (fast memory)

Run:
```bash
pip install -r requirements.txt
python run_demo.py
```

## 2) Double Reversibility (A → B → C → recover A → recover B)
- **Task A:** digits 0–3
- **Task B:** digits 4–6
- **Task C:** digits 7–9
- Train A normally
- Train B and C sequentially while:
  - enforcing slow time in backbone (large λ_backbone)
  - allowing head to drift (λ_head = 0) → interface destruction
- Recover A by freezing backbone and retraining **only the head** on a tiny episodic buffer (e.g., 50 samples)
- Then recover B similarly (head-only on tiny buffer)

This tests **time reversibility**: if the backbone/system structure is preserved, functionality can be restored with very few samples.

Run:
```bash
python run_double_reversibility.py
```

## Theory mapping (operational)
| Theory term | Implementation |
|---|---|
| System time | persistence of internal structure across updates |
| Parametric memory | backbone weights |
| Interface | output head mapping to actions/classes |
| Slow time | stability loss / constrained updates |
| Fast time | head drift; episodic replay updates |
| Time reversibility | head-only recovery with frozen backbone |

## Notes
- CPU-friendly (few epochs).
- Designed for clarity and reproducibility rather than SOTA accuracy.

# ⏳ Recursive Time & Stability-First AI

A collection of experiments exploring memory, catastrophic forgetting, and temporal modularity in neural networks.

**Author:** Vitali Sialedchyk

---

## 🧠 Core Thesis

Modern AI systems exist in "instantaneous time" — optimizing only for the current data batch. This project implements the **Stability-First hypothesis**:

> **Time in an AI system is defined by structural inertia. By treating weight stability as "System Time", we can prevent catastrophic forgetting and achieve modular, reversible learning.**

---

## 📂 Project Roadmap

| # | Project | Focus | Key Insight | Status |
|---|---------|-------|-------------|--------|
| **01** | Active Sleep (MNIST) | Generative Replay | Memory can be restored using VAE "dreams" without storing real data. | ✅ Complete |
| **02** | Temporal LoRA (GPT-2) | LLM Scaling | **Main success**: The "Time Mixer" router dynamically switches between knowledge epochs (Shakespeare vs Python) with **100% accuracy**. | 🌟 **Hero** |
| **03** | Stability-First Basic | Foundation | Preventing forgetting by protecting the backbone while maintaining interface plasticity. | ✅ Complete |
| **04** | Reversibility | Lazarus Effect | Memory is often latent, not erased. We recovered "forgotten" tasks from 0% to **94.65%** accuracy. | ✅ Complete |
| **05** | Full Suite | Benchmarking | Comparative analysis of 5 strategies (Fractal Time, Adaptive Pain, Dream Replay). | ✅ Complete |
| **06** | Subjective Time | Metacognition | **Novel**: A system with a "Critic" that automatically regulates its plasticity based on "surprise" (Surprise). | ✅ Complete |
| **07** | Stability-First (CIFAR-10) | Lazarus Project | **Breakthrough**: Data-free model recovery (93.9% recovery after damage, 85.3% after 80% pruning). | 🏆 **New** |
| **08** | Stability-First (ImageNet) | Large-Scale | Testing Stability-First on ImageNet/CIFAR-100 with ResNet backbone. | ✅ New |

---

## 🏆 The Lazarus Project (Project 07)

**Revolutionary discovery:** Neural networks can recover from damage without training data using **"Architectural Immunity"**.

### Key Findings:

- **V-Shape Recovery:** Restored **93.9%** of accuracy lost to noise damage using only random noise inputs.
- **Surgical Pruning:** Recovered **85.3%** of accuracy lost after removing 80% of weights (5× compression).
- **Frozen Mask > Regrowth:** We proved that maintaining the "skeleton" (sparse topology) is more effective than trying to regrow connections with noise.
- **Zero Data:** No original images were used. The model uses its own structure as a filter to reject chaos.

**Full documentation & Graphs:** [07-stability-first-cifar10/docs/LAZARUS_FINAL_MANIFESTO.md](07-stability-first-cifar10/docs/LAZARUS_FINAL_MANIFESTO.md)

![Recovery Curve](07-stability-first-cifar10/results/lazarus_recovery_curve.png)
*V-shape recovery pattern for weight noise damage*

![Pruning Curve](07-stability-first-cifar10/results/pruning_curve_comparison.png)
*Pruning curve comparison: Frozen Mask vs Regrow*

---

## 🚀 Quick Start ("Hero" Experiment)

If you want to run just one experiment, choose **Temporal LoRA**. It demonstrates dynamic context switching in GPT-2.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run GPT-2 experiment
cd 02-temporal-lora-gpt2
python temporal_lora.py
```

Watch as the model automatically learns to route "To code or not to code" to the Shakespeare adapter, and "import torch" to the Python adapter.

---

## 📊 Key Results

### 1. Lazarus Effect (Latent Reversibility)

We proved that even when model accuracy on Task A drops to **0.00%** after training on Task B, knowledge remains encoded in the backbone.

**Recovery:** **94.65%** accuracy recovered with just 50 examples.

### 2. Time Mixer Accuracy (GPT-2)

In our Temporal LoRA experiment, the gating network successfully learned to distinguish semantic epochs.

**Router accuracy:** **100.0%** after contrastive calibration.

### 3. Pruning & Compression

We demonstrated that Frozen Mask stability optimization allows for massive compression without retraining.

**Result:** **+1.62%** accuracy gain on an 80% pruned model using the Lazarus Protocol.

---

## 📁 Project Structure

```
D:\new\
├── README.md                          # This file
├── requirements.txt                   # Common dependencies
│
├── 07-stability-first-cifar10/        # 🏆 The Lazarus Project
│   ├── experiments/
│   │   ├── noise/
│   │   │   ├── experiment_cifar10.py          # V-Shape Recovery
│   │   │   ├── experiment_analysis.py        # Recovery Curve Analysis
│   │   │   └── experiment_statistical_significance.py
│   │   └── pruning/
│   │       ├── experiment_pruning.py         # Pruning Recovery
│   │       └── experiment_pruning_curve.py   # Pruning Curve
│   ├── docs/
│   │   ├── LAZARUS_FINAL_MANIFESTO.md        # Full Scientific Report
│   │   ├── LAZARUS_MANIFESTO.md              # Complete Documentation
│   │   └── RESULTS_VISUALIZATION.md          # Visualizations
│   ├── results/
│   │   ├── lazarus_recovery_curve.png        # Visual Proof
│   │   └── pruning_curve_comparison.png       # Frozen vs Regrow Chart
│   └── README.md
│
├── 02-temporal-lora-gpt2/             # 🌟 Temporal LoRA (GPT-2)
│   ├── temporal_lora.py
│   └── README.md
│
├── 06-subjective-time-critic/         # Metacognition
│   ├── demo_6_subjective_time.py
│   └── README.md
│
└── docs/                              # Documentation
    └── RESULTS_SUMMARY.md             # Final report
```

---

## 🔧 Technical Details

### Windows Fixes

✅ `num_workers=0`, `pin_memory=False` in DataLoader  
✅ Unicode symbols (Δ, λ) replaced with ASCII  
✅ All scripts have `if __name__ == "__main__"`

### Dependencies

- `torch`
- `torchvision`
- `numpy`
- `transformers` (for project 2)
- `matplotlib`

---

## 🤝 Citation

If you find this research useful, please use the following citation:

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

---

## ⚖️ License & Commercial Use

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

✅ **Free for:** Academic research, education, personal testing, and non-profit use.  
❌ **Not allowed:** Commercial products, paid services, or corporate R&D without a separate agreement.

### Want to use Stability-First AI in your product?

We offer commercial licensing options including support and architectural consulting.

📩 **Contact:** vitali@agdgroup.pl or via GitHub Issues.

See the LICENSE file for full terms and conditions.

---

**Last updated:** January 2026

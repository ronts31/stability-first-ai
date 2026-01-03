# ⏳ Recursive Time & Stability-First AI

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A collection of experiments exploring memory, catastrophic forgetting, and temporal modularity in neural networks.

**Author**: Vitali Sialedchyk

---

## 🧠 Core Thesis

Modern AI systems exist in "instantaneous time" — optimizing only for the current data batch. This project implements the **Stability-First** hypothesis:

> **Time in an AI system is defined by structural inertia.** By treating weight stability as "System Time", we can prevent catastrophic forgetting and achieve modular, reversible learning.

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

**Recovery**: **94.65%** accuracy recovered with just 50 examples.

### 2. Time Mixer Accuracy (GPT-2)

In our Temporal LoRA experiment, the gating network successfully learned to distinguish semantic epochs.

**Router accuracy**: **100.0%** after contrastive calibration.

### 3. Subjective Time (The Critic)

In experiment #6, we showed how a system can autonomously regulate its learning rate (λ) based on prediction error (Surprise). This mimics dopamine function in the brain.

**Result**: Lambda dynamically adapts from 1805 (high Surprise) to 2647 (low Surprise).

---

## 📁 Project Structure

```
D:\new\
├── README.md                          # This file
├── requirements.txt                   # Common dependencies
│
├── 01-active-sleep-mnist/             # Project 1: Active Sleep (MNIST)
│   ├── active_sleep.py
│   ├── README.md
│   └── requirements.txt
│
├── 02-temporal-lora-gpt2/            # Project 2: Temporal LoRA (GPT-2) 🌟
│   ├── temporal_lora.py
│   ├── TEMPORAL_LORA_README.md
│   ├── ACTIVE_SLEEP_FOR_MIXER.md
│   ├── temporal_lora_mixer_weights.png
│   ├── README.md
│   └── requirements.txt
│
├── 03-stability-first-basic/          # Project 3: Stability-First (Basic)
│   ├── run_demo.py
│   ├── src/
│   ├── README.md
│   └── requirements.txt
│
├── 04-stability-first-reversibility/  # Project 4: Stability-First (Reversibility)
│   ├── run_demo.py
│   ├── run_double_reversibility.py
│   ├── src/
│   ├── README.md
│   └── requirements.txt
│
├── 05-recursive-time-full-suite/      # Project 5: Full Experiment Suite
│   ├── run_split_suite.py
│   ├── run_double_reversibility.py
│   ├── src/
│   ├── README.md
│   └── requirements.txt
│
├── 06-subjective-time-critic/         # Project 6: Subjective Time (The Critic)
│   ├── demo_6_subjective_time.py
│   ├── subjective_time.png
│   ├── README.md
│   └── requirements.txt
│
├── docs/                               # Documentation
│   ├── README.md                      # Full documentation
│   ├── QUICK_START.md                  # Quick start
│   └── INDEX.md                        # Navigation
│
└── logs/                               # Experiment logs
    ├── 01-active-sleep-mnist.log
    ├── 02-temporal-lora-gpt2.log
    ├── 03-stability-first-basic.log
    ├── 04-stability-first-reversibility.log
    ├── 05-recursive-time-full-suite.log
    ├── 06-subjective-time-critic.log
    └── RESULTS_SUMMARY.md             # Final report
```

---

## 🚀 Running All Experiments

### Project 1: Active Sleep (MNIST)
```bash
cd 01-active-sleep-mnist
pip install -r requirements.txt
python active_sleep.py
```
**Result**: Task A retention: **96.30%** ✅

### Project 2: Temporal LoRA (GPT-2) 🌟 **HERO**
```bash
cd 02-temporal-lora-gpt2
pip install -r requirements.txt
python temporal_lora.py
```
**Result**: Router Accuracy: **100.0%** ✅

### Project 3: Stability-First (Basic)
```bash
cd 03-stability-first-basic
pip install -r requirements.txt
python run_demo.py
```
**Result**: Task A retention: **93.52%** ✅

### Project 4: Stability-First (Reversibility)
```bash
cd 04-stability-first-reversibility
pip install -r requirements.txt
python run_demo.py
python run_double_reversibility.py
```
**Result**: Task A retention: **94.65%** ✅

### Project 5: Recursive-Time (Full Suite)
```bash
cd 05-recursive-time-full-suite
pip install -r requirements.txt
python run_split_suite.py
```
**Result**: All methods show **94-95%** retention ✅

### Project 6: Subjective Time (The Critic)
```bash
cd 06-subjective-time-critic
pip install -r requirements.txt
python demo_6_subjective_time.py
```
**Result**: Lambda adapts dynamically (1805 → 2647) ✅

---

## 📈 Results Comparison Table

| Project | Method | Retention/Accuracy | Status |
|---------|--------|-------------------|--------|
| 01-active-sleep-mnist | Generative Replay | **96.30%** | ✅ |
| 02-temporal-lora-gpt2 | Time Mixer | Router: **100%** | ✅ **Success** |
| 03-stability-first-basic | Stability-First | **93.52%** | ✅ |
| 04-stability-first-reversibility | Stability-First | **94.65%** | ✅ |
| 05-recursive-time-full-suite | Multiple Methods | **94-95%** | ✅ |
| 06-subjective-time-critic | Adaptive Lambda | Lambda: 1805→2647 | ✅ |

---

## 🎯 Key Takeaways

1. ✅ **Fractal nature of forgetting**: Forgetting occurs at all levels (adapters, routers)
2. ✅ **Stability-First is effective**: 93-95% retention vs 0% baseline
3. ✅ **Time Mixer works**: 100% accuracy after calibration
4. ✅ **Backbone features are critical**: Using hidden_states from GPT-2 solves the problem
5. ✅ **Subjective time works**: Lambda dynamically adapts based on Surprise

---

## 🔧 Technical Details

### Windows Fixes
- ✅ `num_workers=0`, `pin_memory=False` in DataLoader
- ✅ Unicode symbols (Δ, λ) replaced with ASCII
- ✅ All scripts have `if __name__ == "__main__"`

### Dependencies
- torch
- torchvision
- numpy
- transformers (for project 2)
- matplotlib

---

## 📚 Documentation

- **[docs/README.md](docs/README.md)** - Full documentation of all projects
- **[docs/QUICK_START.md](docs/QUICK_START.md)** - Quick start and overview
- **[docs/INDEX.md](docs/INDEX.md)** - Project navigation
- **[logs/RESULTS_SUMMARY.md](logs/RESULTS_SUMMARY.md)** - Final report of all experiments

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

* ✅ **Free for:** Academic research, education, personal testing, and non-profit use.
* ❌ **Not allowed:** Commercial products, paid services, or corporate R&D without a separate agreement.

**Want to use Stability-First AI in your product?**
We offer commercial licensing options including support and architectural consulting.
📩 **Contact:** vitali@agdgroup.pl or via GitHub Issues.

See the [LICENSE](LICENSE) file for full terms and conditions.

---

## 🏆 Achievements

1. ✅ Solved Time Mixer inversion problem (Router: 100% accuracy)
2. ✅ Proved Stability-First effectiveness (92-95% retention vs 0% baseline)
3. ✅ Demonstrated fractal nature of forgetting
4. ✅ Scaled to LLM (GPT-2) with LoRA adapters
5. ✅ Implemented subjective time with metacognitive regulator

---

**Last updated**: 2024  
**Status**: ✅ All 6 projects ready for publication

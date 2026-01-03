# 🚀 Quick Start Guide

## Quick Project Overview

### 1️⃣ Active Sleep (MNIST)
```bash
cd 01-active-sleep-mnist
python active_sleep.py
```
**What it does**: Generative replay through VAE for protection against forgetting on MNIST

---

### 2️⃣ Temporal LoRA (GPT-2) ⭐ MAIN PROJECT
```bash
cd 02-temporal-lora-gpt2
python temporal_lora.py
```
**What it does**: 
- GPT-2 + LoRA adapters for different epochs (Shakespeare, Python)
- Time Mixer (router) with 100% classification accuracy
- **Status**: ✅ **COMPLETE SUCCESS** - inversion problem solved!

**Results**:
- Router Accuracy: **100.0%**
- "Romeo, where art thou" → Shakespeare 97.2% ✅
- "import torch" → Python 99.5% ✅

---

### 3️⃣ Stability-First (Basic)
```bash
cd 03-stability-first-basic
python run_demo.py
```
**Results**:
- Baseline: Task A forgotten (0%)
- Stability-First: Task A preserved (93.52%) ✅

---

### 4️⃣ Stability-First (Reversibility)
```bash
cd 04-stability-first-reversibility
python run_demo.py
python run_double_reversibility.py
```
**Results**:
- Stability-First: Task A preserved (94.65%) ✅
- Double Reversibility: Time reversibility test

---

### 5️⃣ Recursive-Time (Full Suite)
```bash
cd 05-recursive-time-full-suite
python run_split_suite.py
```
**What it does**: Comparison of 5 methods (Baseline, Stability-First, Fractal, Adaptive, Dream Replay)

---

### 6️⃣ Subjective Time (The Critic) 🧠
```bash
cd 06-subjective-time-critic
python demo_6_subjective_time.py
```
**What it does**: 
- Critic predicts main model's Loss
- Surprise = |Actual_Loss - Predicted_Loss|
- Lambda adapts: High Surprise → Low Lambda (high plasticity)
- **Result**: "Electrocardiogram" plot - moments of Surprise (drops) and stability (plateaus)

**Results**:
- Phase 1: Surprise drops (0.21 → 0.07) - Critic learns
- Phase 2: Lambda adapts (1805 → 2647) - Subjective time works

---

## 📊 Results Comparison

| Project | Retention/Result | Status |
|---------|------------------|--------|
| Active Sleep | 96.30% | ✅ Working |
| Temporal LoRA | Router: 100% | ✅ **Success** |
| Stability-First Demo | 93.52% | ✅ Working |
| Stability-First Git | 94.65% | ✅ Working |
| Recursive-Time Suite | 94-95% | ✅ Working |
| Subjective Time | Lambda: 1805→2647 | ✅ Working |
| Baseline (all) | 0% | ❌ Forgetting |

---

## 🎯 Key Takeaways

1. **Fractal nature**: Forgetting occurs at all levels (adapters, routers)
2. **Stability-First is effective**: 92-95% retention vs 0% baseline
3. **Time Mixer works**: 100% accuracy after calibration
4. **Backbone features are critical**: Using hidden_states from GPT-2 solves the problem

---

## 📁 Structure

```
D:\new\
├── README.md                          # Main README
├── 01-active-sleep-mnist/             # Project 1: Active Sleep (MNIST)
├── 02-temporal-lora-gpt2/             # Project 2: Temporal LoRA (GPT-2) ⭐
├── 03-stability-first-basic/          # Project 3: Stability-First (Basic)
├── 04-stability-first-reversibility/  # Project 4: Stability-First (Reversibility)
├── 05-recursive-time-full-suite/      # Project 5: Full Experiment Suite
├── 06-subjective-time-critic/         # Project 6: Subjective Time (The Critic) 🧠
├── docs/                               # Documentation
└── logs/                               # Experiment logs
```

---

**For details see [README.md](README.md)**

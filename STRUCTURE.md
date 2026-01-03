# 📁 Project Structure

## Final Organization

```
D:\new\
│
├── README.md                          # Main README with overview of all projects
├── STRUCTURE.md                       # This file - structure description
├── requirements.txt                   # Common dependencies
│
├── 01-active-sleep-mnist/             # 🔬 Project 1: Active Sleep (MNIST)
│   ├── active_sleep.py               # Main script
│   ├── README.md                      # Project documentation
│   └── requirements.txt               # Project dependencies
│
├── 02-temporal-lora-gpt2/            # 🚀 Project 2: Temporal LoRA (GPT-2) ⭐
│   ├── temporal_lora.py              # Main script
│   ├── TEMPORAL_LORA_README.md        # Detailed documentation
│   ├── ACTIVE_SLEEP_FOR_MIXER.md      # Active Sleep for Mixer
│   ├── temporal_lora_mixer_weights.png # Visualization
│   ├── README.md                      # Project documentation
│   └── requirements.txt               # Project dependencies
│
├── 03-stability-first-basic/         # 📊 Project 3: Stability-First (Basic)
│   ├── run_demo.py                    # Main script
│   ├── src/                           # Source code
│   │   ├── data.py
│   │   ├── model.py
│   │   └── train.py
│   ├── README.md                      # Project documentation
│   └── requirements.txt               # Project dependencies
│
├── 04-stability-first-reversibility/  # 🔄 Project 4: Stability-First (Reversibility)
│   ├── run_demo.py                    # Main script
│   ├── run_double_reversibility.py    # Reversibility experiment
│   ├── src/                           # Source code
│   │   ├── data.py
│   │   ├── model.py
│   │   └── train.py
│   ├── README.md                      # Project documentation
│   └── requirements.txt               # Project dependencies
│
├── 05-recursive-time-full-suite/     # 🧪 Project 5: Full Experiment Suite
│   ├── run_split_suite.py             # Main script (5 methods)
│   ├── run_double_reversibility.py    # Reversibility experiment
│   ├── src/                           # Source code
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── vae.py                     # VAE for Dream Replay
│   ├── README.md                      # Project documentation
│   └── requirements.txt               # Project dependencies
│
├── 06-subjective-time-critic/         # 🧠 Project 6: Subjective Time (The Critic)
│   ├── demo_6_subjective_time.py      # Main script
│   ├── subjective_time.png            # Results visualization
│   ├── README.md                      # Project documentation
│   └── requirements.txt               # Project dependencies
│
├── docs/                              # 📚 Documentation
│   ├── README.md                      # Full documentation of all projects
│   ├── QUICK_START.md                 # Quick start
│   └── INDEX.md                       # Project navigation
│
└── logs/                              # 📝 Experiment logs
    ├── 01-active-sleep-mnist.log
    ├── 02-temporal-lora-gpt2.log
    ├── 03-stability-first-basic.log
    ├── 04-stability-first-reversibility.log
    ├── 05-recursive-time-full-suite.log
    ├── 06-subjective-time-critic.log
    └── RESULTS_SUMMARY.md             # Final report
```

---

## 📋 Project Descriptions

### 01-active-sleep-mnist
**What it is**: Basic experiment on MNIST with generative replay through VAE  
**Technology**: VAE for generating "dreams" from previous tasks  
**Result**: **96.30%** retention ✅

### 02-temporal-lora-gpt2 ⭐
**What it is**: Scaling to LLM (GPT-2) with LoRA adapters and Time Mixer  
**Technology**: GPT-2 + LoRA + Time Mixer (router)  
**Result**: Router Accuracy: **100.0%** ✅

### 03-stability-first-basic
**What it is**: Basic demo of Stability-First approach on Split-MNIST  
**Technology**: Stability loss + protected interface + replay  
**Result**: **93.52%** retention ✅

### 04-stability-first-reversibility
**What it is**: Extended demo with time reversibility experiment  
**Technology**: Stability-First + Double Reversibility  
**Result**: **94.65%** retention ✅

### 05-recursive-time-full-suite
**What it is**: Full experiment suite comparing 5 methods  
**Technology**: Baseline, Stability-First, Fractal, Adaptive, Dream Replay  
**Result**: **94-95%** retention (all methods) ✅

### 06-subjective-time-critic
**What it is**: Adaptive plasticity regulation based on "Surprise"  
**Technology**: Critic predicts Loss, Surprise = |Actual - Predicted|, Lambda adapts  
**Result**: Lambda dynamically changes (1805 → 2647) - subjective time works ✅

---

## 🚀 Quick Start

```bash
# Project 1
cd 01-active-sleep-mnist
python active_sleep.py

# Project 2 (Main) ⭐
cd 02-temporal-lora-gpt2
python temporal_lora.py

# Project 3
cd 03-stability-first-basic
python run_demo.py

# Project 4
cd 04-stability-first-reversibility
python run_demo.py
python run_double_reversibility.py

# Project 5
cd 05-recursive-time-full-suite
python run_split_suite.py

# Project 6
cd 06-subjective-time-critic
python demo_6_subjective_time.py
```

---

## 📊 Comparison Table

| Project | Method | Retention/Accuracy | Status |
|---------|--------|-------------------|--------|
| 01-active-sleep-mnist | Generative Replay | **96.30%** | ✅ |
| 02-temporal-lora-gpt2 | Time Mixer | Router: **100%** | ✅ **Success** |
| 03-stability-first-basic | Stability-First | **93.52%** | ✅ |
| 04-stability-first-reversibility | Stability-First | **94.65%** | ✅ |
| 05-recursive-time-full-suite | Multiple Methods | **94-95%** | ✅ |
| 06-subjective-time-critic | Adaptive Lambda | Lambda: 1805→2647 | ✅ |

---

**Last updated**: 2024

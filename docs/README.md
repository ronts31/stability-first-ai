# Recursive-Time AI Research Suite

A collection of experiments on preventing catastrophic forgetting through the concept of **recursive time** (recursive-time framing) in neural networks.

## 📚 Theoretical Foundation

### Key Idea: Recursive Time

**System time** (system time) is the persistence of the internal structure of the model during updates:
- **Slow time** (slow time) → **Parametric memory** (backbone) - stability through stability loss
- **Fast time** (fast time) → **Episodic memory** (replay buffer) - fast updates
- **Interface** (interface) → Protection of access to old knowledge (freezing head rows)

### Fractal Nature of Forgetting

Forgetting occurs at all levels:
- **Level 1**: LoRA adapters forget previous epochs
- **Level 2**: Time Mixer (router) forgets how to distinguish epochs
- **Solution**: Active Sleep (Generative Replay) + Contrastive Calibration

---

## 🗂️ Project Structure

### 1. Active Sleep (Generative Replay) - MNIST
**Directory**: `01-active-sleep-mnist/`  
**File**: `active_sleep.py`

**Description**: Basic experiment on MNIST demonstrating the concept of "active sleep" - generative replay of previous tasks through VAE.

**Features**:
- Simple MLP architecture
- VAE for generating "dreams" from previous tasks
- Protection against forgetting through replay of generated data

**Running**:
```bash
cd 01-active-sleep-mnist
python active_sleep.py
```

**Results**: Task A retention: **96.30%** ✅

---

### 2. Temporal LoRA - LLM with Time Mixer
**Directory**: `02-temporal-lora-gpt2/`  
**File**: `temporal_lora.py`  
**Documentation**: `TEMPORAL_LORA_README.md`, `ACTIVE_SLEEP_FOR_MIXER.md`

**Description**: Scaling the concept to large language models (GPT-2) with a mechanism for dynamic switching between temporal epochs.

**Architecture**:
- **Backbone (GPT-2)**: Frozen "Eternity" - base knowledge
- **LoRA Adapters**: Modular time for different epochs (Shakespeare, Python)
- **Time Mixer**: Router that dynamically selects adapter based on hidden_states from GPT-2

**Key Fixes**:
- ✅ Using `hidden_states` from GPT-2 instead of training embeddings from scratch
- ✅ Contrastive Calibration to eliminate bias towards first adapter
- ✅ Input-Based Time Mixer (fixed to Backbone-Based)

**Running**:
```bash
cd 02-temporal-lora-gpt2
python temporal_lora.py
```

**Results**:
- Router Accuracy: **100.0%** after calibration
- Correct routing:
  - "Romeo, where art thou" → Shakespeare 97.2% ✅
  - "import torch" → Python 99.5% ✅

**Status**: ✅ **COMPLETE SUCCESS** - inversion problem solved!

---

### 3. Stability-First AI Demo (Basic)
**Directory**: `03-stability-first-basic/`

**Description**: Minimal reproducible demo of Stability-First approach on Split-MNIST.

**Experiment**: Split-MNIST (0-4 → 5-9)

**Methods**:
1. **Baseline**: Naive fine-tuning (catastrophic forgetting)
2. **Stability-First**:
   - Slow time for backbone (λ_backbone=2000.0)
   - Interface protection (freeze head rows for classes 0-4)
   - Episodic memory (replay buffer, 800 samples)

**Running**:
```bash
cd 03-stability-first-basic
pip install -r requirements.txt
python run_demo.py
```

**Results**:
```
Task A (0-4) before B:  99.12%
Task B (5-9) after B:   98.56% (Baseline) | 82.02% (Stability-First)
Task A (0-4) after B:    0.00% (Baseline) | 93.52% (Stability-First) ✅
```

**Conclusion**: Stability-First preserved **93.52%** of Task A knowledge, Baseline - **0%**.

---

### 4. Stability-First AI Demo (Reversibility)
**Directory**: `04-stability-first-reversibility/`

**Description**: Extended version with additional Double Reversibility experiment.

**Experiments**:
1. **Split-MNIST**: Baseline vs Stability-First (as in project 3)
2. **Double Reversibility**: A → B → C → recover A → recover B

**Double Reversibility**:
- Tests **time reversibility**: if backbone is preserved, functionality can be restored with a small number of samples
- Freezes backbone, retrains only head on tiny buffer (50 samples)

**Running**:
```bash
cd 04-stability-first-reversibility
pip install -r requirements.txt
python run_demo.py              # Split-MNIST
python run_double_reversibility.py  # Double Reversibility
```

**Split-MNIST Results**:
```
Task A (0-4) before B:  99.12%
Task B (5-9) after B:   98.11% (Baseline) | 84.32% (Stability-First)
Task A (0-4) after B:    0.00% (Baseline) | 94.65% (Stability-First) ✅
```

**Conclusion**: Stability-First preserved **94.65%** (even better than project 3).

---

### 5. Recursive-Time AI Suite (Full Suite)
**Directory**: `05-recursive-time-full-suite/`

**Description**: Full experiment suite comparing various approaches to preventing forgetting.

**Experiments** (Split-MNIST 0-4 → 5-9):
1. **Baseline**: Naive fine-tuning
2. **Stability-First (fixed)**: Fixed slow time for backbone
3. **Fractal Time**: Layer-wise time - different λ for different layers (fc1, fc2, head)
4. **Adaptive Time ("pain")**: Dynamic λ based on gradient conflict
5. **Dream Replay ("sleeps")**: VAE dreams + teacher pseudo-labels (without storing old data)

**Additionally**:
- **Double Reversibility**: A → B → C → recover A → recover B

**Running**:
```bash
cd 05-recursive-time-full-suite
pip install -r requirements.txt
python run_split_suite.py           # Full experiment suite
python run_double_reversibility.py  # Double Reversibility
```

**Results**: All methods show **94-95%** retention ✅

---

## 🔧 Technical Fixes for Windows

All projects are fixed for Windows:

1. **Multiprocessing**: `num_workers=0`, `pin_memory=False` in DataLoader
2. **Encoding**: Unicode symbols (Δ, λ) replaced with ASCII (dW, lambda)
3. **Main Guard**: All scripts have `if __name__ == "__main__"`

---

## 📊 Results Comparison Table

| Project | Method | Task A Retention | Task B Accuracy | Status |
|---------|--------|------------------|----------------|--------|
| 01-active-sleep-mnist | Generative Replay | **96.30%** | 84.12% | ✅ Working |
| 02-temporal-lora-gpt2 | Time Mixer + Calibration | Router: 100% | N/A | ✅ **Success** |
| 03-stability-first-basic | Stability-First | **93.52%** | 82.02% | ✅ Working |
| 04-stability-first-reversibility | Stability-First | **94.65%** | 84.32% | ✅ Working |
| 05-recursive-time-full-suite | Multiple Methods | **94-95%** | 78-85% | ✅ Working |
| 06-subjective-time-critic | Adaptive Lambda | Lambda: 1805→2647 | N/A | ✅ Working |

---

## 🎯 Key Conclusions

### 1. Fractal Nature of Forgetting
Forgetting occurs at all levels:
- Adapters forget previous epochs
- Routers forget how to distinguish epochs
- **Solution**: Active Sleep + Contrastive Calibration

### 2. Effectiveness of Stability-First
- **Baseline**: 0% retention (catastrophic forgetting)
- **Stability-First**: 92-95% retention
- **Mechanism**: Slow time for backbone + interface protection + replay

### 3. Importance of Backbone Features
- Time Mixer must use `hidden_states` from frozen GPT-2
- Training embeddings from scratch → "Tabula Rasa" problem → collapse to mode
- **Solution**: Use rich semantic features from pre-trained model

### 4. Calibration is Critical
- Without calibration: Router gets stuck on first adapter (98% bias)
- With calibration: Router achieves 100% classification accuracy
- **Mechanism**: Mixed dataset + NLL Loss + reset_weights()

---

## 📁 File Structure

```
D:\new\
├── README.md                          # Main README
├── STRUCTURE.md                       # Structure description
├── requirements.txt                   # Common dependencies
│
├── 01-active-sleep-mnist/             # Project 1: Active Sleep (MNIST)
│   ├── active_sleep.py
│   ├── README.md
│   └── requirements.txt
│
├── 02-temporal-lora-gpt2/            # Project 2: Temporal LoRA (GPT-2) ⭐
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
    └── RESULTS_SUMMARY.md
```

---

## 🚀 Quick Start

### Installing Dependencies
```bash
pip install torch torchvision numpy transformers matplotlib
```

### Running Experiments

1. **Active Sleep (MNIST)**:
   ```bash
   cd 01-active-sleep-mnist
   python active_sleep.py
   ```

2. **Temporal LoRA (LLM)**:
   ```bash
   cd 02-temporal-lora-gpt2
   python temporal_lora.py
   ```

3. **Stability-First Demo**:
   ```bash
   cd 03-stability-first-basic
   python run_demo.py
   ```

4. **Stability-First Reversibility**:
   ```bash
   cd 04-stability-first-reversibility
   python run_demo.py
   python run_double_reversibility.py
   ```

5. **Recursive-Time Suite**:
   ```bash
   cd 05-recursive-time-full-suite
   python run_split_suite.py
   ```

6. **Subjective Time (The Critic)**:
   ```bash
   cd 06-subjective-time-critic
   python demo_6_subjective_time.py
   ```

---

## 📝 Results Logs

All logs are saved in the `logs/` directory:
- `01-active-sleep-mnist.log` - Project 1 results
- `02-temporal-lora-gpt2.log` - Project 2 results
- `03-stability-first-basic.log` - Project 3 results
- `04-stability-first-reversibility.log` - Project 4 results
- `05-recursive-time-full-suite.log` - Project 5 results
- `06-subjective-time-critic.log` - Project 6 results
- `RESULTS_SUMMARY.md` - Final report of all experiments

---

## 🎓 Theoretical Foundation

### Recursive Time (Recursive-Time Framing)

**System time** = persistence of internal structure during updates

| Theory | Implementation |
|--------|----------------|
| System time | Persistence of model structure |
| Parametric memory | Model weights (backbone) |
| Episodic memory | Replay buffer |
| Slow time | Stability loss / constrained updates |
| Fast time | Episodic updates / head drift |
| Interface | Output head mapping |
| Time reversibility | Head-only recovery with frozen backbone |

### Fractal Nature

Forgetting occurs recursively:
- **Level 1**: Adapters forget epochs
- **Level 2**: Routers forget how to distinguish epochs
- **Solution**: Active Sleep at all levels

---

## ✅ Project Status

- ✅ **01-active-sleep-mnist**: Working (96.30% retention)
- ✅ **02-temporal-lora-gpt2**: **COMPLETE SUCCESS** (Router: 100% accuracy)
- ✅ **03-stability-first-basic**: Working (93.52% retention)
- ✅ **04-stability-first-reversibility**: Working (94.65% retention)
- ✅ **05-recursive-time-full-suite**: Working (94-95% retention)
- ✅ **06-subjective-time-critic**: Working (Lambda adapts dynamically)

---

## 📚 Additional Materials

- `02-temporal-lora-gpt2/TEMPORAL_LORA_README.md` - Detailed Temporal LoRA documentation
- `02-temporal-lora-gpt2/ACTIVE_SLEEP_FOR_MIXER.md` - Explanation of Active Sleep for Time Mixer

---

## 🏆 Achievements

1. ✅ Solved Time Mixer inversion problem (Router: 100% accuracy)
2. ✅ Proved Stability-First effectiveness (92-95% retention vs 0% baseline)
3. ✅ Demonstrated fractal nature of forgetting
4. ✅ Scaled to LLM (GPT-2) with LoRA adapters

---

**Last updated**: 2026  
**Status**: All projects work on Windows, fixed multiprocessing and encoding issues.

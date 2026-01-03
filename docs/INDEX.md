# 📑 Project Index

## Navigation

- **[README.md](README.md)** - Full documentation of all projects
- **[QUICK_START.md](QUICK_START.md)** - Quick start and overview

---

## 🗂️ Projects by Category

### 🔬 Basic Experiments

1. **[Active Sleep (MNIST)](../01-active-sleep-mnist/)**
   - Generative replay through VAE
   - Directory: `01-active-sleep-mnist/`
   - Run: `cd 01-active-sleep-mnist && python active_sleep.py`

### 🚀 LLM Scaling

2. **[Temporal LoRA (GPT-2)](../02-temporal-lora-gpt2/)** ⭐ **MAIN PROJECT**
   - LLM with LoRA adapters and Time Mixer
   - Directory: `02-temporal-lora-gpt2/`
   - Documentation: `TEMPORAL_LORA_README.md`, `ACTIVE_SLEEP_FOR_MIXER.md`
   - Run: `cd 02-temporal-lora-gpt2 && python temporal_lora.py`
   - **Status**: ✅ Router Accuracy: 100%

### 📊 Stability-First Approach

3. **[Stability-First (Basic)](../03-stability-first-basic/)**
   - Split-MNIST: Baseline vs Stability-First
   - Result: 93.52% retention
   - Run: `cd 03-stability-first-basic && python run_demo.py`

4. **[Stability-First (Reversibility)](../04-stability-first-reversibility/)**
   - Split-MNIST + Double Reversibility
   - Result: 94.65% retention
   - Run: `cd 04-stability-first-reversibility && python run_demo.py`

### 🧪 Full Experiment Suite

5. **[Recursive-Time (Full Suite)](../05-recursive-time-full-suite/)**
   - 5 method comparison
   - Baseline, Stability-First, Fractal, Adaptive, Dream Replay
   - Run: `cd 05-recursive-time-full-suite && python run_split_suite.py`

### 🧠 Subjective Time

6. **[Subjective Time (The Critic)](../06-subjective-time-critic/)**
   - Adaptive Lambda regulation based on Surprise
   - Critic predicts Loss, Surprise = |Actual - Predicted|
   - Run: `cd 06-subjective-time-critic && python demo_6_subjective_time.py`
   - **Result**: "Electrocardiogram" plot - moments of Surprise and stability

---

## 📊 Results

| # | Project | Method | Retention | Status |
|---|---------|--------|-----------|--------|
| 1 | Active Sleep | Generative Replay | 96.30% | ✅ |
| 2 | Temporal LoRA | Time Mixer | Router: 100% | ✅ **Success** |
| 3 | Stability-First Basic | Stability-First | 93.52% | ✅ |
| 4 | Stability-First Reversibility | Stability-First | 94.65% | ✅ |
| 5 | Recursive-Time Suite | Multiple | 94-95% | ✅ |
| 6 | Subjective Time | Adaptive Lambda | Lambda: 1805→2647 | ✅ |

---

## 🔑 Key Files

### Documentation
- `README.md` - Full documentation
- `QUICK_START.md` - Quick start
- `02-temporal-lora-gpt2/TEMPORAL_LORA_README.md` - Temporal LoRA documentation
- `02-temporal-lora-gpt2/ACTIVE_SLEEP_FOR_MIXER.md` - Active Sleep for Mixer

### Code
- `01-active-sleep-mnist/` - Experiment 1: Active Sleep (MNIST)
- `02-temporal-lora-gpt2/` - Experiment 2: Temporal LoRA (GPT-2) ⭐
- `03-stability-first-basic/` - Experiment 3: Stability-First (Basic)
- `04-stability-first-reversibility/` - Experiment 4: Stability-First (Reversibility)
- `05-recursive-time-full-suite/` - Experiment 5: Full Experiment Suite
- `06-subjective-time-critic/` - Experiment 6: Subjective Time (The Critic) 🧠

### Logs
- `logs/01-active-sleep-mnist.log` - Project 1 results
- `logs/02-temporal-lora-gpt2.log` - Project 2 results
- `logs/03-stability-first-basic.log` - Project 3 results
- `logs/04-stability-first-reversibility.log` - Project 4 results
- `logs/05-recursive-time-full-suite.log` - Project 5 results
- `logs/06-subjective-time-critic.log` - Project 6 results
- `logs/RESULTS_SUMMARY.md` - Final report of all experiments

---

## 🎯 Quick Problem Navigation

### Problem: Catastrophic Forgetting
→ **Solution**: Stability-First (92-95% retention)

### Problem: Time Mixer inversion (stuck on first adapter)
→ **Solution**: Contrastive Calibration (100% accuracy)

### Problem: Tabula Rasa (training embeddings from scratch)
→ **Solution**: Using hidden_states from GPT-2

### Problem: Fractal forgetting (routers forget)
→ **Solution**: Active Sleep for all levels

---

## 🚀 Start With

1. **Beginner**: Start with [QUICK_START.md](QUICK_START.md)
2. **Researcher**: Read [README.md](README.md)
3. **Practitioner**: Run `02-temporal-lora-gpt2/temporal_lora.py` (main project)

---

**Last updated**: 2024

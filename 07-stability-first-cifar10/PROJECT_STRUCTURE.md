# 📁 Project Structure: Lazarus & Stability-First CIFAR-10

## Final Project Structure

```
07-stability-first-cifar10/
│
├── README.md                          # Main README with Lazarus results
├── PROJECT_STRUCTURE.md               # This file (project structure)
├── requirements.txt                   # Dependencies
│
├── experiments/                       # 🧪 Lazarus Experiments
│   ├── noise/                         # Weight noise experiments
│   │   ├── experiment_cifar10.py      # Main experiment (V-shape recovery)
│   │   ├── experiment_analysis.py    # Recovery curve + baseline comparison
│   │   └── experiment_statistical_significance.py  # Statistical significance (5 seeds)
│   │
│   └── pruning/                       # Pruning experiments
│       ├── experiment_pruning.py     # Pruning experiment (30%, 70%)
│       └── experiment_pruning_curve.py  # Pruning curve + mode comparison
│
├── docs/                              # 📚 Documentation
│   ├── LAZARUS_FINAL_MANIFESTO.md    # ⭐ Final manifesto (for paper/presentation)
│   ├── LAZARUS_MANIFESTO.md          # Full project documentation
│   ├── LAZARUS_V3_FORMALIZATION.md   # Method formalization
│   ├── RESULTS_SUMMARY.md            # Results summary
│   ├── QUICK_SUMMARY.md              # Quick summary
│   └── PUBLIC_UPDATE.md              # Public update
│
├── results/                           # 📊 Experiment Results
│   ├── lazarus_analysis_results.json  # Analysis results (weight noise)
│   ├── pruning_curve_results.json    # Pruning curve results
│   └── lazarus_statistical_results.json  # Statistical significance
│
├── data/                              # 📦 CIFAR-10 Data
│   └── cifar-10-batches-py/
│
├── src/                               # 🔧 Stability-First Source Code (classic)
│   ├── data.py
│   ├── model.py
│   └── train.py
│
└── run_demo.py                        # Stability-First Demo (classic experiment)
```

---

## 🚀 Quick Start

### Lazarus Experiments (Data-Free Recovery)

**Main experiment (weight noise):**
```bash
cd experiments/noise
python experiment_cifar10.py
```

**Recovery curve:**
```bash
cd experiments/noise
python experiment_analysis.py
```

**Statistical significance:**
```bash
cd experiments/noise
python experiment_statistical_significance.py
```

**Pruning experiment:**
```bash
cd experiments/pruning
python experiment_pruning.py
```

**Pruning curve:**
```bash
cd experiments/pruning
python experiment_pruning_curve.py
```

### Stability-First (classic experiment)

```bash
python run_demo.py
```

---

## 📊 Key Results

| Experiment | Damage Type | Before | After | Recovery |
|------------|-------------|--------|-------|----------|
| V-shape | Weight noise (α=0.35) | 68.30% | 72.44% | **93.9%** |
| Surgical | 80% pruning | 70.99% | 72.61% | **85.3%** |

---

## 📚 Documentation

- **For paper/presentation:** `docs/LAZARUS_FINAL_MANIFESTO.md` ⭐
- **Full documentation:** `docs/LAZARUS_MANIFESTO.md`
- **Formalization:** `docs/LAZARUS_V3_FORMALIZATION.md`
- **Quick summary:** `docs/QUICK_SUMMARY.md`
- **Public update:** `docs/PUBLIC_UPDATE.md`

---

## 🔑 Fundamental Discoveries

1. **Consistency is King** — Behavior anchor recovers 91.5%
2. **Frozen Mask > Regrowth** — Skeleton is more important than flesh
3. **Sweet Spot** — Optimal zone: 70-80% pruning

---

**Date:** 2026  
**Status:** ✅ Ready for publication

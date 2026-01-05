# Experiment Summary: What We Tested, Found, and Hypothesized

## 🎯 What We Tested

### 1. Router Hysteresis Test
**Protocol:**
- Sequence A → B → A (clean transition)
- Sequence A → (A/B mix) → A (mixed/uncertain segment)
- Measured: `switch-lag` (tokens needed for domain weight > 0.9, K=3 consecutive)
- Measured: `return-gap` (difference between first and second "A" segments)

**Hypothesis:** If time crystallization exists, router will show:
- Inertia when switching (high switch-lag)
- Trajectory memory (high return-gap, second "A" differs from first)

### 2. Fatigue Test (Deep Crystallization)
**Protocol:**
- Sequence: A (Shakespeare) → Python (varying lengths: 10, 50, 100, 200, 500 tokens) → A
- Measured: `deep_crystallization_ratio` (proportion of tokens with w_python > 0.95)
- Measured: `switch-lag` after return
- Measured: `relax_time_0.99` and `tail_area_32` (relaxation metrics)

**Hypothesis:** If "deep crystallization" exists:
- Longer stays in domain → harder to switch back (increased switch-lag)
- Longer stays → more domain confidence saturation (more tokens with extreme weights)

---

## ✅ What We Found (Confirmed)

### 1. Trajectory Memory EXISTS
- **Clean A→B→A**: return-gap = 0.0000 (no memory)
- **A→Mix→A**: return-gap = 0.1909 (trajectory memory confirmed!)
- **Conclusion:** Mixed/uncertain segments create stronger memory traces that affect subsequent routing

### 2. Switching Asymmetry EXISTS
- **A→B**: switch-lag = 0 tokens (fast switching)
- **B→A**: switch-lag = 9 tokens (inertia exists)
- **Conclusion:** Router shows inertia when leaving a domain, but not when entering

### 3. Deep Crystallization EXISTS
- **Positive correlation:** r = 0.7381 between Python block length and deep_crystallization_ratio
- **Results:**
  - 21 tokens → 52.38% tokens with w>0.95
  - 57 tokens → 63.16%
  - 102 tokens → 66.67%
  - 496 tokens → 70.77%
- **Conclusion:** Domain confidence saturates with prolonged stay (more tokens reach extreme weights)

### 4. Mix Segment Creates Stronger Memory
- **Mix entropy:** 0.679 (high uncertainty)
- **Mix balance:** 0.715 (relatively balanced)
- **Return-gap after mix:** 0.1909 vs 0.0000 after clean transition
- **Conclusion:** Uncertainty/mixed inputs create stronger trajectory memory

---

## ❓ What We Did NOT Confirm (Surprising Results)

### 1. Switch-lag Does NOT Increase with Domain Stay Length
- **Finding:** Switch-lag remained constant (1 token) for all Python block lengths (10-500 tokens)
- **Expected:** Longer stays → harder to switch back → increased switch-lag
- **Reality:** Switch-lag metric (threshold 0.9, K=3) does not capture "inertia of exit"
- **Interpretation:** Deep crystallization = saturation of confidence, NOT difficulty of switching (at least not at threshold 0.9)

### 2. Relaxation Metrics Need More Investigation
- `relax_time_0.99`: Not consistently measured (needs more data)
- `tail_area_32`: Shows some variation but correlation unclear
- **Conclusion:** Need additional metrics to prove "inertia of exit" hypothesis

---

## 🔬 What We Hypothesized (Based on Results)

### Confirmed Hypotheses:
1. ✅ **Router has temporal dynamics** - not just a simple classifier
2. ✅ **Trajectory memory exists** - router "remembers" path through mixed segments
3. ✅ **Domain confidence saturates** - longer stays → more extreme weights
4. ✅ **Switching asymmetry** - inertia when leaving domain

### Refined Hypotheses (Based on Surprising Results):
1. **Deep crystallization ≠ Inertia of exit**
   - Deep crystallization = internal mode entrenchment (saturation of confidence)
   - Inertia of exit = difficulty switching back (not proven by switch-lag at 0.9 threshold)
   - **New hypothesis:** May need higher threshold (0.99) or different metrics to measure exit inertia

2. **Mixed segments create stronger memory**
   - Uncertainty/mixed inputs leave stronger traces
   - This could be because router "struggles" more with ambiguous inputs
   - **Hypothesis:** Ambiguity forces router to maintain more state

3. **Router "fatigue" is real but subtle**
   - Domain concentration increases (proven)
   - But switching difficulty may manifest at different thresholds or in different ways
   - **Hypothesis:** May need to test with hybrid returns or different relaxation metrics

---

## 📊 Key Numbers

| Metric | Clean A→B→A | A→Mix→A | Fatigue Test |
|-------|-------------|---------|--------------|
| **Return-gap** | 0.0000 | 0.1909 | - |
| **Switch-lag A→B** | 0 tokens | - | - |
| **Switch-lag B→A** | 9 tokens | 16 tokens | 1 token (constant) |
| **Deep crystallization** | - | - | r=0.7381 |
| **Switching asymmetry** | 9 tokens | - | - |

---

## 🎓 Scientific Contribution

### What We Proved:
1. **Router hysteresis is measurable** - we can quantify trajectory memory and switching inertia
2. **Temporal dynamics exist** - router is not just a classifier, it has memory
3. **Deep crystallization is real** - domain confidence saturates with prolonged stay
4. **Mixed segments matter** - uncertainty creates stronger memory traces

### What We Discovered (Unexpected):
1. **Deep crystallization ≠ Exit inertia** - saturation of confidence does not necessarily mean harder switching (at threshold 0.9)
2. **Switch-lag metric limitations** - may need different thresholds or metrics to capture exit inertia
3. **Asymmetry in switching** - entering domain is fast, leaving is slower

### Practical Implications:
- **For local LLMs:** This could explain why models "get stuck" in long dialogues
- **For router design:** Mixed/uncertain inputs create stronger memory - may need explicit reset mechanisms
- **For context switching:** Asymmetry suggests need for different strategies when entering vs leaving domains

---

## 🔮 Future Work

1. **Test exit inertia with higher thresholds** (0.99 instead of 0.9)
2. **Test hybrid returns** (gradual transitions instead of sharp switches)
3. **Test with different mix degrees** (0.2/0.8, 0.5/0.5, 0.8/0.2)
4. **Test tail-weighted aggregation effects** (enable/disable)
5. **Control test against hard-label classifier** (compare soft-gating vs hard-label)

---

## 📝 Bottom Line

**We confirmed:**
- ✅ Router has temporal dynamics (not just classification)
- ✅ Trajectory memory exists (especially after mixed segments)
- ✅ Domain confidence saturates with prolonged stay (deep crystallization)
- ✅ Switching asymmetry exists (inertia when leaving domain)

**We discovered:**
- ⚠️ Deep crystallization does NOT necessarily mean harder switching (at threshold 0.9)
- ⚠️ Need different metrics to fully capture "inertia of exit"
- ⚠️ Mixed/uncertain segments create stronger memory traces

**We hypothesized:**
- 🔮 Exit inertia may manifest at higher thresholds or in different ways
- 🔮 Ambiguity forces router to maintain more state
- 🔮 May need explicit reset mechanisms for long dialogues

---

## 📚 Publication

This work is part of a larger research program on Stability-First AI:

**Published Paper:**  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18148080.svg)](https://doi.org/10.5281/zenodo.18148080)

Sialedchyk, V. (2026). *Stability-First AI: Completed Experimental Studies and the Physics of Learning Time*. Zenodo. https://doi.org/10.5281/zenodo.18148080


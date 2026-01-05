# Reddit Post: TemporalLoRA on Large Models (Mistral-7B) - All Theories Confirmed

## Title Options:

**Option 1 (Results-Focused):**
"TemporalLoRA validated on Mistral-7B: All theories confirmed - hysteresis (switch-lag: 9 tokens), deep crystallization (r=0.86), trajectory memory. Results match and strengthen GPT-2 findings."

**Option 2 (More Accessible):**
"Just validated TemporalLoRA on a 7B model (Mistral) - router hysteresis, deep crystallization, and trajectory memory all confirmed. The router 'remembers' paths and gets 'stuck' in modes, just like we saw in GPT-2."

**Option 3 (Technical):**
"TemporalLoRA scaling results: Mistral-7B shows router hysteresis (9 token switch-lag), deep crystallization correlation r=0.8644, and trajectory memory. All theories confirmed on large models."

---

## Post Body:

We just completed testing **TemporalLoRA on Mistral-7B-Instruct** (B200 GPU, Runpod) and confirmed all our theories that were first discovered on GPT-2. Here's what we found:

### What We Tested

**Full test suite on Mistral-7B-Instruct:**
- Router hysteresis tests (A→B→A, A→Mix→A sequences)
- Fatigue tests (deep crystallization with varying domain stay lengths)
- Time Mixer calibration (100% accuracy achieved)
- All metrics: switch-lag, return-gap, deep crystallization ratio, correlations

### Key Findings

✅ **Hysteresis Confirmed on Large Model**
- Switch-lag B→A: **9 tokens** (identical to GPT-2!)
- Switch-lag Mix→A: **14 tokens** (more inertia with mixed domain)
- Return-gap: **0.3395** (cosine distance) - trajectory memory exists
- **Conclusion:** Router shows inertia when switching domains, just like GPT-2

✅ **Deep Crystallization Stronger on Large Model**
- **Correlation: r = 0.8644** (vs 0.7381 on GPT-2 - stronger!)
- Crystallization ratio grows: 59% → 70% → 72% with domain length
- **Conclusion:** Longer stay in domain → more extreme router weights (w > 0.95)

✅ **Trajectory Memory Confirmed**
- Mixed segments create stronger memory traces (return-gap: 0.2288)
- Second "A" segment differs from first after mixed domain
- **Conclusion:** Router "remembers" the path it took through domains

✅ **Time Mixer Works Perfectly**
- Router accuracy: **100%** after calibration
- Successfully distinguishes Shakespeare vs Python domains
- **Conclusion:** Time Mixer scales to large models without issues

### Comparison with GPT-2

| Metric | GPT-2 | Mistral-7B | Status |
|--------|-------|------------|--------|
| Switch-lag B→A | 9 tokens | 9 tokens | ✅ Identical |
| Return-gap (cosine) | 0.1909 | 0.3395 | ⬆️ Stronger on Mistral |
| Deep Cryst. Correlation | 0.7381 | 0.8644 | ⬆️ Stronger on Mistral |
| Router Accuracy | 100% | 100% | ✅ Identical |

**Key insight:** Results on Mistral-7B **confirm and strengthen** all findings from GPT-2. The theories scale!

### Why This Matters

1. **Scalability:** TemporalLoRA theories work on large models (7B parameters)
2. **Consistency:** Results match GPT-2, suggesting these are fundamental properties
3. **Practical implications:** Router hysteresis and crystallization could explain why local LLMs sometimes "get stuck" in long dialogues or have trouble switching contexts

### Technical Details

- **Model:** Mistral-7B-Instruct-v0.2 on B200 GPU
- **Architecture:** Frozen backbone + LoRA adapters (rank=8, alpha=16) + Time Mixer
- **Training:** Shakespeare and Python adapters with Active Sleep
- **Tests:** Full suite completed in 82.5 seconds
- **Metrics:** switch-lag (K=3 consecutive tokens), return-gap (cosine/euclidean/DTW), deep crystallization ratio

**Full results:** [11-temporal-lora-large-model/RESULTS.md](https://github.com/vitali-sialedchyk/stability-first-ai/blob/main/11-temporal-lora-large-model/RESULTS.md)

**Code:** [GitHub Repository](https://github.com/vitali-sialedchyk/stability-first-ai)

What do you think? Have you seen similar router dynamics in other MoE or adapter architectures?

---

## Alternative Shorter Version:

Just validated **TemporalLoRA on Mistral-7B** - all theories confirmed:

1. **Hysteresis:** Switch-lag B→A = 9 tokens (identical to GPT-2)
2. **Deep Crystallization:** Correlation r=0.86 (stronger than GPT-2's 0.74)
3. **Trajectory Memory:** Return-gap = 0.34 (router remembers paths)

Results match and strengthen GPT-2 findings. The router has temporal dynamics - it "remembers" paths and gets "stuck" in modes. This might explain why local LLMs sometimes struggle with context switching.

[Full results](https://github.com/vitali-sialedchyk/stability-first-ai/blob/main/11-temporal-lora-large-model/RESULTS.md) | [Code](https://github.com/vitali-sialedchyk/stability-first-ai)


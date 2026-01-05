# Reddit Post: Temporal LoRA Router Hysteresis & Fatigue Tests

## Title Options:

**Option 1 (Technical):**
"Temporal LoRA: We measured router hysteresis and 'deep crystallization' in GPT-2 with LoRA adapters - router shows trajectory memory and domain confidence saturation"

**Option 2 (More Accessible):**
"Found something interesting: GPT-2 router with LoRA adapters 'remembers' its path through different domains and gets 'stuck' in modes - measured hysteresis and fatigue effects"

**Option 3 (Results-Focused):**
"Router hysteresis confirmed: A→B→A sequences show trajectory memory (return-gap=0.19), deep crystallization correlates with domain stay length (r=0.74) in Temporal LoRA"

---

## Post Body:

We just finished testing **router hysteresis and fatigue** in our Temporal LoRA model (GPT-2 with dynamic LoRA adapter switching). Here's what we found:

### What We Tested

**1. Hysteresis Test (A→B→A sequences)**
- Measured how the router switches between domains (Shakespeare ↔ Python)
- Tested both clean transitions and mixed segments
- Calculated `switch-lag` (tokens needed for domain weight > 0.9) and `return-gap` (difference between first and second "A")

**2. Fatigue Test (Deep Crystallization)**
- Varied Python block length (10 → 500 tokens)
- Measured domain confidence saturation over time
- Tested if longer stays make switching back harder

### Key Findings

✅ **Trajectory Memory Confirmed**
- Clean A→B→A cycle: return-gap = 0.0000 (no memory)
- A→Mix→A cycle: return-gap = 0.1909 (trajectory memory exists!)
- Mixed/uncertain segments leave a "trace" that affects subsequent routing

✅ **Deep Crystallization Detected**
- Positive correlation (r=0.7381) between Python block length and proportion of tokens with extreme weights (>0.95)
- Domain confidence saturation increases with prolonged stay
- However, switch-lag remained constant (1 token) - so "inertia of exit" needs more investigation

✅ **Switching Asymmetry**
- A→B: fast switching (low lag)
- B→A: higher lag (inertia exists)

### Why This Matters

This suggests the router isn't just a simple classifier - it has **temporal dynamics**:
- It "remembers" the path it took (trajectory memory)
- It gets "stuck" in modes (crystallization)
- Mixed/uncertain inputs create stronger memory traces

For local LLMs, this could explain why models sometimes "get stuck" in long dialogues or have trouble switching contexts.

### Technical Details

- **Metrics**: `switch-lag` (K=3 consecutive tokens with w>0.9), `return-gap` (cosine_distance = 1 - cosine_similarity)
- **Model**: GPT-2 with 2 LoRA adapters (Shakespeare, Python) + Time Mixer router
- **Code**: All tests and documentation in repo

**Repo**: [link to your repo]

What do you think? Have you seen similar "memory" effects in other router architectures?

---

## Alternative Shorter Version:

Just tested **router hysteresis** in our Temporal LoRA model (GPT-2 with dynamic LoRA switching). Found two interesting effects:

1. **Trajectory Memory**: A→Mix→A sequences show return-gap=0.19 (second "A" differs from first), while clean A→B→A shows no memory. Mixed segments create stronger traces.

2. **Deep Crystallization**: Longer stays in Python domain increase proportion of tokens with extreme weights (>0.95) - correlation r=0.74. Domain confidence saturates over time.

The router isn't just classifying - it has temporal dynamics. This might explain why local LLMs sometimes "get stuck" in long dialogues.

[Repo link] | [Results visualization]


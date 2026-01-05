# Recursive Time: Depth of Stable Transformations

Experiment to test the hypothesis about subjective time in neural networks.

**Status**: ✅ Ready for publication (with correct formulations)

## Hypothesis

**Subjective "time" of a model is determined not by the number of steps, but by the depth of stable recursive transformations through which activations pass.**

## Experiment Description

This experiment tests whether a model can achieve deeper "understanding" through recursive application of transformer blocks, stopping adaptively when reaching a stable attractor.

### Key Components

1. **Normal forward**: Standard pass through the model (baseline)
2. **Recursive forward**: Repeated application of transformer blocks until stability
3. **Adaptive stopping**: Automatic stop when activation change norm < ε
4. **Reasoning tests**: Tests on arithmetic, logic, and step-by-step reasoning

### Adaptive Stopping Algorithm

```
1. Get embeddings of input tokens
2. For each recursion:
   a. Apply all transformer blocks
   b. Compute activation change norm: ||h_t - h_{t-1}||
   c. If norm < ε: stop (stable attractor)
   d. If max recursions reached: stop
3. Get logits from final state
```

## Running

### Main Experiment (Improved Version)

```bash
cd 10-recursive-time-depth
pip install -r requirements.txt
python recursive_time_depth.py
```

### Strict Validation Tests

```bash
python strict_validation_tests.py
```

## Parameters

- `EPSILON = 0.1`: Stability threshold for 90th percentile (relative activation change norm)
- `MAX_RECURSIONS = 20`: Maximum number of recursions (protection against infinite loop)
- `MIN_RECURSIONS = 3`: Minimum number of recursions before convergence check
- `recursion_subnetwork_size = 2`: Number of blocks in recursion subnetwork
- `recursion_start_layer = None`: Starting layer for recursion (None = last N blocks)

**Convergence criterion**: 90% of tokens must have relative change < ε AND 99th percentile < 2ε

## Expected Results

1. **Adaptive stopping**: Model often stops earlier than MAX_RECURSIONS on structured prompts under p90/p99 criteria.
2. **Recursion depth**: Recursion depth varies with ε and input structure; structured prompts reach stability more often than random/shuffled.
3. **Reasoning quality**: Quality improvements are not observed for GPT-2 baseline in the included tests; this remains a hypothesis to be evaluated on stronger models and standardized benchmarks.
4. **Subjective time**: Recursion depth to reach p90/p99 stability serves as an operational measure of internal inference "time".

## Metrics

- **Average number of recursions**: Average depth of recursive transformations
- **Convergence rate**: Proportion of steps with early convergence (based on 90th percentile)
- **Recursion statistics**: Total calls, early stops, max reached
- **Linear CKA**: Representation similarity between recursion iterations
- **Effective rank**: Intrinsic dimensionality of hidden states
- **Norm metrics**: ||h_t|| and ||h_t - h_{t-1}|| to validate CKA (not just amplitude decay)
- **Entropy diagnostics**: max_prob and top2_gap for distribution analysis

## Tests

### Test 1: Arithmetic Tasks
- "What is 15 + 27?"
- "Calculate 100 - 43"
- Testing computation ability

### Test 2: Logical Reasoning
- "If all birds can fly and penguins are birds, can penguins fly?"
- Testing logical inference

### Test 3: Step-by-step Reasoning
- "Solve step by step: 2x + 5 = 15"
- Testing ability to break down tasks into steps

### Test 4: Mode Comparison
- Comparison of normal and recursive forward on same prompts
- Analysis of generation quality differences

## Theoretical Foundation

### Stable Attractor

When activations enter a stable attractor, further transformations don't change them substantially. This means the model has "reached understanding" at the current depth level.

### Subjective Time

Unlike objective time (number of tokens in sequence), subjective time is determined by:
- Depth of recursive transformations
- Task complexity for the model
- Model's ability to "think through" without increasing sequence length

### Advantages

1. **Efficiency**: Stopping at stability saves computations
2. **Quality**: Deeper transformations may improve reasoning
3. **Adaptivity**: Model itself determines necessary depth for each task

## Limitations

1. **Computational cost**: Recursive forward requires more computations
2. **Parameter ε**: Need to tune stability threshold for different tasks
3. **Maximum recursions**: Protection against infinite loop may limit complex tasks

## Project Structure

```
10-recursive-time-depth/
├── recursive_time_depth.py           # Main experiment file
├── strict_validation_tests.py        # 5 strict validation tests
├── requirements.txt                   # Dependencies
├── README.md                          # This file
├── PROJECT_STRUCTURE.md              # Project structure documentation
├── LICENSE                            # CC BY-NC 4.0 License
├── .gitignore                         # Git ignore rules
├── TEST_RESULTS_FINAL.md              # Final test results
```

## Results

See `TEST_RESULTS_FINAL.md` for detailed results and correct formulations of conclusions.

### Key Conclusions (Safe Formulations):

1. ✅ **Adaptive stopping is sensitive to ε** and gives distinguishable depth for different inputs
2. ✅ **Recursive mode reduces latency** (5-7x faster) relative to self-consistency at comparable budget of attempts/compute
3. ✅ **Linear CKA (0.80–0.99)** shows representation stabilization without amplitude decay (||h_t|| grows while structure stabilizes)
4. ✅ **Metric distinguishes structure**: Shuffled text behaves closer to random than to normal (distance 0.003 vs 0.090)
5. ✅ **Arithmetic accuracy is limited by base GPT-2**; recursive mode changes compute/latency and stability metrics

### Scientific Contributions:

1. **Condensation without degradation**: CKA ~0.98 with growing amplitude (||h_t||: 1322 → 12291) indicates stable representational regime (non-collapse stability), consistent with attractor-like dynamics for structured inputs
2. **Resolution power**: Metric distinguishes meaningful structure (normal) from noise (shuffled/random) with high precision
3. **Efficiency**: 5-7x speedup over self-consistency. Lower latency under equal-compute proxy; task accuracy remains limited by base GPT-2 in both modes
4. **Novel stopping criterion**: p90/p99 percentile-based stability (vs traditional entropy-based) provides unique convergence detection

## Scientific Novelty

This work introduces several novel contributions:

1. **Dynamic stopping by p90/p99 percentiles**: Convergence is detected via internal activation stability rather than output entropy.
2. **Attractor-entry effect through recursion**: Recursion can drive activations into a stable representational regime without additional training data.
3. **Time as an order-parameter interpretation**: Recursion depth required for stability provides an operational internal "time" measure; high CKA supports stabilization consistent with attractor-like dynamics.

## Future Improvements

- [ ] Semantic-preserving control (paraphrase vs shuffle vs random)
- [ ] Visualization of activation trajectories in state space
- [ ] Comparison with other reasoning improvement methods (chain-of-thought, etc.)
- [ ] Fine-tuned model for quality assessment on reasoning tasks

## Connection to Other Experiments

This experiment is related to:
- **02-temporal-lora-gpt2**: Working with LLM and temporal epochs
- **06-subjective-time-critic**: Subjective time concept
- **05-recursive-time-full-suite**: Recursive temporal structures

## Literature

- Attractor theory in neural networks
- Recursive neural networks
- Reasoning in language models

## 📚 Publication

This work is part of a larger research program on Stability-First AI:

**Published Paper:**  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18148080.svg)](https://doi.org/10.5281/zenodo.18148080)

Sialedchyk, V. (2026). *Stability-First AI: Completed Experimental Studies and the Physics of Learning Time*. Zenodo. https://doi.org/10.5281/zenodo.18148080

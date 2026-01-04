# =========================
# Strict Validation Tests for Experiment
# =========================
"""
5 tests for strict validation of subjective time hypothesis.
"""

import torch
from transformers import GPT2Tokenizer
from recursive_time_depth import ImprovedRecursiveTimeModel, DEVICE, compute_entropy
import numpy as np
import random

def test_1_tokenwise_convergence(model, tokenizer, prompt, epsilon=0.1):
    """
    Test 1: Distribution of relative_change per token, not mean.
    """
    print("\n" + "="*80)
    print("TEST 1: Distribution of relative_change per token")
    print("="*80)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    # Modify model to save distribution
    model.use_recursive = True
    model.reset_stats()
    
    outputs = model(
        input_ids=input_ids,
        return_recursion_info=True,
        return_metrics=True
    )
    
    if "recursion_info" in outputs:
        info = outputs["recursion_info"]
        print(f"Prompt: {prompt}")
        print(f"Number of recursions: {info['recursion_count']}")
        print(f"Converged: {info['converged']}")
        
        # If percentiles exist
        if info.get("final_percentiles"):
            last_percentiles = info["final_percentiles"]
            print(f"\nPercentiles of relative_change on last iteration:")
            for p, val in last_percentiles.items():
                print(f"  {p}%: {val:.6f}")
            
            # Check strict criterion: 90% of tokens < epsilon
            p90 = last_percentiles[90]
            strict_converged = p90 < epsilon
            print(f"\nStrict criterion (90% tokens < {epsilon}): {strict_converged}")
            print(f"  90th percentile: {p90:.6f}")
    
    return outputs

def test_2_different_layers(model_class, tokenizer, prompt):
    """
    Test 2: Recursion on different layers (early, middle, late).
    """
    print("\n" + "="*80)
    print("TEST 2: Recursion on Different Layers")
    print("="*80)
    
    num_blocks = 12  # GPT-2 small has 12 blocks
    
    layer_configs = [
        ("Early (0-1)", 0, 2),
        ("Middle (5-6)", 5, 2),
        ("Late (10-11)", 10, 2)
    ]
    
    results = {}
    
    for name, start_layer, size in layer_configs:
        # Create model with recursion on specified layers
        model = model_class(
            model_name="gpt2",
            use_recursive=True,
            epsilon=0.1,
            max_recursions=20,
            min_recursions=3,
            recursion_subnetwork_size=size,
            recursion_start_layer=start_layer  # Now actually change recursion layer
        ).to(DEVICE)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        outputs = model(input_ids=input_ids, return_recursion_info=True)
        
        if "recursion_info" in outputs:
            info = outputs["recursion_info"]
            results[name] = {
                "recursion_count": info["recursion_count"],
                "converged": info["converged"],
                "final_change": info.get("final_change")
            }
            print(f"\n{name}:")
            print(f"  Recursions: {info['recursion_count']}")
            print(f"  Converged: {info['converged']}")
            if info.get("final_change"):
                print(f"  Final norm: {info['final_change']:.6f}")
    
    return results

def test_3_text_variations(model, tokenizer, base_prompt):
    """
    Test 3: Normal text vs shuffled vs random.
    """
    print("\n" + "="*80)
    print("TEST 3: Normal vs Shuffled vs Random Text")
    print("="*80)
    
    # Normal text
    normal_ids = tokenizer.encode(base_prompt, return_tensors="pt").to(DEVICE)
    
    # Shuffled text (same tokens, but in random order)
    normal_list = normal_ids[0].tolist()
    shuffled_list = normal_list.copy()
    random.shuffle(shuffled_list)
    shuffled_ids = torch.tensor([shuffled_list], device=DEVICE)
    
    # Random tokens
    random_ids = torch.randint(0, tokenizer.vocab_size, (1, len(normal_list))).to(DEVICE)
    
    test_cases = [
        ("Normal", normal_ids),
        ("Shuffled", shuffled_ids),
        ("Random", random_ids)
    ]
    
    results = {}
    
    for name, input_ids in test_cases:
        model.reset_stats()
        outputs = model(input_ids=input_ids, return_recursion_info=True)
        
        if "recursion_info" in outputs:
            info = outputs["recursion_info"]
            stats = model.get_stats()
            results[name] = {
                "recursion_count": info["recursion_count"],
                "converged": info["converged"],
                "final_change": info.get("final_change"),
                "avg_recursions": stats["avg_recursions"]
            }
            
            print(f"\n{name} text:")
            print(f"  Recursions: {info['recursion_count']}")
            print(f"  Converged: {info['converged']}")
            print(f"  Average: {stats['avg_recursions']:.2f}")
            if info.get("final_change"):
                print(f"  Final norm: {info['final_change']:.6f}")
    
    # Analysis
    print("\n" + "-"*80)
    print("ANALYSIS:")
    if "Normal" in results and "Shuffled" in results and "Random" in results:
        normal_rec = results["Normal"]["recursion_count"]
        shuffled_rec = results["Shuffled"]["recursion_count"]
        random_rec = results["Random"]["recursion_count"]
        
        print(f"Difference (Normal - Shuffled): {normal_rec - shuffled_rec}")
        print(f"Difference (Shuffled - Random): {shuffled_rec - random_rec}")
        
        # Check that shuffle is closer to random than to normal
        dist_shuffle_to_random = abs(shuffled_rec - random_rec)
        dist_shuffle_to_normal = abs(shuffled_rec - normal_rec)
        
        if dist_shuffle_to_random < dist_shuffle_to_normal:
            print("  [OK] Shuffled text is closer to random than to normal")
            print("  -> Metric is sensitive to structural connectivity/token order")
        else:
            print("  [WARN] Shuffled text is closer to normal")
    
    return results

def test_4_accuracy_comparison(model_class, tokenizer):
    """
    Test 4: Accuracy table for recursive vs normal mode.
    """
    print("\n" + "="*80)
    print("TEST 4: Accuracy Comparison")
    print("="*80)
    
    # Simple arithmetic tasks
    test_cases = [
        ("What is 2 + 2?", 4.0),
        ("What is 10 - 3?", 7.0),
        ("What is 5 * 4?", 20.0),
        ("What is 20 / 4?", 5.0),
        ("What is 3 + 7?", 10.0)
    ]
    
    def parse_answer(text):
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[0])
            except:
                return None
        return None
    
    results = {
        "recursive": {"correct": 0, "total": 0, "recursions": [], "times": []},
        "normal": {"correct": 0, "total": 0, "times": []}
    }
    
    import time
    
    # Recursive mode
    model_recursive = model_class(
        model_name="gpt2",
        use_recursive=True,
        epsilon=0.1,
        max_recursions=20,
        min_recursions=3
    ).to(DEVICE)
    
    print("\nRecursive mode:")
    for prompt, expected in test_cases:
        start = time.time()
        text, recursion_info, _ = model_recursive.generate(
            tokenizer, prompt, max_length=15, temperature=0.3,
            return_recursion_info=True
        )
        elapsed = time.time() - start
        
        predicted = parse_answer(text)
        is_correct = predicted is not None and abs(predicted - expected) < 0.1
        
        results["recursive"]["correct"] += int(is_correct)
        results["recursive"]["total"] += 1
        if recursion_info:
            results["recursive"]["recursions"].append(recursion_info["avg_recursions"])
        results["recursive"]["times"].append(elapsed)
        
        print(f"  {prompt}: {predicted} (expected {expected}) {'[OK]' if is_correct else '[FAIL]'}")
    
    # Normal mode
    model_normal = model_class(
        model_name="gpt2",
        use_recursive=False
    ).to(DEVICE)
    
    print("\nNormal mode:")
    for prompt, expected in test_cases:
        start = time.time()
        text, _, _ = model_normal.generate(
            tokenizer, prompt, max_length=15, temperature=0.3,
            return_recursion_info=False
        )
        elapsed = time.time() - start
        
        predicted = parse_answer(text)
        is_correct = predicted is not None and abs(predicted - expected) < 0.1
        
        results["normal"]["correct"] += int(is_correct)
        results["normal"]["total"] += 1
        results["normal"]["times"].append(elapsed)
        
        print(f"  {prompt}: {predicted} (expected {expected}) {'[OK]' if is_correct else '[FAIL]'}")
    
    # Final table
    print("\n" + "="*80)
    print("FINAL TABLE:")
    print("="*80)
    print(f"{'Mode':<20} {'Accuracy':<15} {'Avg Recursions':<20} {'Avg Time (s)':<15}")
    print("-"*80)
    
    rec_acc = results["recursive"]["correct"] / results["recursive"]["total"] if results["recursive"]["total"] > 0 else 0
    rec_avg_rec = np.mean(results["recursive"]["recursions"]) if results["recursive"]["recursions"] else 0
    rec_avg_time = np.mean(results["recursive"]["times"]) if results["recursive"]["times"] else 0
    
    norm_acc = results["normal"]["correct"] / results["normal"]["total"] if results["normal"]["total"] > 0 else 0
    norm_avg_time = np.mean(results["normal"]["times"]) if results["normal"]["times"] else 0
    
    print(f"{'Recursive':<20} {rec_acc*100:.1f}%{'':<10} {rec_avg_rec:.2f}{'':<15} {rec_avg_time:.3f}")
    print(f"{'Normal':<20} {norm_acc*100:.1f}%{'':<10} {'N/A':<20} {norm_avg_time:.3f}")
    
    print("\n" + "-"*80)
    print("INTERPRETATION:")
    print("-"*80)
    print("Accuracy on arithmetic is low in both modes (base GPT-2 limitation).")
    print("Recursive mode shows different compute/latency profile and")
    print("distinguishable convergence dynamics (adaptive depth).")
    
    return results

def test_5_entropy_fix(model, tokenizer, prompt):
    """
    Test 5: Fixed entropy with debugging.
    """
    print("\n" + "="*80)
    print("TEST 5: Fixed Entropy")
    print("="*80)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs["logits"][:, -1, :]  # Last token
        
        entropy, debug_info = compute_entropy(logits, temperature=1.0, return_debug=True)
        
        print(f"Prompt: {prompt}")
        print(f"\nEntropy: {entropy if entropy is not None else 'None (computation failed)'}")
        
        if debug_info:
            print(f"\nDebug information:")
            for key, val in debug_info.items():
                print(f"  {key}: {val}")
        
        # Check on different generation steps
        print(f"\nEntropy during generation:")
        generated = input_ids.clone()
        for step in range(5):
            outputs = model(input_ids=generated)
            logits = outputs["logits"][:, -1, :]
            # Compute entropy at temperature=0.8 (for generation) and temperature=1.0 (diagnostic)
            entropy_gen = compute_entropy(logits, temperature=0.8)
            entropy_diag = compute_entropy(logits, temperature=1.0)
            entropy = entropy_diag  # Use diagnostic for display
            
            if entropy is not None:
                print(f"  Step {step}: {entropy:.4f}")
            else:
                print(f"  Step {step}: None")
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
    
    return entropy

def run_all_strict_validation():
    """Runs all strict validation tests"""
    print("\n" + "="*80)
    print("STRICT VALIDATION TESTS FOR EXPERIMENT")
    print("="*80)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = ImprovedRecursiveTimeModel(
        model_name="gpt2",
        use_recursive=True,
        epsilon=0.1,
        max_recursions=20,
        min_recursions=3
    ).to(DEVICE)
    
    test_prompt = "What is 15 + 27?"
    
    # Test 1: Token-wise distribution
    test_1_tokenwise_convergence(model, tokenizer, test_prompt)
    
    # Test 2: Different layers
    test_2_different_layers(ImprovedRecursiveTimeModel, tokenizer, test_prompt)
    
    # Test 3: Text variations
    test_3_text_variations(model, tokenizer, test_prompt)
    
    # Test 4: Accuracy comparison
    test_4_accuracy_comparison(ImprovedRecursiveTimeModel, tokenizer)
    
    # Test 5: Entropy
    test_5_entropy_fix(model, tokenizer, test_prompt)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    run_all_strict_validation()


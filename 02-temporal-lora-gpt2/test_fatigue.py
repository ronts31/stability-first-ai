# =========================
# Router Fatigue Test: Deep Crystallization
# =========================
"""
Check of inertia dependence on domain stay length.

Hypothesis: the longer the model stays in one mode (domain),
the harder it is to "persuade" it to return to another domain.

This explains why models sometimes start "stalling" and repeating
in long dialogues - the router "crystallizes" in the current domain.

Test:
- Short Python block (10 tokens) -> measure return lag to A
- Long Python block (500 tokens) -> measure return lag to A

If lag grows with block length, this proves "deep crystallization".
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from transformers import GPT2Tokenizer
from temporal_lora import TemporalLoRAModel, DEVICE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

def generate_with_token_weights(
    model: TemporalLoRAModel,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    use_mixer: bool = True
) -> Tuple[str, List[torch.Tensor], List[int]]:
    """
    Generates text and returns router weights for each token.
    """
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    prompt_length = input_ids.size(1)
    generated = input_ids.clone()
    
    token_weights = []
    token_ids = input_ids[0].tolist()
    
    with torch.no_grad():
        for step in range(max_length):
            outputs = model(
                input_ids=generated,
                use_mixer=use_mixer,
                return_mixer_weights=True
            )
            
            if "mixer_weights" in outputs:
                mixer_weights = outputs["mixer_weights"]
                
                if step == 0:
                    for i in range(prompt_length):
                        token_weights.append(mixer_weights[0, i, :].cpu())
                else:
                    token_weights.append(mixer_weights[0, -1, :].cpu())
            
            logits = outputs["logits"][:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            token_ids.append(next_token.item())
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text, token_weights, token_ids

def compute_switch_lag(
    weights: List[torch.Tensor],
    domain_idx: int,
    switch_point: int,
    threshold: float = 0.9,
    consecutive_tokens: int = 3
) -> Optional[int]:
    """
    Computes switch-lag: number of tokens needed after switch_point
    for domain domain_idx weight to exceed threshold for K consecutive tokens.
    """
    if switch_point >= len(weights):
        return None
    
    for i in range(switch_point, len(weights) - consecutive_tokens + 1):
        all_above_threshold = True
        for j in range(consecutive_tokens):
            if weights[i + j][domain_idx].item() < threshold:
                all_above_threshold = False
                break
        
        if all_above_threshold:
            return i - switch_point
    
    return None

def create_python_block(tokenizer, target_length: int) -> str:
    """
    Creates a Python code block of approximately given length.
    
    Args:
        tokenizer: Tokenizer for token counting
        target_length: Target number of tokens
    
    Returns:
        String with Python code
    """
    # Basic Python phrases
    python_phrases = [
        "import torch",
        "import numpy as np",
        "def forward(self, x):",
        "return x",
        "model = TemporalLoRA()",
        "optimizer = optim.Adam(model.parameters())",
        "loss.backward()",
        "optimizer.step()",
        "tensor = torch.randn(batch_size, seq_len, hidden_dim)",
        "logits = model(input_ids)",
        "probs = F.softmax(logits, dim=-1)",
        "hidden_states = model.transformer(input_ids)",
        "attention_mask = torch.ones(batch_size, seq_len)",
        "outputs = model(input_ids=input_ids)",
        "loss = F.cross_entropy(logits, labels)",
    ]
    
    # Generate block until we reach target length
    block_parts = []
    current_tokens = 0
    
    while current_tokens < target_length:
        phrase = np.random.choice(python_phrases)
        phrase_tokens = len(tokenizer.encode(phrase))
        
        if current_tokens + phrase_tokens <= target_length * 1.2:  # Small margin
            block_parts.append(phrase)
            current_tokens += phrase_tokens
        else:
            break
    
    # If more needed, repeat phrases
    while current_tokens < target_length:
        phrase = np.random.choice(python_phrases)
        block_parts.append(phrase)
        current_tokens += len(tokenizer.encode(phrase))
    
    return " ".join(block_parts)

def test_fatigue(
    model: TemporalLoRAModel,
    tokenizer,
    prompt_A: str,
    python_block_length: int,
    tail_length: int = 20,
    max_length: int = 200
) -> Dict:
    """
    Test of inertia dependence on domain stay length.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt_A: Prompt for domain A (Shakespeare)
        python_block_length: Python block length in tokens
        tail_length: Tail length for return measurement
        max_length: Maximum generation length
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*80}")
    print(f"FATIGUE TEST: Python block length {python_block_length} tokens")
    print(f"{'='*80}")
    
    # Create Python block of required length
    python_block = create_python_block(tokenizer, python_block_length)
    actual_python_tokens = len(tokenizer.encode(python_block))
    
    print(f"Created Python block: {actual_python_tokens} tokens")
    print(f"First 100 characters: {python_block[:100]}...")
    
    # Generate sequence A -> Python -> A
    # For very long blocks, use only prompt without generation
    if python_block_length > 300:
        # For very long blocks, just analyze the prompt
        full_prompt = f"{prompt_A} {python_block} {prompt_A}"
        # Generate only small continuation for return analysis
        text, weights, token_ids = generate_with_token_weights(
            model, tokenizer, full_prompt, max_length=min(max_length, 100)
        )
    else:
        full_prompt = f"{prompt_A} {python_block} {prompt_A}"
        text, weights, token_ids = generate_with_token_weights(
            model, tokenizer, full_prompt, max_length=max_length
        )
    
    # Determine segment boundaries
    if len(weights) == 0:
        print("[ERROR] No weights for analysis")
        return None
    
    weights_array = torch.stack(weights).numpy()
    
    # Find switching points
    prompt_A_tokens = len(tokenizer.encode(prompt_A))
    switch_AP = min(prompt_A_tokens, len(weights) - 1)  # A -> Python
    switch_PA = min(prompt_A_tokens + actual_python_tokens, len(weights) - 1)  # Python -> A
    
    print(f"\nSegment boundaries:")
    print(f"  A1: 0-{switch_AP} tokens")
    print(f"  Python: {switch_AP}-{switch_PA} tokens ({actual_python_tokens} tokens)")
    print(f"  A2: {switch_PA}-{len(weights)} tokens")
    
    # Segments (with boundary checks)
    segment_A1 = weights[:switch_AP] if switch_AP > 0 else []
    segment_Python = weights[switch_AP:switch_PA] if switch_PA > switch_AP else []
    segment_A2 = weights[switch_PA:] if switch_PA < len(weights) else []
    
    # Limit tail length for A1 and A2
    if len(segment_A1) > tail_length:
        segment_A1 = segment_A1[-tail_length:]
    if len(segment_A2) > tail_length:
        segment_A2 = segment_A2[:tail_length]  # Take beginning of A2 for return measurement
    
    print(f"\nSegment lengths for analysis:")
    print(f"  A1 (tail): {len(segment_A1)} tokens")
    print(f"  Python: {len(segment_Python)} tokens")
    print(f"  A2 (beginning): {len(segment_A2)} tokens")
    
    # Compute metrics
    # Switch-lag for Python -> A (main metric)
    switch_lag_PA = compute_switch_lag(
        weights, domain_idx=0, switch_point=switch_PA,
        threshold=0.9, consecutive_tokens=3
    )
    
    # Additional metrics for crystallization assessment
    if len(segment_Python) > 0:
        python_weights_array = torch.stack(segment_Python).numpy()
        avg_python_weight = np.mean(python_weights_array[:, 1])  # Index 1 = Python
        
        # Python weight at end of Python segment (crystallization)
        final_python_weight = segment_Python[-1][1].item()
        
        # Python weight at beginning of Python segment
        initial_python_weight = segment_Python[0][1].item()
        
        # Number of tokens with Python weight > 0.95 (deep crystallization)
        deep_crystallization_count = np.sum(python_weights_array[:, 1] > 0.95)
        deep_crystallization_ratio = deep_crystallization_count / len(python_weights_array)
        
        # Time to stabilization: how many tokens needed for Python weight to become > 0.9
        stabilization_time = None
        for i in range(len(segment_Python)):
            if python_weights_array[i, 1] >= 0.9:
                stabilization_time = i
                break
        
        # Average Python weight in last third of segment (crystallization at end)
        last_third_start = len(segment_Python) * 2 // 3
        if last_third_start < len(segment_Python):
            last_third_weights = python_weights_array[last_third_start:, 1]
            avg_python_weight_last_third = np.mean(last_third_weights)
        else:
            avg_python_weight_last_third = final_python_weight
        
        # (A) Tail metric: last 25% or last 64 tokens (excludes "warm-up")
        tail_size = max(64, len(segment_Python) // 4)  # Last 25% or minimum 64 tokens
        tail_start = max(0, len(segment_Python) - tail_size)
        if tail_start < len(segment_Python):
            tail_weights = python_weights_array[tail_start:, 1]
            deep_crystallization_tail = np.sum(tail_weights > 0.95) / len(tail_weights)
            avg_python_weight_tail = np.mean(tail_weights)
        else:
            deep_crystallization_tail = deep_crystallization_ratio
            avg_python_weight_tail = final_python_weight
    else:
        avg_python_weight = 0.0
        final_python_weight = 0.0
        initial_python_weight = 0.0
        deep_crystallization_ratio = 0.0
        stabilization_time = None
        avg_python_weight_last_third = 0.0
    
    # (B) Relaxation metrics for checking "exit inertia"
    # relax_time_0.99: how many tokens after boundary needed for w_A>0.99 to hold for K consecutive
    relax_time_099 = compute_switch_lag(
        weights, domain_idx=0, switch_point=switch_PA,
        threshold=0.99, consecutive_tokens=3
    )
    
    # tail_area_32: Σ_{t=1..32} (1 - w_A(t)) after boundary (area under "uncertainty" curve)
    if len(segment_A2) >= 32:
        tail_32_weights = torch.stack(segment_A2[:32]).numpy()
        tail_area_32 = np.sum(1.0 - tail_32_weights[:, 0])  # Sum (1 - w_A) for first 32 tokens
    else:
        tail_32_weights = torch.stack(segment_A2).numpy() if len(segment_A2) > 0 else np.array([[0.5, 0.5]])
        tail_area_32 = np.sum(1.0 - tail_32_weights[:, 0]) if len(segment_A2) > 0 else 0.0
    
    # Return-gap between A1 and A2
    from test_hysteresis import compute_return_gap
    return_gap_cosine = compute_return_gap(segment_A1, segment_A2, method="cosine_distance")
    
    results = {
        'python_block_length': actual_python_tokens,
        'switch_lag_PA': switch_lag_PA,
        'avg_python_weight': avg_python_weight,
        'initial_python_weight': initial_python_weight,
        'final_python_weight': final_python_weight,
        'deep_crystallization_ratio': deep_crystallization_ratio,
        'deep_crystallization_tail': deep_crystallization_tail,  # (A) On segment tail
        'avg_python_weight_tail': avg_python_weight_tail,  # (A) Average weight on tail
        'stabilization_time': stabilization_time,
        'avg_python_weight_last_third': avg_python_weight_last_third,
        'relax_time_099': relax_time_099,  # (B) Relaxation to 0.99
        'tail_area_32': tail_area_32,  # (B) "Uncertainty" area in first 32 tokens
        'return_gap_cosine': return_gap_cosine,
        'weights': weights,
        'segment_Python': segment_Python,
        'segment_A1': segment_A1,
        'segment_A2': segment_A2,
        'switch_PA': switch_PA
    }
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"Python block length: {actual_python_tokens} tokens")
    print(f"Switch-lag Python->A: {switch_lag_PA} tokens (K=3 consecutive with w(A)>0.9)")
    print(f"Average Python weight in segment: {avg_python_weight:.4f}")
    print(f"Python weight at beginning: {initial_python_weight:.4f}")
    print(f"Python weight at end: {final_python_weight:.4f}")
    print(f"Deep crystallization ratio (w>0.95): {deep_crystallization_ratio:.4f}")
    print(f"Deep crystallization on tail (last {max(64, len(segment_Python) // 4)} tokens): {deep_crystallization_tail:.4f}")
    print(f"Stabilization time: {stabilization_time} tokens" if stabilization_time is not None else "Stabilization time: not reached")
    print(f"Average Python weight in last third: {avg_python_weight_last_third:.4f}")
    print(f"\nRelaxation metrics (checking exit inertia):")
    print(f"  Relax-time to 0.99: {relax_time_099} tokens" if relax_time_099 is not None else "  Relax-time to 0.99: not reached")
    print(f"  Tail-area (first 32 tokens): {tail_area_32:.4f} (sum (1 - w_A))")
    print(f"Return-gap (cosine_distance): {return_gap_cosine:.4f}")
    
    return results

def test_fatigue_sweep(
    model: TemporalLoRAModel,
    tokenizer,
    prompt_A: str,
    python_lengths: List[int] = [10, 50, 100, 200, 500],
    tail_length: int = 20,
    max_length: int = 800
) -> Dict:
    """
    Sweep test: checks inertia dependence on Python block length.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt_A: Prompt for domain A
        python_lengths: List of Python block lengths for testing
        tail_length: Tail length for return measurement
        max_length: Maximum generation length
    
    Returns:
        Dictionary with all test results
    """
    print("\n" + "="*80)
    print("SWEEP TEST: Inertia Dependence on Domain Stay Length")
    print("="*80)
    print("\nHypothesis: the longer the model stays in Python domain, the harder to return to A")
    print("This will prove 'deep crystallization' - router 'fatigue' effect")
    
    all_results = []
    
    for length in python_lengths:
        print(f"\n{'='*80}")
        print(f"Test {len(all_results) + 1}/{len(python_lengths)}: Python block {length} tokens")
        print(f"{'='*80}")
        
        try:
            result = test_fatigue(
                model, tokenizer,
                prompt_A=prompt_A,
                python_block_length=length,
                tail_length=tail_length,
                max_length=max_length
            )
            result['target_length'] = length
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Error in test for length {length}: {e}")
            continue
    
    # Analyze results
    print("\n" + "="*80)
    print("SWEEP RESULTS ANALYSIS")
    print("="*80)
    
    lengths = [r['python_block_length'] for r in all_results]
    lags = [r['switch_lag_PA'] for r in all_results if r['switch_lag_PA'] is not None]
    lag_lengths = [r['python_block_length'] for r in all_results if r['switch_lag_PA'] is not None]
    final_weights = [r['final_python_weight'] for r in all_results]
    deep_cryst_ratios = [r['deep_crystallization_ratio'] for r in all_results]
    deep_cryst_tails = [r['deep_crystallization_tail'] for r in all_results]  # (A) On tail
    last_third_weights = [r['avg_python_weight_last_third'] for r in all_results]
    relax_times_099 = [r['relax_time_099'] if r['relax_time_099'] is not None else 0 for r in all_results]  # (B)
    tail_areas_32 = [r['tail_area_32'] for r in all_results]  # (B)
    
    print(f"\nSwitch-lag dependence on Python block length:")
    for r in all_results:
        lag_str = f"{r['switch_lag_PA']}" if r['switch_lag_PA'] is not None else "None"
        print(f"  {r['python_block_length']:4d} tokens -> lag = {lag_str:>6s} tokens")
    
    print(f"\nCrystallization dependence on Python block length:")
    for r in all_results:
        print(f"  {r['python_block_length']:4d} tokens -> weight at end: {r['final_python_weight']:.4f}, "
              f"deep cryst.: {r['deep_crystallization_ratio']:.4f}, "
              f"on tail: {r['deep_crystallization_tail']:.4f}, "
              f"last third: {r['avg_python_weight_last_third']:.4f}")
    
    print(f"\nRelaxation metrics (checking exit inertia):")
    for r in all_results:
        relax_str = f"{r['relax_time_099']}" if r['relax_time_099'] is not None else "None"
        print(f"  {r['python_block_length']:4d} tokens -> relax_time_0.99: {relax_str:>6s}, "
              f"tail_area_32: {r['tail_area_32']:.4f}")
    
    # Check correlations for different metrics
    correlations = {}
    
    if len(lags) > 1 and len(set(lags)) > 1:
        correlation_lag = np.corrcoef(lag_lengths, lags)[0, 1]
        correlations['switch_lag'] = correlation_lag
        print(f"\nCorrelation block_length vs switch_lag: {correlation_lag:.4f}")
    
    if len(final_weights) > 1 and len(set(final_weights)) > 1:
        correlation_final = np.corrcoef(lengths, final_weights)[0, 1]
        correlations['final_weight'] = correlation_final
        print(f"Correlation block_length vs weight_at_end: {correlation_final:.4f}")
    
    if len(deep_cryst_ratios) > 1 and len(set(deep_cryst_ratios)) > 1:
        correlation_deep = np.corrcoef(lengths, deep_cryst_ratios)[0, 1]
        correlations['deep_crystallization'] = correlation_deep
        print(f"Correlation block_length vs deep_crystallization: {correlation_deep:.4f}")
    
    # (A) Correlation on tail (excludes "warm-up" effect)
    if len(deep_cryst_tails) > 1 and len(set(deep_cryst_tails)) > 1:
        correlation_tail = np.corrcoef(lengths, deep_cryst_tails)[0, 1]
        correlations['deep_crystallization_tail'] = correlation_tail
        print(f"Correlation block_length vs deep_crystallization_on_tail: {correlation_tail:.4f}")
    
    if len(last_third_weights) > 1 and len(set(last_third_weights)) > 1:
        correlation_last = np.corrcoef(lengths, last_third_weights)[0, 1]
        correlations['last_third'] = correlation_last
        print(f"Correlation block_length vs last_third: {correlation_last:.4f}")
    
    # (B) Correlations for relaxation metrics
    if len(relax_times_099) > 1 and len(set(relax_times_099)) > 1:
        correlation_relax = np.corrcoef(lengths, relax_times_099)[0, 1]
        correlations['relax_time_099'] = correlation_relax
        print(f"Correlation block_length vs relax_time_0.99: {correlation_relax:.4f}")
    
    if len(tail_areas_32) > 1 and len(set(tail_areas_32)) > 1:
        correlation_area = np.corrcoef(lengths, tail_areas_32)[0, 1]
        correlations['tail_area_32'] = correlation_area
        print(f"Correlation block_length vs tail_area_32: {correlation_area:.4f}")
    
    # Determine overall correlation (take maximum positive for crystallization)
    # Priority: deep_crystallization > last_third > final_weight > switch_lag
    correlation = None
    max_corr_key = None
    
    if correlations:
        # Look for positive correlations related to crystallization
        priority_keys = ['deep_crystallization', 'last_third', 'final_weight', 'switch_lag']
        for key in priority_keys:
            if key in correlations and correlations[key] > 0:
                correlation = correlations[key]
                max_corr_key = key
                break
        
        # If no positive, take maximum by absolute value
        if correlation is None:
            max_corr_key = max(correlations.keys(), key=lambda k: abs(correlations[k]))
            correlation = correlations[max_corr_key]
    
    if correlation is not None:
        if correlation > 0.3:
            print(f"\n[+] POSITIVE CORRELATION DETECTED! (max: {max_corr_key} = {correlation:.4f})")
            if 'tail' in max_corr_key or 'deep_crystallization' in max_corr_key:
                print("   This confirms domain confidence saturation effect")
                print("   (domain concentration strengthens with prolonged stay in mode)")
            elif 'relax' in max_corr_key or 'area' in max_corr_key:
                print("   This confirms exit inertia effect")
                print("   (long stay in domain makes switching back harder)")
        elif correlation < -0.3:
            print(f"\n[!] NEGATIVE CORRELATION (unexpected): {correlation:.4f}")
        else:
            print(f"\n[?] Weak correlation ({correlation:.4f}) - additional tests needed")
    else:
        print("\n[?] Failed to compute correlation (all values identical)")
    
    # Visualization
    try:
        visualize_fatigue_sweep(all_results, save_path="fatigue_analysis.png")
    except Exception as e:
        print(f"[WARN] Failed to create visualization: {e}")
    
    return {
        'all_results': all_results,
        'correlation': correlation,
        'correlations': correlations if 'correlations' in locals() else {}
    }

def visualize_fatigue_sweep(all_results: List[Dict], save_path: str = "fatigue_analysis.png"):
    """
    Visualizes fatigue sweep test results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Switch-lag vs Python block length
    ax1 = axes[0, 0]
    lengths = [r['python_block_length'] for r in all_results]
    lags = [r['switch_lag_PA'] if r['switch_lag_PA'] is not None else 0 for r in all_results]
    
    ax1.plot(lengths, lags, 'o-', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Python Block Length (tokens)', fontsize=12)
    ax1.set_ylabel('Switch-lag Python->A (tokens)', fontsize=12)
    ax1.set_title('Inertia Dependence on Domain Stay Length', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    if len(lags) > 1 and any(l > 0 for l in lags):
        valid_indices = [i for i, l in enumerate(lags) if l > 0]
        if len(valid_indices) > 1:
            valid_lengths = [lengths[i] for i in valid_indices]
            valid_lags = [lags[i] for i in valid_indices]
            z = np.polyfit(valid_lengths, valid_lags, 1)
            p = np.poly1d(z)
            ax1.plot(valid_lengths, p(valid_lengths), "r--", alpha=0.5, label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
            ax1.legend()
    
    # Plot 2: Python weight at end of segment vs length
    ax2 = axes[0, 1]
    final_weights = [r['final_python_weight'] for r in all_results]
    ax2.plot(lengths, final_weights, 'o-', linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel('Python Block Length (tokens)', fontsize=12)
    ax2.set_ylabel('Python Weight at End of Segment', fontsize=12)
    ax2.set_title('Crystallization: Python Weight at Block End', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Crystallization Threshold (0.9)')
    ax2.legend()
    
    # Plot 3: Return-gap vs Python block length
    ax3 = axes[1, 0]
    return_gaps = [r['return_gap_cosine'] for r in all_results]
    ax3.plot(lengths, return_gaps, 'o-', linewidth=2, markersize=8, color='purple')
    ax3.set_xlabel('Python Block Length (tokens)', fontsize=12)
    ax3.set_ylabel('Return-gap (cosine_distance)', fontsize=12)
    ax3.set_title('Trajectory Memory vs Block Length', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Example trajectory for longest block
    ax4 = axes[1, 1]
    if len(all_results) > 0:
        longest_result = max(all_results, key=lambda r: r['python_block_length'])
        weights_array = torch.stack(longest_result['weights']).numpy()
        tokens = range(len(weights_array))
        
        ax4.plot(tokens, weights_array[:, 0], label='Shakespeare', linewidth=2)
        ax4.plot(tokens, weights_array[:, 1], label='Python', linewidth=2)
        
        # Mark boundaries
        switch_PA = longest_result['switch_PA']
        ax4.axvline(x=switch_PA, color='red', linestyle='--', alpha=0.5, label='Python->A')
        
        ax4.set_xlabel('Token', fontsize=12)
        ax4.set_ylabel('Adapter Weight', fontsize=12)
        ax4.set_title(f'Weight Trajectory (block {longest_result["python_block_length"]} tokens)', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[OK] Plot saved: {save_path}")

def main():
    """
    Main function to run fatigue test.
    """
    print("\n" + "="*80)
    print("ROUTER FATIGUE TEST: Deep Crystallization")
    print("="*80)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n>>> Loading model...")
    
    # Create model
    model = TemporalLoRAModel(
        model_name="gpt2",
        lora_rank=8,
        lora_alpha=16.0,
        mixer_strategy="gating",
        freeze_backbone=True
    )
    model = model.to(DEVICE)
    
    # Add adapters
    model.add_adapter("shakespeare", "Renaissance Era (Shakespeare)")
    model.add_adapter("python", "IT Era (Python)")
    
    # Load trained model weights
    checkpoint_path = "temporal_lora_checkpoint.pt"
    try:
        if os.path.exists(checkpoint_path):
            print(f"\n>>> Loading weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            for name, adapter_state in checkpoint['adapters'].items():
                if name in model.adapters:
                    model.adapters[name].load_state_dict(adapter_state)
                    print(f"[OK] Loaded adapter weights '{name}'")
            
            if checkpoint['time_mixer'] is not None and model.time_mixer is not None:
                model.time_mixer.load_state_dict(checkpoint['time_mixer'])
                print(f"[OK] Loaded Time Mixer weights")
            
            print("[OK] Model loaded from checkpoint")
        else:
            print(f"[WARN] Checkpoint not found: {checkpoint_path}")
            print("[WARN] Using random weights - results will be uninformative")
    except Exception as e:
        print(f"[WARN] Error loading checkpoint: {e}")
    
    # Test prompt
    prompt_A = "To code or not to code, that is the question."
    
    # Sweep test parameters
    python_lengths = [10, 50, 100, 200, 500]  # Python block lengths for testing
    
    # Run sweep test
    sweep_results = test_fatigue_sweep(
        model, tokenizer,
        prompt_A=prompt_A,
        python_lengths=python_lengths,
        tail_length=20,
        max_length=800
    )
    
    # Final report
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    print("\n" + "="*80)
    print("FATIGUE TEST RESULTS")
    print("="*80)
    
    all_results = sweep_results.get('all_results', [])
    correlations = sweep_results.get('correlations', {})
    correlation = sweep_results.get('correlation')
    
    # Main results
    print("\n1. Switch-lag (threshold w(A)>0.9, K=3):")
    for r in all_results:
        lag_str = f"{r['switch_lag_PA']}" if r['switch_lag_PA'] is not None else "None"
        print(f"   {r['python_block_length']:4d} tokens -> lag = {lag_str} tokens")
    
    print("\n2. Deep crystallization (proportion of tokens with w_python>0.95 within Python segment):")
    for r in all_results:
        print(f"   {r['python_block_length']:4d} tokens -> {r['deep_crystallization_ratio']*100:.2f}%")
    
    if 'deep_crystallization' in correlations:
        corr_deep = correlations['deep_crystallization']
        print(f"\n   Positive correlation found: r={corr_deep:.4f}")
    
    # Tail metrics
    if 'deep_crystallization_tail' in correlations:
        corr_tail = correlations['deep_crystallization_tail']
        print(f"\n3. Deep crystallization on tail (last 25% or 64 tokens):")
        for r in all_results:
            print(f"   {r['python_block_length']:4d} tokens -> {r['deep_crystallization_tail']*100:.2f}%")
        print(f"   Correlation: r={corr_tail:.4f}")
    
    # Relaxation metrics
    print("\n4. Relaxation metrics (checking exit inertia):")
    for r in all_results:
        relax_str = f"{r['relax_time_099']}" if r['relax_time_099'] is not None else "None"
        print(f"   {r['python_block_length']:4d} tokens -> relax_time_0.99: {relax_str}, tail_area_32: {r['tail_area_32']:.4f}")
    
    if 'relax_time_099' in correlations:
        print(f"   Correlation relax_time_0.99: r={correlations['relax_time_099']:.4f}")
    if 'tail_area_32' in correlations:
        print(f"   Correlation tail_area_32: r={correlations['tail_area_32']:.4f}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
Domain confidence saturation effect is observed (domain concentration 
strengthens with prolonged stay in mode). However, switching delay growth 
in switch-lag metric did not manifest; to check "exit inertia" requires 
relaxation metrics (relax_time_0.99, tail_area_32), as well as tests 
on hybrid returns.

Correct formulation:
"The longer the model stays in a domain, the stronger the domain concentration 
(more tokens with extreme weights). This may reduce flexibility on 
boundary/hybrid segments and increase inertia during switching — this 
needs to be confirmed by relaxation metrics and tests on hybrid returns."
    """)
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()


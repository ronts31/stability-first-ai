# =========================
# Router Hysteresis Test: Time Crystallization
# =========================
"""
Router hysteresis (crystallization as "epoch inertia")

Test sequences:
- A → B → A
- A → (A/B mix) → A
- With tail length control

Metrics:
- switch-lag: number of tokens needed for domain weight > 0.9 after switch
- return-gap: how much second "A" differs from first "A" by weight trajectory w(t)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
from transformers import GPT2Tokenizer
from temporal_lora import TemporalLoRAModel, DEVICE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

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
    
    Returns:
        text: Generated text
        token_weights: List of weights for each token [num_tokens, num_adapters]
        token_ids: List of token IDs
    """
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    prompt_length = input_ids.size(1)
    generated = input_ids.clone()
    
    token_weights = []  # Weights for each token
    token_ids = input_ids[0].tolist()
    
    with torch.no_grad():
        for step in range(max_length):
            outputs = model(
                input_ids=generated,
                use_mixer=use_mixer,
                return_mixer_weights=True
            )
            
            # Get weights for all tokens in current sequence
            if "mixer_weights" in outputs:
                mixer_weights = outputs["mixer_weights"]  # [1, seq_len, num_adapters]
                
                # For first step: log prompt weights
                if step == 0:
                    # Log weights for each prompt token
                    for i in range(prompt_length):
                        token_weights.append(mixer_weights[0, i, :].cpu())
                else:
                    # Log weight only for last (new) token
                    token_weights.append(mixer_weights[0, -1, :].cpu())
            
            # Generate next token
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
    
    Formal definition:
    - Domain is considered active when w(domain) > threshold
    - Switch-lag = number of tokens after segment boundary until condition
      is met for K consecutive tokens (excludes single peaks/spikes)
    
    Args:
        weights: List of weights for each token
        domain_idx: Domain index (0 for A, 1 for B)
        switch_point: Token index where domain switch occurred
        threshold: Threshold for "switching" determination (default 0.9)
        consecutive_tokens: Number of consecutive tokens for confirmation (K=3)
    
    Returns:
        Number of tokens until stable switching, or None if not reached
    """
    if switch_point >= len(weights):
        return None
    
    # Find first position where sequence of K consecutive tokens with weight >= threshold begins
    for i in range(switch_point, len(weights) - consecutive_tokens + 1):
        # Check that next K tokens all have weight >= threshold
        all_above_threshold = True
        for j in range(consecutive_tokens):
            if weights[i + j][domain_idx].item() < threshold:
                all_above_threshold = False
                break
        
        if all_above_threshold:
            return i - switch_point
    
    return None  # Stable switching not reached

def compute_return_gap(
    weights_A1: List[torch.Tensor],
    weights_A2: List[torch.Tensor],
    method: str = "cosine"
) -> float:
    """
    Computes return-gap: how much second "A" differs from first "A".
    
    Args:
        weights_A1: Weights for first segment A
        weights_A2: Weights for second segment A
        method: Comparison method ("cosine", "dtw", "euclidean")
    
    Returns:
        Measure of difference between trajectories
    """
    if len(weights_A1) == 0 or len(weights_A2) == 0:
        return float('inf')
    
    # Convert to numpy arrays
    w1 = torch.stack(weights_A1).numpy()  # [len_A1, num_adapters]
    w2 = torch.stack(weights_A2).numpy()  # [len_A2, num_adapters]
    
    if method == "cosine" or method == "cosine_distance":
        # Cosine distance = 1 - cosine_similarity
        # This is a difference measure: 0.0 = identical, 1.0 = maximally different
        # Do NOT confuse with cosine_similarity (where 1.0 = identical, 0.0 = different)
        # 
        # Normalize sequences to same length
        min_len = min(len(w1), len(w2))
        w1_norm = w1[:min_len]
        w2_norm = w2[:min_len]
        
        distances = []
        for i in range(min_len):
            # scipy.spatial.distance.cosine returns cosine distance (1 - cos_sim)
            # 0.0 = vectors identical, 1.0 = opposite
            dist = cosine(w1_norm[i], w2_norm[i])
            distances.append(dist)
        
        return np.mean(distances)
    
    elif method == "euclidean":
        # Average euclidean distance
        min_len = min(len(w1), len(w2))
        w1_norm = w1[:min_len]
        w2_norm = w2[:min_len]
        
        distances = []
        for i in range(min_len):
            dist = euclidean(w1_norm[i], w2_norm[i])
            distances.append(dist)
        
        return np.mean(distances)
    
    elif method == "dtw":
        # Simple DTW (Dynamic Time Warping) implementation
        # Use euclidean distance between weight vectors
        def dtw_distance(seq1, seq2):
            n, m = len(seq1), len(seq2)
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = euclidean(seq1[i-1], seq2[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],
                        dtw_matrix[i, j-1],
                        dtw_matrix[i-1, j-1]
                    )
            
            return dtw_matrix[n, m]
        
        return dtw_distance(w1, w2)
    
    else:
        raise ValueError(f"Unknown method: {method}")

def test_sequence_aba(
    model: TemporalLoRAModel,
    tokenizer,
    prompt_A: str,
    prompt_B: str,
    tail_length: int = 10,
    max_length: int = 150
) -> Dict:
    """
    Test sequence A → B → A.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt_A: Prompt for domain A (Shakespeare)
        prompt_B: Prompt for domain B (Python)
        tail_length: Tail length for each segment
        max_length: Maximum generation length
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*80}")
    print(f"TEST: A -> B -> A")
    print(f"{'='*80}")
    
    # Generate sequence
    full_prompt = f"{prompt_A} {prompt_B} {prompt_A}"
    print(f"Full prompt: '{full_prompt}'")
    
    text, weights, token_ids = generate_with_token_weights(
        model, tokenizer, full_prompt, max_length=max_length
    )
    
    # Determine segment boundaries
    # Simple heuristic: find points where domain weight changes
    weights_array = torch.stack(weights).numpy()  # [num_tokens, num_adapters]
    
    # Find switching points (when domain B weight becomes > 0.5)
    switch_points = []
    for i in range(1, len(weights)):
        if weights_array[i-1, 1] < 0.5 and weights_array[i, 1] >= 0.5:
            switch_points.append(i)
        elif weights_array[i-1, 1] >= 0.5 and weights_array[i, 1] < 0.5:
            switch_points.append(i)
    
    print(f"\nFound switching points: {switch_points}")
    
    # If not found automatically, use heuristic based on prompt length
    if len(switch_points) < 2:
        prompt_A_tokens = len(tokenizer.encode(prompt_A))
        prompt_B_tokens = len(tokenizer.encode(prompt_B))
        switch_AB = prompt_A_tokens
        switch_BA = prompt_A_tokens + prompt_B_tokens
        switch_points = [switch_AB, switch_BA]
        print(f"Using heuristic: switch_AB={switch_AB}, switch_BA={switch_BA}")
    
    if len(switch_points) >= 2:
        switch_AB = switch_points[0]
        switch_BA = switch_points[1]
    else:
        # Fallback: divide into three equal parts
        switch_AB = len(weights) // 3
        switch_BA = 2 * len(weights) // 3
        print(f"Fallback: switch_AB={switch_AB}, switch_BA={switch_BA}")
    
    # Segments
    segment_A1 = weights[:switch_AB]  # First A
    segment_B = weights[switch_AB:switch_BA]  # B
    segment_A2 = weights[switch_BA:]  # Second A
    
    # Limit tail length
    if len(segment_A1) > tail_length:
        segment_A1 = segment_A1[-tail_length:]
    if len(segment_A2) > tail_length:
        segment_A2 = segment_A2[-tail_length:]
    
    print(f"\nSegment lengths:")
    print(f"  A1 (tail): {len(segment_A1)} tokens")
    print(f"  B: {len(segment_B)} tokens")
    print(f"  A2 (tail): {len(segment_A2)} tokens")
    
    # Compute metrics
    # 1. Switch-lag for A → B (formally: K=3 consecutive tokens with w(B) > 0.9)
    switch_lag_AB = compute_switch_lag(weights, domain_idx=1, switch_point=switch_AB,
                                       threshold=0.9, consecutive_tokens=3)
    
    # 2. Switch-lag for B → A (formally: K=3 consecutive tokens with w(A) > 0.9)
    switch_lag_BA = compute_switch_lag(weights, domain_idx=0, switch_point=switch_BA,
                                        threshold=0.9, consecutive_tokens=3)
    
    # 3. Return-gap between A1 and A2
    return_gap_cosine = compute_return_gap(segment_A1, segment_A2, method="cosine_distance")
    return_gap_euclidean = compute_return_gap(segment_A1, segment_A2, method="euclidean")
    return_gap_dtw = compute_return_gap(segment_A1, segment_A2, method="dtw")
    
    # Additional metrics: asymmetry of switching
    switching_asymmetry = None
    if switch_lag_AB is not None and switch_lag_BA is not None:
        switching_asymmetry = abs(switch_lag_AB - switch_lag_BA)
    
    # Average weights in each segment
    segment_A1_weights = torch.stack(segment_A1).numpy()
    segment_B_weights = torch.stack(segment_B).numpy()
    segment_A2_weights = torch.stack(segment_A2).numpy()
    
    avg_weight_A1 = np.mean(segment_A1_weights, axis=0)
    avg_weight_B = np.mean(segment_B_weights, axis=0)
    avg_weight_A2 = np.mean(segment_A2_weights, axis=0)
    
    results = {
        'sequence': 'A->B->A',
        'text': text,
        'weights': weights,  # Keep as tensors for visualization
        'weights_json': [w.tolist() for w in weights],  # Convert to list for JSON
        'token_ids': token_ids,
        'switch_AB': switch_AB,
        'switch_BA': switch_BA,
        'segment_A1': segment_A1,  # Keep for visualization
        'segment_B': segment_B,
        'segment_A2': segment_A2,  # Keep for visualization
        'switch_lag_AB': switch_lag_AB,
        'switch_lag_BA': switch_lag_BA,
        'switching_asymmetry': switching_asymmetry,
        'return_gap_cosine_distance': float(return_gap_cosine),
        'return_gap_euclidean': float(return_gap_euclidean),
        'return_gap_dtw': float(return_gap_dtw),
        'avg_weight_A1': avg_weight_A1.tolist(),
        'avg_weight_B': avg_weight_B.tolist(),
        'avg_weight_A2': avg_weight_A2.tolist(),
        'segment_lengths': {
            'A1': len(segment_A1),
            'B': len(segment_B),
            'A2': len(segment_A2)
        }
    }
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"Switch-lag A->B: {switch_lag_AB} tokens (K=3 consecutive tokens with w>0.9)")
    print(f"Switch-lag B->A: {switch_lag_BA} tokens (K=3 consecutive tokens with w>0.9)")
    print(f"Return-gap (cosine_distance): {return_gap_cosine:.4f} (0.0=identical, 1.0=maximally different)")
    print(f"Return-gap (euclidean): {return_gap_euclidean:.4f}")
    print(f"Return-gap (DTW): {return_gap_dtw:.4f}")
    
    return results

def test_sequence_amixba(
    model: TemporalLoRAModel,
    tokenizer,
    prompt_A: str,
    prompt_mix: str,
    tail_length: int = 10,
    max_length: int = 150
) -> Dict:
    """
    Test sequence A → (A/B mix) → A.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt_A: Prompt for domain A (Shakespeare)
        prompt_mix: Prompt for mixed domain (A/B mix)
        tail_length: Tail length for each segment
        max_length: Maximum generation length
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*80}")
    print(f"TEST: A -> (A/B mix) -> A")
    print(f"{'='*80}")
    
    # Generate sequence
    full_prompt = f"{prompt_A} {prompt_mix} {prompt_A}"
    print(f"Full prompt: '{full_prompt}'")
    
    text, weights, token_ids = generate_with_token_weights(
        model, tokenizer, full_prompt, max_length=max_length
    )
    
    # Determine segment boundaries
    weights_array = torch.stack(weights).numpy()
    
    # Find switching points
    # For mix segment, find where weights become mixed (0.3 < weight < 0.7)
    switch_points = []
    in_mix = False
    for i in range(1, len(weights)):
        w_A = weights_array[i, 0]
        w_B = weights_array[i, 1]
        is_mix = 0.3 < w_A < 0.7 and 0.3 < w_B < 0.7
        
        if not in_mix and is_mix:
            switch_points.append(i)
            in_mix = True
        elif in_mix and not is_mix:
            switch_points.append(i)
            in_mix = False
    
    print(f"\nFound switching points: {switch_points}")
    
    # If not found automatically, use heuristic
    if len(switch_points) < 2:
        prompt_A_tokens = len(tokenizer.encode(prompt_A))
        prompt_mix_tokens = len(tokenizer.encode(prompt_mix))
        switch_Amix = prompt_A_tokens
        switch_mixA = prompt_A_tokens + prompt_mix_tokens
        switch_points = [switch_Amix, switch_mixA]
        print(f"Using heuristic: switch_Amix={switch_Amix}, switch_mixA={switch_mixA}")
    
    if len(switch_points) >= 2:
        switch_Amix = switch_points[0]
        switch_mixA = switch_points[1]
    else:
        switch_Amix = len(weights) // 3
        switch_mixA = 2 * len(weights) // 3
        print(f"Fallback: switch_Amix={switch_Amix}, switch_mixA={switch_mixA}")
    
    # Segments
    segment_A1 = weights[:switch_Amix]
    segment_mix = weights[switch_Amix:switch_mixA]
    segment_A2 = weights[switch_mixA:]
    
    # Limit tail length
    if len(segment_A1) > tail_length:
        segment_A1 = segment_A1[-tail_length:]
    if len(segment_A2) > tail_length:
        segment_A2 = segment_A2[-tail_length:]
    
    print(f"\nSegment lengths:")
    print(f"  A1 (tail): {len(segment_A1)} tokens")
    print(f"  Mix: {len(segment_mix)} tokens")
    print(f"  A2 (tail): {len(segment_A2)} tokens")
    
    # Compute metrics
    # For mix segment, switch-lag doesn't make sense (no clear switching)
    # But we can measure how quickly it returns to A (K=3 consecutive tokens with w(A) > 0.9)
    switch_lag_mixA = compute_switch_lag(weights, domain_idx=0, switch_point=switch_mixA,
                                         threshold=0.9, consecutive_tokens=3)
    
    # Return-gap between A1 and A2
    return_gap_cosine = compute_return_gap(segment_A1, segment_A2, method="cosine_distance")
    return_gap_euclidean = compute_return_gap(segment_A1, segment_A2, method="euclidean")
    return_gap_dtw = compute_return_gap(segment_A1, segment_A2, method="dtw")
    
    # Additionally: measure of "mixedness" of mix segment
    mix_weights_array = torch.stack(segment_mix).numpy()
    mix_entropy = []
    for w in mix_weights_array:
        # Entropy of weight distribution (higher = more mixed)
        entropy = -np.sum(w * np.log(w + 1e-10))
        mix_entropy.append(entropy)
    avg_mix_entropy = np.mean(mix_entropy)
    
    # Average weights in each segment
    segment_A1_weights = torch.stack(segment_A1).numpy()
    segment_mix_weights = torch.stack(segment_mix).numpy()
    segment_A2_weights = torch.stack(segment_A2).numpy()
    
    avg_weight_A1 = np.mean(segment_A1_weights, axis=0)
    avg_weight_mix = np.mean(segment_mix_weights, axis=0)
    avg_weight_A2 = np.mean(segment_A2_weights, axis=0)
    
    # Mix segment balance (how balanced the mix is)
    mix_balance = np.min(avg_weight_mix) / np.max(avg_weight_mix) if np.max(avg_weight_mix) > 0 else 0.0
    
    results = {
        'sequence': 'A->Mix->A',
        'text': text,
        'weights': weights,  # Keep as tensors for visualization
        'weights_json': [w.tolist() for w in weights],  # Convert to list for JSON
        'token_ids': token_ids,
        'switch_Amix': switch_Amix,
        'switch_mixA': switch_mixA,
        'segment_A1': segment_A1,  # Keep for visualization
        'segment_mix': segment_mix,
        'segment_A2': segment_A2,  # Keep for visualization
        'switch_lag_mixA': switch_lag_mixA,
        'return_gap_cosine_distance': float(return_gap_cosine),
        'return_gap_euclidean': float(return_gap_euclidean),
        'return_gap_dtw': float(return_gap_dtw),
        'avg_mix_entropy': float(avg_mix_entropy),
        'mix_balance': float(mix_balance),
        'avg_weight_A1': avg_weight_A1.tolist(),
        'avg_weight_mix': avg_weight_mix.tolist(),
        'avg_weight_A2': avg_weight_A2.tolist(),
        'segment_lengths': {
            'A1': len(segment_A1),
            'Mix': len(segment_mix),
            'A2': len(segment_A2)
        }
    }
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"Switch-lag Mix->A: {switch_lag_mixA} tokens (K=3 consecutive tokens with w>0.9)")
    print(f"Return-gap (cosine_distance): {return_gap_cosine:.4f} (0.0=identical, 1.0=maximally different)")
    print(f"Return-gap (euclidean): {return_gap_euclidean:.4f}")
    print(f"Return-gap (DTW): {return_gap_dtw:.4f}")
    print(f"Average mix segment entropy: {avg_mix_entropy:.4f}")
    print(f"Mix balance (min/max ratio): {mix_balance:.4f} (1.0=perfectly balanced)")
    
    return results

def visualize_hysteresis(results_aba: Dict, results_amixba: Dict, adapter_names: List[str], save_path: str = "hysteresis_analysis.png"):
    """
    Visualizes hysteresis test results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: A→B→A - weights by tokens
    ax1 = axes[0, 0]
    weights_aba = torch.stack(results_aba['weights']).numpy()
    tokens_aba = range(len(weights_aba))
    
    for i, name in enumerate(adapter_names):
        ax1.plot(tokens_aba, weights_aba[:, i], label=name, linewidth=2)
    
    # Mark switching points
    ax1.axvline(x=results_aba['switch_AB'], color='red', linestyle='--', alpha=0.5, label='A->B')
    ax1.axvline(x=results_aba['switch_BA'], color='blue', linestyle='--', alpha=0.5, label='B->A')
    
    ax1.set_xlabel('Token', fontsize=12)
    ax1.set_ylabel('Adapter Weight', fontsize=12)
    ax1.set_title('A -> B -> A: Weight Trajectory', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: A→Mix→A - weights by tokens
    ax2 = axes[0, 1]
    weights_amixba = torch.stack(results_amixba['weights']).numpy()
    tokens_amixba = range(len(weights_amixba))
    
    for i, name in enumerate(adapter_names):
        ax2.plot(tokens_amixba, weights_amixba[:, i], label=name, linewidth=2)
    
    # Mark switching points
    ax2.axvline(x=results_amixba['switch_Amix'], color='red', linestyle='--', alpha=0.5, label='A->Mix')
    ax2.axvline(x=results_amixba['switch_mixA'], color='blue', linestyle='--', alpha=0.5, label='Mix->A')
    
    ax2.set_xlabel('Token', fontsize=12)
    ax2.set_ylabel('Adapter Weight', fontsize=12)
    ax2.set_title('A -> (A/B mix) -> A: Weight Trajectory', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Comparison of A1 and A2 for A→B→A
    ax3 = axes[1, 0]
    segment_A1 = torch.stack(results_aba['segment_A1']).numpy()
    segment_A2 = torch.stack(results_aba['segment_A2']).numpy()
    
    tokens_A1 = range(len(segment_A1))
    tokens_A2 = range(len(segment_A2))
    
    for i, name in enumerate(adapter_names):
        ax3.plot(tokens_A1, segment_A1[:, i], label=f'A1: {name}', linewidth=2, linestyle='-')
        ax3.plot(tokens_A2, segment_A2[:, i], label=f'A2: {name}', linewidth=2, linestyle='--')
    
    ax3.set_xlabel('Token (relative to segment)', fontsize=12)
    ax3.set_ylabel('Adapter Weight', fontsize=12)
    ax3.set_title(f'Return-gap: A1 vs A2 (cosine_dist={results_aba["return_gap_cosine_distance"]:.4f})', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparison of A1 and A2 for A→Mix→A
    ax4 = axes[1, 1]
    segment_A1_mix = torch.stack(results_amixba['segment_A1']).numpy()
    segment_A2_mix = torch.stack(results_amixba['segment_A2']).numpy()
    
    tokens_A1_mix = range(len(segment_A1_mix))
    tokens_A2_mix = range(len(segment_A2_mix))
    
    for i, name in enumerate(adapter_names):
        ax4.plot(tokens_A1_mix, segment_A1_mix[:, i], label=f'A1: {name}', linewidth=2, linestyle='-')
        ax4.plot(tokens_A2_mix, segment_A2_mix[:, i], label=f'A2: {name}', linewidth=2, linestyle='--')
    
    ax4.set_xlabel('Token (relative to segment)', fontsize=12)
    ax4.set_ylabel('Adapter Weight', fontsize=12)
    ax4.set_title(f'Return-gap: A1 vs A2 (cosine_dist={results_amixba["return_gap_cosine_distance"]:.4f})', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[OK] Plot saved: {save_path}")

def main():
    """
    Main function to run hysteresis tests.
    """
    print("\n" + "="*80)
    print("ROUTER HYSTERESIS TEST: Time Crystallization")
    print("="*80)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n>>> Loading model...")
    print("[INFO] NOTE: Model must be trained!")
    print("[INFO] If model is not trained, run temporal_lora.py first")
    
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
    
    # Load trained model weights if available
    checkpoint_path = "temporal_lora_checkpoint.pt"
    try:
        if os.path.exists(checkpoint_path):
            print(f"\n>>> Loading weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            # Load adapter weights
            for name, adapter_state in checkpoint['adapters'].items():
                if name in model.adapters:
                    model.adapters[name].load_state_dict(adapter_state)
                    print(f"[OK] Loaded adapter weights '{name}'")
            
            # Load Time Mixer weights
            if checkpoint['time_mixer'] is not None and model.time_mixer is not None:
                model.time_mixer.load_state_dict(checkpoint['time_mixer'])
                print(f"[OK] Loaded Time Mixer weights")
            
            print("[OK] Model loaded from checkpoint")
        else:
            print(f"[WARN] Checkpoint not found: {checkpoint_path}")
            print("[WARN] Using random weights - results will be uninformative")
            print("[WARN] Run temporal_lora.py first to train the model")
    except Exception as e:
        print(f"[WARN] Error loading checkpoint: {e}")
        print("[WARN] Using random weights")
    
    # Test prompts
    prompt_A = "To code or not to code, that is the question."
    prompt_B = "import torch; model = TemporalLoRA()"
    prompt_mix = "class Fate: def __init__(self, star_crossed=True):"
    
    # Test parameters
    tail_length = 10  # Tail length for comparison
    max_length = 150  # Maximum generation length
    
    # Test 1: A -> B -> A
    results_aba = test_sequence_aba(
        model, tokenizer,
        prompt_A=prompt_A,
        prompt_B=prompt_B,
        tail_length=tail_length,
        max_length=max_length
    )
    
    # Test 2: A -> (A/B mix) -> A
    results_amixba = test_sequence_amixba(
        model, tokenizer,
        prompt_A=prompt_A,
        prompt_mix=prompt_mix,
        tail_length=tail_length,
        max_length=max_length
    )
    
    # Visualization
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)
    
    try:
        visualize_hysteresis(
            results_aba, results_amixba,
            adapter_names=model.adapter_names,
            save_path="hysteresis_analysis.png"
        )
    except Exception as e:
        print(f"[WARN] Failed to create visualization: {e}")
        print("(matplotlib may not be installed)")
    
    # Save results to JSON
    results_summary = {
        'test_aba': {
            'switch_lag_AB': results_aba['switch_lag_AB'],
            'switch_lag_BA': results_aba['switch_lag_BA'],
            'switching_asymmetry': results_aba.get('switching_asymmetry'),
            'return_gap_cosine_distance': results_aba['return_gap_cosine_distance'],
            'return_gap_euclidean': results_aba['return_gap_euclidean'],
            'return_gap_dtw': results_aba['return_gap_dtw'],
            'segment_lengths': results_aba['segment_lengths']
        },
        'test_amixba': {
            'switch_lag_mixA': results_amixba['switch_lag_mixA'],
            'return_gap_cosine_distance': results_amixba['return_gap_cosine_distance'],
            'return_gap_euclidean': results_amixba['return_gap_euclidean'],
            'return_gap_dtw': results_amixba['return_gap_dtw'],
            'avg_mix_entropy': results_amixba['avg_mix_entropy'],
            'mix_balance': results_amixba.get('mix_balance'),
            'segment_lengths': results_amixba['segment_lengths']
        }
    }
    
    json_path = "hysteresis_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved to: {json_path}")
    
    # Final report
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    print(f"\nTest A->B->A:")
    print(f"  Switch-lag A->B: {results_aba['switch_lag_AB']} tokens (K=3 consecutive with w>0.9)")
    print(f"  Switch-lag B->A: {results_aba['switch_lag_BA']} tokens (K=3 consecutive with w>0.9)")
    if results_aba.get('switching_asymmetry') is not None:
        print(f"  Switching asymmetry: {results_aba['switching_asymmetry']:.2f} tokens")
    print(f"  Return-gap (cosine_distance): {results_aba['return_gap_cosine_distance']:.4f} (0.0=identical)")
    print(f"  Return-gap (euclidean): {results_aba['return_gap_euclidean']:.4f}")
    print(f"  Return-gap (DTW): {results_aba['return_gap_dtw']:.4f}")
    
    print(f"\nTest A->Mix->A:")
    print(f"  Switch-lag Mix->A: {results_amixba['switch_lag_mixA']} tokens (K=3 consecutive with w>0.9)")
    print(f"  Return-gap (cosine_distance): {results_amixba['return_gap_cosine_distance']:.4f} (0.0=identical)")
    print(f"  Return-gap (euclidean): {results_amixba['return_gap_euclidean']:.4f}")
    print(f"  Return-gap (DTW): {results_amixba['return_gap_dtw']:.4f}")
    print(f"  Average mix segment entropy: {results_amixba['avg_mix_entropy']:.4f}")
    if results_amixba.get('mix_balance') is not None:
        print(f"  Mix balance: {results_amixba['mix_balance']:.4f} (1.0=perfectly balanced)")
    
    print(f"\n{'='*80}")
    print("RESULT INTERPRETATION")
    print(f"{'='*80}")
    print("""
Switch-lag (switching inertia):
  - High switch-lag (>5 tokens) -> inertia exists, router "crystallizes"
  - Low switch-lag (<=2 tokens) -> fast switching, no inertia

Return-gap (trajectory memory):
  - Metric: cosine_distance = 1 - cosine_similarity (0.0=identical, 1.0=maximally different)
  - High return-gap (>0.1) -> second "A" differs from first -> trajectory memory exists
  - Low return-gap (<0.05) -> second "A" similar to first -> no trajectory memory

If time crystallization exists:
  + Switch-lag will be high
  + Return-gap will be high
  + This confirms "epoch inertia" and "trajectory memory"
    """)
    
    print("="*80)

if __name__ == "__main__":
    main()


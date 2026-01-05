# =========================
# Router Hysteresis Test для LLaMA-3-8B: Time Crystallization
# =========================
"""
Router hysteresis (crystallization as "epoch inertia") для LLaMA-3-8B

Test sequences:
- A → B → A
- A → (A/B mix) → A

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
from transformers import AutoTokenizer
from temporal_lora_llama3 import TemporalLoRALlama3Model, DEVICE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean

def generate_with_token_weights(
    model: TemporalLoRALlama3Model,
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
    """Computes switch-lag: number of tokens needed after switch_point"""
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

def compute_return_gap(
    weights_A1: List[torch.Tensor],
    weights_A2: List[torch.Tensor],
    method: str = "cosine"
) -> float:
    """Computes return-gap: how much second "A" differs from first "A"."""
    if len(weights_A1) == 0 or len(weights_A2) == 0:
        return float('inf')
    
    w1 = torch.stack(weights_A1).float().cpu().numpy()
    w2 = torch.stack(weights_A2).float().cpu().numpy()
    
    if method == "cosine" or method == "cosine_distance":
        min_len = min(len(w1), len(w2))
        w1_norm = w1[:min_len]
        w2_norm = w2[:min_len]
        
        distances = []
        for i in range(min_len):
            dist = cosine(w1_norm[i], w2_norm[i])
            distances.append(dist)
        
        return np.mean(distances)
    
    elif method == "euclidean":
        min_len = min(len(w1), len(w2))
        w1_norm = w1[:min_len]
        w2_norm = w2[:min_len]
        
        distances = []
        for i in range(min_len):
            dist = euclidean(w1_norm[i], w2_norm[i])
            distances.append(dist)
        
        return np.mean(distances)
    
    elif method == "dtw":
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
    model: TemporalLoRALlama3Model,
    tokenizer,
    prompt_A: str,
    prompt_B: str,
    tail_length: int = 10,
    max_length: int = 150
) -> Dict:
    """Test sequence A → B → A."""
    print(f"\n{'='*80}", flush=True)
    print(f"TEST: A -> B -> A", flush=True)
    print(f"{'='*80}", flush=True)
    
    full_prompt = f"{prompt_A} {prompt_B} {prompt_A}"
    print(f"Full prompt: '{full_prompt}'", flush=True)
    
    text, weights, token_ids = generate_with_token_weights(
        model, tokenizer, full_prompt, max_length=max_length
    )
    
    weights_array = torch.stack(weights).float().cpu().numpy()
    
    # Find switching points
    switch_points = []
    for i in range(1, len(weights)):
        if weights_array[i-1, 1] < 0.5 and weights_array[i, 1] >= 0.5:
            switch_points.append(i)
        elif weights_array[i-1, 1] >= 0.5 and weights_array[i, 1] < 0.5:
            switch_points.append(i)
    
    print(f"\nFound switching points: {switch_points}", flush=True)
    
    if len(switch_points) < 2:
        prompt_A_tokens = len(tokenizer.encode(prompt_A))
        prompt_B_tokens = len(tokenizer.encode(prompt_B))
        switch_AB = prompt_A_tokens
        switch_BA = prompt_A_tokens + prompt_B_tokens
        switch_points = [switch_AB, switch_BA]
        print(f"Using heuristic: switch_AB={switch_AB}, switch_BA={switch_BA}", flush=True)
    
    if len(switch_points) >= 2:
        switch_AB = switch_points[0]
        switch_BA = switch_points[1]
    else:
        switch_AB = len(weights) // 3
        switch_BA = 2 * len(weights) // 3
        print(f"Fallback: switch_AB={switch_AB}, switch_BA={switch_BA}", flush=True)
    
    segment_A1 = weights[:switch_AB]
    segment_B = weights[switch_AB:switch_BA]
    segment_A2 = weights[switch_BA:]
    
    if len(segment_A1) > tail_length:
        segment_A1 = segment_A1[-tail_length:]
    if len(segment_A2) > tail_length:
        segment_A2 = segment_A2[-tail_length:]
    
    print(f"\nSegment lengths:", flush=True)
    print(f"  A1 (tail): {len(segment_A1)} tokens", flush=True)
    print(f"  B: {len(segment_B)} tokens", flush=True)
    print(f"  A2 (tail): {len(segment_A2)} tokens", flush=True)
    
    switch_lag_AB = compute_switch_lag(weights, domain_idx=1, switch_point=switch_AB,
                                       threshold=0.9, consecutive_tokens=3)
    switch_lag_BA = compute_switch_lag(weights, domain_idx=0, switch_point=switch_BA,
                                        threshold=0.9, consecutive_tokens=3)
    
    return_gap_cosine = compute_return_gap(segment_A1, segment_A2, method="cosine_distance")
    return_gap_euclidean = compute_return_gap(segment_A1, segment_A2, method="euclidean")
    return_gap_dtw = compute_return_gap(segment_A1, segment_A2, method="dtw")
    
    switching_asymmetry = None
    if switch_lag_AB is not None and switch_lag_BA is not None:
        switching_asymmetry = abs(switch_lag_AB - switch_lag_BA)
    
    segment_A1_weights = torch.stack(segment_A1).float().cpu().numpy()
    segment_B_weights = torch.stack(segment_B).float().cpu().numpy()
    segment_A2_weights = torch.stack(segment_A2).float().cpu().numpy()
    
    avg_weight_A1 = np.mean(segment_A1_weights, axis=0)
    avg_weight_B = np.mean(segment_B_weights, axis=0)
    avg_weight_A2 = np.mean(segment_A2_weights, axis=0)
    
    results = {
        'sequence': 'A->B->A',
        'text': text,
        'weights_json': [w.tolist() for w in weights],
        'token_ids': token_ids,
        'switch_AB': switch_AB,
        'switch_BA': switch_BA,
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
    
    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS:", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Switch-lag A->B: {switch_lag_AB} tokens", flush=True)
    print(f"Switch-lag B->A: {switch_lag_BA} tokens", flush=True)
    print(f"Return-gap (cosine_distance): {return_gap_cosine:.4f}", flush=True)
    print(f"Return-gap (euclidean): {return_gap_euclidean:.4f}", flush=True)
    print(f"Return-gap (DTW): {return_gap_dtw:.4f}", flush=True)
    
    return results

def test_sequence_amixba(
    model: TemporalLoRALlama3Model,
    tokenizer,
    prompt_A: str,
    prompt_mix: str,
    tail_length: int = 10,
    max_length: int = 150
) -> Dict:
    """Test sequence A → (A/B mix) → A."""
    print(f"\n{'='*80}", flush=True)
    print(f"TEST: A -> (A/B mix) -> A", flush=True)
    print(f"{'='*80}", flush=True)
    
    full_prompt = f"{prompt_A} {prompt_mix} {prompt_A}"
    print(f"Full prompt: '{full_prompt}'", flush=True)
    
    text, weights, token_ids = generate_with_token_weights(
        model, tokenizer, full_prompt, max_length=max_length
    )
    
    weights_array = torch.stack(weights).float().cpu().numpy()
    
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
    
    print(f"\nFound switching points: {switch_points}", flush=True)
    
    if len(switch_points) < 2:
        prompt_A_tokens = len(tokenizer.encode(prompt_A))
        prompt_mix_tokens = len(tokenizer.encode(prompt_mix))
        switch_Amix = prompt_A_tokens
        switch_mixA = prompt_A_tokens + prompt_mix_tokens
        switch_points = [switch_Amix, switch_mixA]
        print(f"Using heuristic: switch_Amix={switch_Amix}, switch_mixA={switch_mixA}", flush=True)
    
    if len(switch_points) >= 2:
        switch_Amix = switch_points[0]
        switch_mixA = switch_points[1]
    else:
        switch_Amix = len(weights) // 3
        switch_mixA = 2 * len(weights) // 3
        print(f"Fallback: switch_Amix={switch_Amix}, switch_mixA={switch_mixA}", flush=True)
    
    segment_A1 = weights[:switch_Amix]
    segment_mix = weights[switch_Amix:switch_mixA]
    segment_A2 = weights[switch_mixA:]
    
    if len(segment_A1) > tail_length:
        segment_A1 = segment_A1[-tail_length:]
    if len(segment_A2) > tail_length:
        segment_A2 = segment_A2[-tail_length:]
    
    print(f"\nSegment lengths:", flush=True)
    print(f"  A1 (tail): {len(segment_A1)} tokens", flush=True)
    print(f"  Mix: {len(segment_mix)} tokens", flush=True)
    print(f"  A2 (tail): {len(segment_A2)} tokens", flush=True)
    
    switch_lag_mixA = compute_switch_lag(weights, domain_idx=0, switch_point=switch_mixA,
                                         threshold=0.9, consecutive_tokens=3)
    
    return_gap_cosine = compute_return_gap(segment_A1, segment_A2, method="cosine_distance")
    return_gap_euclidean = compute_return_gap(segment_A1, segment_A2, method="euclidean")
    return_gap_dtw = compute_return_gap(segment_A1, segment_A2, method="dtw")
    
    mix_weights_array = torch.stack(segment_mix).float().cpu().numpy()
    mix_entropy = []
    for w in mix_weights_array:
        entropy = -np.sum(w * np.log(w + 1e-10))
        mix_entropy.append(entropy)
    avg_mix_entropy = np.mean(mix_entropy)
    
    segment_A1_weights = torch.stack(segment_A1).float().cpu().numpy()
    segment_mix_weights = torch.stack(segment_mix).float().cpu().numpy()
    segment_A2_weights = torch.stack(segment_A2).float().cpu().numpy()
    
    avg_weight_A1 = np.mean(segment_A1_weights, axis=0)
    avg_weight_mix = np.mean(segment_mix_weights, axis=0)
    avg_weight_A2 = np.mean(segment_A2_weights, axis=0)
    
    mix_balance = np.min(avg_weight_mix) / np.max(avg_weight_mix) if np.max(avg_weight_mix) > 0 else 0.0
    
    results = {
        'sequence': 'A->Mix->A',
        'text': text,
        'weights_json': [w.tolist() for w in weights],
        'token_ids': token_ids,
        'switch_Amix': switch_Amix,
        'switch_mixA': switch_mixA,
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
    
    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS:", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Switch-lag Mix->A: {switch_lag_mixA} tokens", flush=True)
    print(f"Return-gap (cosine_distance): {return_gap_cosine:.4f}", flush=True)
    print(f"Return-gap (euclidean): {return_gap_euclidean:.4f}", flush=True)
    print(f"Return-gap (DTW): {return_gap_dtw:.4f}", flush=True)
    print(f"Average mix segment entropy: {avg_mix_entropy:.4f}", flush=True)
    print(f"Mix balance: {mix_balance:.4f}", flush=True)
    
    return results

def visualize_hysteresis(results_aba: Dict, results_amixba: Dict, adapter_names: List[str], save_path: str = "hysteresis_analysis_llama3.png"):
    """Visualizes hysteresis test results."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: A→B→A
        ax1 = axes[0, 0]
        weights_aba = np.array(results_aba['weights_json'])
        tokens_aba = range(len(weights_aba))
        
        for i, name in enumerate(adapter_names):
            ax1.plot(tokens_aba, weights_aba[:, i], label=name, linewidth=2)
        
        ax1.axvline(x=results_aba['switch_AB'], color='red', linestyle='--', alpha=0.5, label='A->B')
        ax1.axvline(x=results_aba['switch_BA'], color='blue', linestyle='--', alpha=0.5, label='B->A')
        
        ax1.set_xlabel('Token', fontsize=12)
        ax1.set_ylabel('Adapter Weight', fontsize=12)
        ax1.set_title('A -> B -> A: Weight Trajectory', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: A→Mix→A
        ax2 = axes[0, 1]
        weights_amixba = np.array(results_amixba['weights_json'])
        tokens_amixba = range(len(weights_amixba))
        
        for i, name in enumerate(adapter_names):
            ax2.plot(tokens_amixba, weights_amixba[:, i], label=name, linewidth=2)
        
        ax2.axvline(x=results_amixba['switch_Amix'], color='red', linestyle='--', alpha=0.5, label='A->Mix')
        ax2.axvline(x=results_amixba['switch_mixA'], color='blue', linestyle='--', alpha=0.5, label='Mix->A')
        
        ax2.set_xlabel('Token', fontsize=12)
        ax2.set_ylabel('Adapter Weight', fontsize=12)
        ax2.set_title('A -> (A/B mix) -> A: Weight Trajectory', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Comparison A1 vs A2 for A→B→A
        ax3 = axes[1, 0]
        segment_A1 = np.array([w[:len(adapter_names)] for w in results_aba['weights_json'][:results_aba['switch_AB']]])
        segment_A2 = np.array([w[:len(adapter_names)] for w in results_aba['weights_json'][results_aba['switch_BA']:]])
        
        if len(segment_A1) > 0 and len(segment_A2) > 0:
            tokens_A1 = range(len(segment_A1))
            tokens_A2 = range(len(segment_A2))
            
            for i, name in enumerate(adapter_names):
                if len(segment_A1) > 0:
                    ax3.plot(tokens_A1, segment_A1[:, i], label=f'A1: {name}', linewidth=2, linestyle='-')
                if len(segment_A2) > 0:
                    ax3.plot(tokens_A2, segment_A2[:, i], label=f'A2: {name}', linewidth=2, linestyle='--')
        
        ax3.set_xlabel('Token (relative to segment)', fontsize=12)
        ax3.set_ylabel('Adapter Weight', fontsize=12)
        ax3.set_title(f'Return-gap: A1 vs A2 (cosine_dist={results_aba["return_gap_cosine_distance"]:.4f})', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Comparison A1 vs A2 for A→Mix→A
        ax4 = axes[1, 1]
        segment_A1_mix = np.array([w[:len(adapter_names)] for w in results_amixba['weights_json'][:results_amixba['switch_Amix']]])
        segment_A2_mix = np.array([w[:len(adapter_names)] for w in results_amixba['weights_json'][results_amixba['switch_mixA']:]])
        
        if len(segment_A1_mix) > 0 and len(segment_A2_mix) > 0:
            tokens_A1_mix = range(len(segment_A1_mix))
            tokens_A2_mix = range(len(segment_A2_mix))
            
            for i, name in enumerate(adapter_names):
                if len(segment_A1_mix) > 0:
                    ax4.plot(tokens_A1_mix, segment_A1_mix[:, i], label=f'A1: {name}', linewidth=2, linestyle='-')
                if len(segment_A2_mix) > 0:
                    ax4.plot(tokens_A2_mix, segment_A2_mix[:, i], label=f'A2: {name}', linewidth=2, linestyle='--')
        
        ax4.set_xlabel('Token (relative to segment)', fontsize=12)
        ax4.set_ylabel('Adapter Weight', fontsize=12)
        ax4.set_title(f'Return-gap: A1 vs A2 (cosine_dist={results_amixba["return_gap_cosine_distance"]:.4f})', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[OK] Plot saved: {save_path}", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to create visualization: {e}", flush=True)

def main():
    """Main function to run hysteresis tests."""
    print("\n" + "="*80, flush=True)
    print("ROUTER HYSTERESIS TEST: Time Crystallization (LLaMA-3-8B)", flush=True)
    print("="*80, flush=True)
    
    model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n>>> Loading model...", flush=True)
    model = TemporalLoRALlama3Model(
        model_name=model_name,
        lora_rank=8,
        lora_alpha=16.0,
        mixer_strategy="gating",
        freeze_backbone=True
    )
    model = model.to(DEVICE)
    
    model.add_adapter("shakespeare", "Renaissance Era (Shakespeare)")
    model.add_adapter("python", "IT Era (Python)")
    
    checkpoint_path = "temporal_lora_checkpoint_llama3.pt"
    try:
        if os.path.exists(checkpoint_path):
            print(f"\n>>> Loading weights from {checkpoint_path}...", flush=True)
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            for name, adapter_state in checkpoint['adapters'].items():
                if name in model.adapters:
                    model.adapters[name].load_state_dict(adapter_state)
                    print(f"[OK] Loaded adapter weights '{name}'", flush=True)
            
            if checkpoint['time_mixer'] is not None and model.time_mixer is not None:
                model.time_mixer.load_state_dict(checkpoint['time_mixer'])
                print(f"[OK] Loaded Time Mixer weights", flush=True)
            
            print("[OK] Model loaded from checkpoint", flush=True)
        else:
            print(f"[WARN] Checkpoint not found: {checkpoint_path}", flush=True)
            print("[WARN] Using random weights - results will be uninformative", flush=True)
    except Exception as e:
        print(f"[WARN] Error loading checkpoint: {e}", flush=True)
    
    prompt_A = "To code or not to code, that is the question."
    prompt_B = "import torch; model = TemporalLoRA()"
    prompt_mix = "class Fate: def __init__(self, star_crossed=True):"
    
    tail_length = 10
    max_length = 150
    
    results_aba = test_sequence_aba(
        model, tokenizer,
        prompt_A=prompt_A,
        prompt_B=prompt_B,
        tail_length=tail_length,
        max_length=max_length
    )
    
    results_amixba = test_sequence_amixba(
        model, tokenizer,
        prompt_A=prompt_A,
        prompt_mix=prompt_mix,
        tail_length=tail_length,
        max_length=max_length
    )
    
    print("\n" + "="*80, flush=True)
    print("CREATING VISUALIZATION", flush=True)
    print("="*80, flush=True)
    
    try:
        visualize_hysteresis(
            results_aba, results_amixba,
            adapter_names=model.adapter_names,
            save_path="hysteresis_analysis_llama3.png"
        )
    except Exception as e:
        print(f"[WARN] Failed to create visualization: {e}", flush=True)
    
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
    
    json_path = "hysteresis_results_llama3.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved to: {json_path}", flush=True)
    
    print("\n" + "="*80, flush=True)
    print("FINAL REPORT", flush=True)
    print("="*80, flush=True)
    print(f"\nTest A->B->A:", flush=True)
    print(f"  Switch-lag A->B: {results_aba['switch_lag_AB']} tokens", flush=True)
    print(f"  Switch-lag B->A: {results_aba['switch_lag_BA']} tokens", flush=True)
    if results_aba.get('switching_asymmetry') is not None:
        print(f"  Switching asymmetry: {results_aba['switching_asymmetry']:.2f} tokens", flush=True)
    print(f"  Return-gap (cosine_distance): {results_aba['return_gap_cosine_distance']:.4f}", flush=True)
    
    print(f"\nTest A->Mix->A:", flush=True)
    print(f"  Switch-lag Mix->A: {results_amixba['switch_lag_mixA']} tokens", flush=True)
    print(f"  Return-gap (cosine_distance): {results_amixba['return_gap_cosine_distance']:.4f}", flush=True)
    print(f"  Average mix segment entropy: {results_amixba['avg_mix_entropy']:.4f}", flush=True)
    
    print("="*80, flush=True)

if __name__ == "__main__":
    main()


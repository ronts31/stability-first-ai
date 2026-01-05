# =========================
# Router Fatigue Test для LLaMA-3-8B: Deep Crystallization
# =========================
"""
Check of inertia dependence on domain stay length для LLaMA-3-8B.

Hypothesis: the longer the model stays in one mode (domain),
the harder it is to "persuade" it to return to another domain.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer
from temporal_lora_llama3 import TemporalLoRALlama3Model, DEVICE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

def convert_to_python_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        return obj

def generate_with_token_weights(
    model: TemporalLoRALlama3Model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    use_mixer: bool = True
) -> Tuple[str, List[torch.Tensor], List[int]]:
    """Generates text and returns router weights for each token."""
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
    """Computes switch-lag."""
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
    """Creates a Python code block of approximately given length."""
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
    
    block_parts = []
    current_tokens = 0
    
    while current_tokens < target_length:
        phrase = np.random.choice(python_phrases)
        phrase_tokens = len(tokenizer.encode(phrase))
        
        if current_tokens + phrase_tokens <= target_length * 1.2:
            block_parts.append(phrase)
            current_tokens += phrase_tokens
        else:
            break
    
    while current_tokens < target_length:
        phrase = np.random.choice(python_phrases)
        block_parts.append(phrase)
        current_tokens += len(tokenizer.encode(phrase))
    
    return " ".join(block_parts)

def compute_return_gap(weights_A1: List[torch.Tensor], weights_A2: List[torch.Tensor], method: str = "cosine") -> float:
    """Computes return-gap."""
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
    
    return 0.0

def test_fatigue(
    model: TemporalLoRALlama3Model,
    tokenizer,
    prompt_A: str,
    python_block_length: int,
    tail_length: int = 20,
    max_length: int = 200
) -> Dict:
    """Test of inertia dependence on domain stay length."""
    print(f"\n{'='*80}", flush=True)
    print(f"FATIGUE TEST: Python block length {python_block_length} tokens", flush=True)
    print(f"{'='*80}", flush=True)
    
    python_block = create_python_block(tokenizer, python_block_length)
    actual_python_tokens = len(tokenizer.encode(python_block))
    
    print(f"Created Python block: {actual_python_tokens} tokens", flush=True)
    
    full_prompt = f"{prompt_A} {python_block} {prompt_A}"
    text, weights, token_ids = generate_with_token_weights(
        model, tokenizer, full_prompt, max_length=max_length
    )
    
    if len(weights) == 0:
        print("[ERROR] No weights for analysis", flush=True)
        return None
    
    weights_array = torch.stack(weights).float().cpu().numpy()
    
    prompt_A_tokens = len(tokenizer.encode(prompt_A))
    switch_AP = min(prompt_A_tokens, len(weights) - 1)
    switch_PA = min(prompt_A_tokens + actual_python_tokens, len(weights) - 1)
    
    segment_A1 = weights[:switch_AP] if switch_AP > 0 else []
    segment_Python = weights[switch_AP:switch_PA] if switch_PA > switch_AP else []
    segment_A2 = weights[switch_PA:] if switch_PA < len(weights) else []
    
    if len(segment_A1) > tail_length:
        segment_A1 = segment_A1[-tail_length:]
    if len(segment_A2) > tail_length:
        segment_A2 = segment_A2[:tail_length]
    
    switch_lag_PA = compute_switch_lag(
        weights, domain_idx=0, switch_point=switch_PA,
        threshold=0.9, consecutive_tokens=3
    )
    
    if len(segment_Python) > 0:
        python_weights_array = torch.stack(segment_Python).float().cpu().numpy()
        avg_python_weight = np.mean(python_weights_array[:, 1])
        final_python_weight = segment_Python[-1][1].item()
        initial_python_weight = segment_Python[0][1].item()
        
        deep_crystallization_count = np.sum(python_weights_array[:, 1] > 0.95)
        deep_crystallization_ratio = deep_crystallization_count / len(python_weights_array)
        
        tail_size = max(64, len(segment_Python) // 4)
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
        deep_crystallization_tail = 0.0
        avg_python_weight_tail = 0.0
    
    relax_time_099 = compute_switch_lag(
        weights, domain_idx=0, switch_point=switch_PA,
        threshold=0.99, consecutive_tokens=3
    )
    
    if len(segment_A2) >= 32:
        tail_32_weights = torch.stack(segment_A2[:32]).float().cpu().numpy()
        tail_area_32 = np.sum(1.0 - tail_32_weights[:, 0])
    else:
        tail_32_weights = torch.stack(segment_A2).float().cpu().numpy() if len(segment_A2) > 0 else np.array([[0.5, 0.5]])
        tail_area_32 = np.sum(1.0 - tail_32_weights[:, 0]) if len(segment_A2) > 0 else 0.0
    
    return_gap_cosine = compute_return_gap(segment_A1, segment_A2, method="cosine_distance")
    
    results = {
        'python_block_length': actual_python_tokens,
        'switch_lag_PA': switch_lag_PA,
        'avg_python_weight': avg_python_weight,
        'initial_python_weight': initial_python_weight,
        'final_python_weight': final_python_weight,
        'deep_crystallization_ratio': deep_crystallization_ratio,
        'deep_crystallization_tail': deep_crystallization_tail,
        'avg_python_weight_tail': avg_python_weight_tail,
        'relax_time_099': relax_time_099,
        'tail_area_32': tail_area_32,
        'return_gap_cosine': return_gap_cosine,
    }
    
    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS:", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Python block length: {actual_python_tokens} tokens", flush=True)
    print(f"Switch-lag Python->A: {switch_lag_PA} tokens", flush=True)
    print(f"Deep crystallization ratio (w>0.95): {deep_crystallization_ratio:.4f}", flush=True)
    print(f"Relax-time to 0.99: {relax_time_099} tokens" if relax_time_099 is not None else "Relax-time to 0.99: not reached", flush=True)
    print(f"Tail-area (first 32 tokens): {tail_area_32:.4f}", flush=True)
    
    return results

def test_fatigue_sweep(
    model: TemporalLoRALlama3Model,
    tokenizer,
    prompt_A: str,
    python_lengths: List[int] = [10, 50, 100, 200, 500],
    tail_length: int = 20,
    max_length: int = 800
) -> Dict:
    """Sweep test: checks inertia dependence on Python block length."""
    print("\n" + "="*80, flush=True)
    print("SWEEP TEST: Inertia Dependence on Domain Stay Length", flush=True)
    print("="*80, flush=True)
    
    all_results = []
    
    for length in python_lengths:
        print(f"\n{'='*80}", flush=True)
        print(f"Test {len(all_results) + 1}/{len(python_lengths)}: Python block {length} tokens", flush=True)
        print(f"{'='*80}", flush=True)
        
        try:
            result = test_fatigue(
                model, tokenizer,
                prompt_A=prompt_A,
                python_block_length=length,
                tail_length=tail_length,
                max_length=max_length
            )
            if result is not None:
                result['target_length'] = length
                all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Error in test for length {length}: {e}", flush=True)
    
    if len(all_results) == 0:
        print("[ERROR] No successful tests", flush=True)
        return {}
    
    # Compute correlations
    python_lengths_actual = [r['python_block_length'] for r in all_results]
    deep_cryst_ratios = [r['deep_crystallization_ratio'] for r in all_results]
    switch_lags = [r['switch_lag_PA'] if r['switch_lag_PA'] is not None else 0 for r in all_results]
    
    correlations = {}
    if len(python_lengths_actual) > 1:
        correlations['deep_crystallization'] = float(np.corrcoef(python_lengths_actual, deep_cryst_ratios)[0, 1])
        correlations['switch_lag'] = float(np.corrcoef(python_lengths_actual, switch_lags)[0, 1])
    
    # Visualization
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Deep crystallization ratio vs block length
        ax1 = axes[0, 0]
        ax1.scatter(python_lengths_actual, deep_cryst_ratios, s=100, alpha=0.7)
        ax1.plot(python_lengths_actual, deep_cryst_ratios, 'b--', alpha=0.5)
        ax1.set_xlabel('Python Block Length (tokens)', fontsize=12)
        ax1.set_ylabel('Deep Crystallization Ratio', fontsize=12)
        ax1.set_title(f'Deep Crystallization vs Block Length (r={correlations.get("deep_crystallization", 0):.4f})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Switch-lag vs block length
        ax2 = axes[0, 1]
        ax2.scatter(python_lengths_actual, switch_lags, s=100, alpha=0.7, color='orange')
        ax2.plot(python_lengths_actual, switch_lags, 'r--', alpha=0.5)
        ax2.set_xlabel('Python Block Length (tokens)', fontsize=12)
        ax2.set_ylabel('Switch-lag (tokens)', fontsize=12)
        ax2.set_title(f'Switch-lag vs Block Length (r={correlations.get("switch_lag", 0):.4f})', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Tail area vs block length
        tail_areas = [r['tail_area_32'] for r in all_results]
        ax3 = axes[1, 0]
        ax3.scatter(python_lengths_actual, tail_areas, s=100, alpha=0.7, color='green')
        ax3.plot(python_lengths_actual, tail_areas, 'g--', alpha=0.5)
        ax3.set_xlabel('Python Block Length (tokens)', fontsize=12)
        ax3.set_ylabel('Tail Area (first 32 tokens)', fontsize=12)
        ax3.set_title('Tail Area vs Block Length', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
Summary Statistics:

Deep Crystallization Correlation: {correlations.get('deep_crystallization', 0):.4f}
Switch-lag Correlation: {correlations.get('switch_lag', 0):.4f}

Tests Completed: {len(all_results)}/{len(python_lengths)}
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        plt.savefig("fatigue_analysis_llama3.png", dpi=150)
        print(f"\n[OK] Plot saved: fatigue_analysis_llama3.png", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to create visualization: {e}", flush=True)
    
    results_summary = {
        'tests': all_results,
        'correlations': correlations,
        'summary': {
            'num_tests': len(all_results),
            'python_lengths': python_lengths_actual,
            'avg_deep_crystallization': float(np.mean(deep_cryst_ratios)),
            'avg_switch_lag': float(np.mean(switch_lags)),
        }
    }
    
    json_path = "fatigue_results_llama3.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_python_types(results_summary), f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved to: {json_path}", flush=True)
    
    return results_summary

def main():
    """Main function to run fatigue tests."""
    print("\n" + "="*80, flush=True)
    print("ROUTER FATIGUE TEST: Deep Crystallization (LLaMA-3-8B)", flush=True)
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
    python_lengths = [10, 50, 100, 200, 500]
    
    results = test_fatigue_sweep(
        model, tokenizer,
        prompt_A=prompt_A,
        python_lengths=python_lengths,
        tail_length=20,
        max_length=800
    )
    
    print("\n" + "="*80, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("="*80, flush=True)
    if 'correlations' in results:
        print(f"Deep Crystallization Correlation: {results['correlations'].get('deep_crystallization', 0):.4f}", flush=True)
        print(f"Switch-lag Correlation: {results['correlations'].get('switch_lag', 0):.4f}", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()


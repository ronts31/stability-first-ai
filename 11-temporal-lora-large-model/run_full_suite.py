# =========================
# Full Test Suite for TemporalLoRA on Large Language Models
# =========================
"""
Main script for complete validation of all TemporalLoRA theories on large language models:
1. Environment check (PyTorch 2.8+, CUDA, BF16)
2. Model loading (LLaMA-3, Mistral, etc.)
3. Adapter training (Shakespeare, Python)
4. Time Mixer calibration
5. Hysteresis tests (A→B→A, A→Mix→A)
6. Fatigue tests (deep crystallization)
7. Output all results and metrics
"""

import os
import sys
import time
import json
import torch
import numpy as np
from transformers import AutoTokenizer

# Imports from our modules
from temporal_lora import (
    TemporalLoRAModel, DEVICE,
    DomainDataset, generate_shakespeare_data, generate_python_data,
    train_adapter, calibrate_mixer
)
from test_hysteresis import test_sequence_aba, test_sequence_amixba
from test_fatigue import test_fatigue_sweep

def parse_version(v: str):
    """Convert torch version string to tuple."""
    base = v.split("+")[0]
    parts = base.split(".")
    parts = (parts + ["0", "0", "0"])[:3]
    return tuple(int(x) for x in parts)

def check_environment():
    """Check execution environment."""
    print("="*80, flush=True)
    print("PHASE 0: Environment Check", flush=True)
    print("="*80, flush=True)
    
    print(f"torch.__version__ = {torch.__version__}", flush=True)
    v = parse_version(torch.__version__)
    if v < (2, 8, 0):
        raise RuntimeError("PyTorch >= 2.8.0 required for B200!")
    
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available, test will run on CPU (NOT B200).", flush=True)
    else:
        print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}", flush=True)
        props = torch.cuda.get_device_properties(0)
        print(f"GPU total memory (GB): {props.total_memory / 1024 ** 3:.2f}", flush=True)
    
    print("[OK] Environment check passed", flush=True)
    return True

def main():
    """Главная функция полного suite."""
    print("\n" + "="*80, flush=True)
    print("FULL SUITE: TemporalLoRA Theory Tests on Large Language Models", flush=True)
    print("="*80, flush=True)
    
    # Параметры
    model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    fast_mode = os.getenv("FAST_MODE", "True").lower() == "true"
    
    print(f"\nModel: {model_name}", flush=True)
    print(f"Fast mode: {fast_mode}", flush=True)
    
    # 0. Environment check
    check_environment()
    
    # 1. Load model and tokenizer
    print("\n" + "="*80, flush=True)
    print("PHASE 1: Loading Model", flush=True)
    print("="*80, flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("[INFO] Loading model (this may take time)...", flush=True)
    start_time = time.time()
    model = TemporalLoRAModel(
        model_name=model_name,
        lora_rank=8,
        lora_alpha=16.0,
        mixer_strategy="gating",
        freeze_backbone=True
    )
    model = model.to(DEVICE)
    load_time = time.time() - start_time
    print(f"[OK] Model loaded in {load_time:.1f} seconds", flush=True)
    
    # Add adapters
    model.add_adapter("shakespeare", "Renaissance Era (Shakespeare)")
    model.add_adapter("python", "IT Era (Python)")
    
    # Ensure all adapters and Time Mixer are on correct device
    for adapter in model.adapters.values():
        adapter = adapter.to(DEVICE)
    if model.time_mixer is not None:
        model.time_mixer = model.time_mixer.to(DEVICE)
    
    # 2. Data generation
    print("\n" + "="*80, flush=True)
    print("PHASE 2: Data Generation", flush=True)
    print("="*80, flush=True)
    
    n_samples = 50 if fast_mode else 200
    max_len = 32 if fast_mode else 64
    
    print(f"Generating Shakespeare data ({n_samples} examples)...", flush=True)
    shakespeare_texts = generate_shakespeare_data(n_samples=n_samples)
    print(f"[OK] Generated {len(shakespeare_texts)} Shakespeare examples", flush=True)
    
    print(f"Generating Python data ({n_samples} examples)...", flush=True)
    python_texts = generate_python_data(n_samples=n_samples)
    print(f"[OK] Generated {len(python_texts)} Python examples", flush=True)
    
    shakespeare_dataset = DomainDataset(shakespeare_texts, tokenizer, max_length=max_len)
    python_dataset = DomainDataset(python_texts, tokenizer, max_length=max_len)
    
    # 3. Adapter training
    print("\n" + "="*80, flush=True)
    print("PHASE 3: Training Adapters", flush=True)
    print("="*80, flush=True)
    
    adapter_epochs = 1 if fast_mode else 3
    batch_sz = 8 if fast_mode else 4
    
    phase3_start = time.time()
    
    # Train Shakespeare adapter
    train_adapter(
        model=model,
        tokenizer=tokenizer,
        dataset=shakespeare_dataset,
        adapter_name="shakespeare",
        epochs=adapter_epochs,
        lr=1e-4,
        batch_size=batch_sz,
        use_active_sleep=False
    )
    
    # Save teacher Time Mixer
    import copy
    teacher_mixer = None
    if model.time_mixer is not None:
        teacher_mixer = copy.deepcopy(model.time_mixer)
        for param in teacher_mixer.parameters():
            param.requires_grad = False
        teacher_mixer.eval()
        print("[OK] Teacher Time Mixer saved", flush=True)
    
    # Train Python adapter with Active Sleep
    train_adapter(
        model=model,
        tokenizer=tokenizer,
        dataset=python_dataset,
        adapter_name="python",
        epochs=adapter_epochs,
        lr=1e-4,
        batch_size=batch_sz,
        previous_datasets=[shakespeare_dataset],
        teacher_mixer=teacher_mixer,
        use_active_sleep=True
    )
    
    phase3_time = time.time() - phase3_start
    print(f"[OK] PHASE 3 completed in {phase3_time:.1f} seconds", flush=True)
    
    # 4. Time Mixer calibration
    print("\n" + "="*80, flush=True)
    print("PHASE 4: Time Mixer Calibration", flush=True)
    print("="*80, flush=True)
    
    calib_epochs = 10 if fast_mode else 20
    phase4_start = time.time()
    
    calibrate_mixer(
        model=model,
        tokenizer=tokenizer,
        datasetA=shakespeare_dataset,
        datasetB=python_dataset,
        adapter_nameA="shakespeare",
        adapter_nameB="python",
        epochs=calib_epochs,
        lr=2e-3,
        batch_size=batch_sz * 2
    )
    
    phase4_time = time.time() - phase4_start
    print(f"[OK] PHASE 4 completed in {phase4_time:.1f} seconds", flush=True)
    
    # Save checkpoint
    print("\n>>> Saving checkpoint...", flush=True)
    try:
        checkpoint = {
            'adapters': {name: adapter.state_dict() for name, adapter in model.adapters.items()},
            'time_mixer': model.time_mixer.state_dict() if model.time_mixer is not None else None,
            'adapter_names': model.adapter_names
        }
        torch.save(checkpoint, 'temporal_lora_checkpoint.pt')
        print(f"[OK] Checkpoint saved: temporal_lora_checkpoint.pt", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to save checkpoint: {e}", flush=True)
    
    # 5. Hysteresis tests
    print("\n" + "="*80, flush=True)
    print("PHASE 5: Hysteresis Tests", flush=True)
    print("="*80, flush=True)
    
    phase5_start = time.time()
    
    prompt_A = "To code or not to code, that is the question."
    prompt_B = "import torch; model = TemporalLoRA()"
    prompt_mix = "class Fate: def __init__(self, star_crossed=True):"
    
    results_aba = test_sequence_aba(
        model, tokenizer,
        prompt_A=prompt_A,
        prompt_B=prompt_B,
        tail_length=10,
        max_length=150
    )
    
    results_amixba = test_sequence_amixba(
        model, tokenizer,
        prompt_A=prompt_A,
        prompt_mix=prompt_mix,
        tail_length=10,
        max_length=150
    )
    
    phase5_time = time.time() - phase5_start
    print(f"[OK] PHASE 5 completed in {phase5_time:.1f} seconds", flush=True)
    
    # 6. Fatigue tests
    print("\n" + "="*80, flush=True)
    print("PHASE 6: Fatigue Tests (Deep Crystallization)", flush=True)
    print("="*80, flush=True)
    
    phase6_start = time.time()
    
    python_lengths = [10, 50, 100] if fast_mode else [10, 50, 100, 200, 500]
    fatigue_results = test_fatigue_sweep(
        model, tokenizer,
        prompt_A=prompt_A,
        python_lengths=python_lengths,
        tail_length=20,
        max_length=800
    )
    
    phase6_time = time.time() - phase6_start
    print(f"[OK] PHASE 6 completed in {phase6_time:.1f} seconds", flush=True)
    
    # 7. Save all results
    print("\n" + "="*80, flush=True)
    print("PHASE 7: Saving Results", flush=True)
    print("="*80, flush=True)
    
    all_results = {
        'model_name': model_name,
        'torch_version': torch.__version__,
        'fast_mode': fast_mode,
        'timing': {
            'model_load_sec': load_time,
            'phase3_training_sec': phase3_time,
            'phase4_calibration_sec': phase4_time,
            'phase5_hysteresis_sec': phase5_time,
            'phase6_fatigue_sec': phase6_time,
            'total_sec': load_time + phase3_time + phase4_time + phase5_time + phase6_time
        },
        'hysteresis': {
            'test_aba': {
                'switch_lag_AB': results_aba['switch_lag_AB'],
                'switch_lag_BA': results_aba['switch_lag_BA'],
                'switching_asymmetry': results_aba.get('switching_asymmetry'),
                'return_gap_cosine_distance': results_aba['return_gap_cosine_distance'],
                'return_gap_euclidean': results_aba['return_gap_euclidean'],
                'return_gap_dtw': results_aba['return_gap_dtw'],
            },
            'test_amixba': {
                'switch_lag_mixA': results_amixba['switch_lag_mixA'],
                'return_gap_cosine_distance': results_amixba['return_gap_cosine_distance'],
                'avg_mix_entropy': results_amixba['avg_mix_entropy'],
                'mix_balance': results_amixba.get('mix_balance'),
            }
        },
        'fatigue': fatigue_results if fatigue_results else {}
    }
    
    # Helper function to convert numpy types to Python types
    def convert_to_python_types(obj):
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
    
    results_path = "full_suite_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_python_types(all_results), f, indent=2, ensure_ascii=False)
    print(f"[OK] Results saved to: {results_path}", flush=True)
    
    # 8. Final report
    print("\n" + "="*80, flush=True)
    print("FINAL REPORT", flush=True)
    print("="*80, flush=True)
    
    print(f"\nTiming Summary:", flush=True)
    print(f"  Model load: {load_time:.1f}s", flush=True)
    print(f"  Training: {phase3_time:.1f}s", flush=True)
    print(f"  Calibration: {phase4_time:.1f}s", flush=True)
    print(f"  Hysteresis tests: {phase5_time:.1f}s", flush=True)
    print(f"  Fatigue tests: {phase6_time:.1f}s", flush=True)
    print(f"  Total: {all_results['timing']['total_sec']:.1f}s", flush=True)
    
    print(f"\nHysteresis Results:", flush=True)
    print(f"  A->B->A:", flush=True)
    print(f"    Switch-lag A->B: {results_aba['switch_lag_AB']} tokens", flush=True)
    print(f"    Switch-lag B->A: {results_aba['switch_lag_BA']} tokens", flush=True)
    print(f"    Return-gap: {results_aba['return_gap_cosine_distance']:.4f}", flush=True)
    print(f"  A->Mix->A:", flush=True)
    print(f"    Switch-lag Mix->A: {results_amixba['switch_lag_mixA']} tokens", flush=True)
    print(f"    Return-gap: {results_amixba['return_gap_cosine_distance']:.4f}", flush=True)
    
    if fatigue_results and 'correlations' in fatigue_results:
        print(f"\nFatigue Results:", flush=True)
        print(f"  Deep Crystallization Correlation: {fatigue_results['correlations'].get('deep_crystallization', 0):.4f}", flush=True)
        print(f"  Switch-lag Correlation: {fatigue_results['correlations'].get('switch_lag', 0):.4f}", flush=True)
    
    print("\n" + "="*80, flush=True)
    print("[OK] FULL SUITE COMPLETED SUCCESSFULLY", flush=True)
    print("="*80, flush=True)
    
    return all_results

if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Suite failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


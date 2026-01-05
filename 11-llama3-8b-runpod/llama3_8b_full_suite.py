import os
import time
import math
import json
from typing import List, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_version(v: str) -> Tuple[int, int, int]:
    """
    Преобразует строку версии torch вида "2.8.0+cu128" в кортеж (2, 8, 0).
    """
    base = v.split("+")[0]
    parts = base.split(".")
    parts = (parts + ["0", "0", "0"])[:3]
    return tuple(int(x) for x in parts)


def print_env_info() -> torch.device:
    """
    Печатает информацию о среде / устройстве и возвращает выбранный torch.device.
    """
    print("=== ENV / DEVICE INFO ===", flush=True)
    print(f"torch.__version__ = {torch.__version__}", flush=True)
    v = parse_version(torch.__version__)
    if v < (2, 8, 0):
        raise RuntimeError("Нужен PyTorch >= 2.8.0 для B200 (в образе Runpod он уже есть).")

    if not torch.cuda.is_available():
        print("[WARN] CUDA недоступен, тест будет на CPU (НЕ B200).", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)
        print(f"CUDA current device: {torch.cuda.current_device()}", flush=True)
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}", flush=True)
        props = torch.cuda.get_device_properties(0)
        print(f"GPU total memory (GB): {props.total_memory / 1024 ** 3:.2f}", flush=True)

    return device


def load_llama(model_name: str, device: torch.device):
    """
    Загружает LLaMA‑3‑8B (или совместимую causal LM) с оптимальными настройками для B200.
    """
    print("\n=== LOADING LLAMA MODEL ===", flush=True)
    print(f"Model: {model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Для B200: предпочтительно bfloat16, иначе fp16, иначе fp32
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Using dtype: {dtype}, device_map='auto' if CUDA", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()
    print("[OK] Модель загружена.", flush=True)
    return tokenizer, model


def simple_generation(tokenizer, model, device: torch.device) -> Dict[str, Any]:
    """
    Одиночная генерация текста (sanity check).
    """
    print("\n=== SIMPLE GENERATION TEST ===", flush=True)
    prompt = "Explain the concept of temporal dynamics in large language models in simple terms."
    print(f"Prompt:\n{prompt}\n", flush=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        start = time.time()
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        dt = time.time() - start

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text, flush=True)
    print(f"\n[OK] Simple generation done in {dt:.2f}s", flush=True)

    return {
        "type": "simple_generation",
        "prompt": prompt,
        "time_sec": dt,
        "generated_preview": text[:500],
    }


def batch_generation(tokenizer, model, device: torch.device) -> Dict[str, Any]:
    """
    Батчевая генерация по нескольким промптам.
    """
    print("\n=== BATCH GENERATION TEST ===", flush=True)
    prompts: List[str] = [
        "Write a short Shakespeare-style monologue about coding.",
        "Write a Python function that computes Fibonacci numbers with memoization.",
        "Describe the idea of 'time mixer' for language models.",
        "Explain why bfloat16 is useful for large models on modern GPUs.",
    ]
    batch = tokenizer(prompts, return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        start = time.time()
        out = model.generate(
            **batch,
            max_new_tokens=64,
            temperature=0.9,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        dt = time.time() - start

    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    previews: List[str] = []
    for i, t in enumerate(texts):
        snippet = t[:500]
        previews.append(snippet)
        print(f"\n--- Sample {i} ---")
        print(snippet, "...\n", flush=True)

    print(f"[OK] Batch generation for {len(prompts)} prompts in {dt:.2f}s", flush=True)

    return {
        "type": "batch_generation",
        "num_prompts": len(prompts),
        "time_sec": dt,
        "generated_previews": previews,
    }


def train_step_test(tokenizer, model, device: torch.device) -> Dict[str, Any]:
    """
    Один тренировочный шаг (forward + backward + optimizer.step)
    для проверки, что градиенты и оптимизация работают корректно.
    """
    print("\n=== TRAIN STEP TEST (forward + backward) ===", flush=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    prompt = "The concept of stability in learning dynamics can be connected to physics as follows:"
    batch = tokenizer(
        [prompt] * 2,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch["input_ids"].clone()

    start = time.time()
    out = model(**batch, labels=labels)
    loss = out.loss
    print(f"Loss (before step): {loss.item():.4f}", flush=True)

    loss.backward()

    # Оценка нормы градиента
    total_norm_sq = 0.0
    counted = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            if not math.isnan(param_norm) and not math.isinf(param_norm):
                total_norm_sq += param_norm ** 2
                counted += 1
    grad_norm = math.sqrt(total_norm_sq) if counted > 0 else 0.0
    print(f"Grad L2 norm (approx): {grad_norm:.4f}", flush=True)

    optimizer.step()
    optimizer.zero_grad()
    dt = time.time() - start
    print(f"[OK] One train step finished in {dt:.2f}s", flush=True)
    model.eval()

    return {
        "type": "train_step",
        "loss_before_step": float(loss.item()),
        "grad_l2_norm": float(grad_norm),
        "time_sec": dt,
    }


def throughput_benchmark(
    tokenizer,
    model,
    device: torch.device,
    seq_len: int = 512,
    batch_size: int = 4,
    steps: int = 5,
) -> Dict[str, Any]:
    """
    Простой бенчмарк пропускной способности (токенов в секунду)
    на фиксированной длине последовательности и размере батча.
    """
    print("\n=== THROUGHPUT BENCHMARK ===", flush=True)
    base_text = (
        "This is a dummy sentence for benchmarking large language model throughput. "
        "We repeat it multiple times to reach the desired sequence length. "
    )
    # Сконструировать достаточно длинный текст; последовательность всё равно режется tokenizer'ом
    long_prompt = (base_text * 64)[: seq_len * 8]

    batch = tokenizer(
        [long_prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        # Тёплый прогон
        _ = model(**batch)

        start = time.time()
        for _ in range(steps):
            _ = model(**batch)
        dt = time.time() - start

    tokens_per_step = batch["input_ids"].numel()
    total_tokens = tokens_per_step * steps
    tok_per_sec = total_tokens / dt if dt > 0 else float("inf")

    print(f"Batch size: {batch_size}, seq_len: {seq_len}, steps: {steps}", flush=True)
    print(f"Total tokens: {total_tokens}, time: {dt:.2f}s, throughput: {tok_per_sec:,.0f} tok/s", flush=True)
    print("[OK] Throughput benchmark finished.", flush=True)

    return {
        "type": "throughput_benchmark",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps": steps,
        "total_tokens": int(total_tokens),
        "time_sec": dt,
        "tokens_per_sec": tok_per_sec,
    }


def main():
    """
    Полный прогон проверки LLaMA‑3‑8B на B200 / PyTorch 2.8+.
    """
    # Можно переопределить модель через переменную окружения
    model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

    device = print_env_info()
    tokenizer, model = load_llama(model_name, device)

    results: Dict[str, Any] = {
        "model_name": model_name,
        "torch_version": torch.__version__,
        "device_type": device.type,
        "tests": [],
    }

    # 1. Одиночный инференс
    res_simple = simple_generation(tokenizer, model, device)
    results["tests"].append(res_simple)

    # 2. Батчевый инференс
    res_batch = batch_generation(tokenizer, model, device)
    results["tests"].append(res_batch)

    # 3. Тренировочный шаг
    res_train = train_step_test(tokenizer, model, device)
    results["tests"].append(res_train)

    # 4. Бенчмарк пропускной способности
    res_bench = throughput_benchmark(
        tokenizer,
        model,
        device,
        seq_len=512,
        batch_size=4,
        steps=5,
    )
    results["tests"].append(res_bench)

    # Сохранение результатов
    out_path = os.path.join(
        os.path.dirname(__file__),
        "llama3_8b_results.json",
    )
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Результаты сохранены в: {out_path}", flush=True)
    except Exception as e:
        print(f"\n[WARN] Не удалось сохранить результаты: {e}", flush=True)

    print("\n[OK] Полная проверка LLaMA‑3‑8B на B200 завершена.", flush=True)


if __name__ == "__main__":
    main()



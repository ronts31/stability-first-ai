# =========================
# Тестирование Temporal LoRA в разных условиях
# =========================

import subprocess
import sys
import time
import os

def run_test(config_name, modifications):
    """
    Запускает тест с заданными модификациями
    
    Args:
        config_name: Название конфигурации
        modifications: Словарь с изменениями в коде
    """
    print("\n" + "="*80)
    print(f"ТЕСТ: {config_name}")
    print("="*80)
    
    # Читаем оригинальный файл
    with open("temporal_lora.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Применяем модификации
    modified_content = content
    for old, new in modifications.items():
        modified_content = modified_content.replace(old, new)
    
    # Сохраняем временный файл
    temp_file = f"temporal_lora_temp_{config_name.replace(' ', '_').lower()}.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(modified_content)
    
    try:
        # Запускаем тест
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=600  # 10 минут максимум
        )
        elapsed = time.time() - start_time
        
        print(f"\nРезультат: {'УСПЕХ' if result.returncode == 0 else 'ОШИБКА'}")
        print(f"Время выполнения: {elapsed:.1f} секунд")
        
        # Выводим ключевые метрики
        output = result.stdout + result.stderr
        if "Router Acc:" in output:
            lines = output.split("\n")
            for line in lines:
                if "Router Acc:" in line:
                    print(f"  {line.strip()}")
        
        if "Average adapter weights" in output:
            lines = output.split("\n")
            in_weights = False
            for line in lines:
                if "Average adapter weights" in line:
                    in_weights = True
                if in_weights and ("shakespeare" in line or "python" in line):
                    print(f"  {line.strip()}")
                if in_weights and line.strip() == "":
                    break
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("ТЕСТ ПРЕРВАН: превышено время ожидания (10 минут)")
        return False
    except Exception as e:
        print(f"ОШИБКА при запуске: {e}")
        return False
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    """Запускает серию тестов в разных условиях"""
    
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ TEMPORAL LORA В РАЗНЫХ УСЛОВИЯХ")
    print("="*80)
    
    results = {}
    
    # Тест 1: Полный режим (FAST_MODE=False)
    print("\n>>> Тест 1: Полный режим (FAST_MODE=False)")
    results["Полный режим"] = run_test(
        "full_mode",
        {
            "FAST_MODE = True": "FAST_MODE = False"
        }
    )
    
    # Тест 2: Меньший LoRA rank
    print("\n>>> Тест 2: Меньший LoRA rank (rank=4)")
    results["LoRA rank=4"] = run_test(
        "lora_rank_4",
        {
            "FAST_MODE = True": "FAST_MODE = True",
            "lora_rank=8": "lora_rank=4"
        }
    )
    
    # Тест 3: Больший LoRA rank
    print("\n>>> Тест 3: Больший LoRA rank (rank=16)")
    results["LoRA rank=16"] = run_test(
        "lora_rank_16",
        {
            "FAST_MODE = True": "FAST_MODE = True",
            "lora_rank=8": "lora_rank=16"
        }
    )
    
    # Тест 4: Меньший LoRA alpha
    print("\n>>> Тест 4: Меньший LoRA alpha (alpha=8.0)")
    results["LoRA alpha=8.0"] = run_test(
        "lora_alpha_8",
        {
            "FAST_MODE = True": "FAST_MODE = True",
            "lora_alpha=16.0": "lora_alpha=8.0"
        }
    )
    
    # Тест 5: Больший LoRA alpha
    print("\n>>> Тест 5: Больший LoRA alpha (alpha=32.0)")
    results["LoRA alpha=32.0"] = run_test(
        "lora_alpha_32",
        {
            "FAST_MODE = True": "FAST_MODE = True",
            "lora_alpha=16.0": "lora_alpha=32.0"
        }
    )
    
    # Тест 6: Больше эпох для адаптеров
    print("\n>>> Тест 6: Больше эпох для адаптеров (epochs=5)")
    results["Больше эпох"] = run_test(
        "more_epochs",
        {
            "FAST_MODE = True": "FAST_MODE = True",
            "adapter_epochs = 1 if FAST_MODE else 3": "adapter_epochs = 5 if FAST_MODE else 3"
        }
    )
    
    # Тест 7: Больше данных
    print("\n>>> Тест 7: Больше данных (n_samples=100)")
    results["Больше данных"] = run_test(
        "more_data",
        {
            "FAST_MODE = True": "FAST_MODE = True",
            "n_samples = 50 if FAST_MODE else 200": "n_samples = 100 if FAST_MODE else 200"
        }
    )
    
    # Тест 8: Меньший learning rate
    print("\n>>> Тест 8: Меньший learning rate (lr=5e-5)")
    results["LR=5e-5"] = run_test(
        "lr_5e5",
        {
            "FAST_MODE = True": "FAST_MODE = True",
            "lr=1e-4": "lr=5e-5"
        }
    )
    
    # Тест 9: Больший learning rate
    print("\n>>> Тест 9: Больший learning rate (lr=2e-4)")
    results["LR=2e-4"] = run_test(
        "lr_2e4",
        {
            "FAST_MODE = True": "FAST_MODE = True",
            "lr=1e-4": "lr=2e-4"
        }
    )
    
    # Итоговый отчет
    print("\n" + "="*80)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*80)
    
    for test_name, success in results.items():
        status = "✓ УСПЕХ" if success else "✗ ОШИБКА"
        print(f"{test_name:30s}: {status}")
    
    success_count = sum(1 for s in results.values() if s)
    total_count = len(results)
    print(f"\nУспешно: {success_count}/{total_count} тестов")
    print("="*80)

if __name__ == "__main__":
    main()


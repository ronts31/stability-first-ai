import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import matplotlib.pyplot as plt

# 1. Простая архитектура (как в твоем примере)
class SimpleNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 2. Оператор Стабильности (упрощенная версия для "Сна")
# Мы хотим, чтобы f(x) и f(x + noise) были близки. 
# Это заставляет сеть "игнорировать" мелкие возмущения (сглаживает многообразие).
def stability_loss(model, x, epsilon=0.1):
    output = model(x)
    
    # Создаем возмущение входа (имитация внутреннего шума мозга)
    noise = torch.randn_like(x) * epsilon
    output_perturbed = model(x + noise)
    
    # Мы хотим, чтобы выход был стабилен несмотря на шум
    # Это создает "Широкий Аттрактор"
    loss = torch.nn.functional.mse_loss(output, output_perturbed)
    return loss

# 3. Фаза "Сна" (Dreaming Phase) - Версия 3.1: Tuned Elastic Weight Consolidation
def sleep_cycle(model, iterations=2000, consistency_weight=2.0, stability_weight=1.0, drift_weight=10.0):
    # Уменьшили LR для тонкой настройки
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    
    # 1. ЯКОРЬ ПОВЕДЕНИЯ (Output Anchor)
    frozen_model = copy.deepcopy(model)
    frozen_model.eval()
    
    # 2. ЯКОРЬ ВЕСОВ (Weight Anchor)
    # Сохраняем стартовые веса, чтобы не уплыть далеко
    initial_weights = [p.clone().detach() for p in model.parameters()]
    
    print(f"[SLEEP] Entering Elastic Sleep (Tuned: Weaker Drift, Stronger Consistency)...")
    print(f"Parameters: consistency_weight={consistency_weight}, stability_weight={stability_weight}, drift_weight={drift_weight}")
    
    losses_consistency = []
    losses_stability = []
    losses_drift = []
    losses_total = []
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Генерируем сны
        dream_input = torch.randn(64, 10)
        
        # A. Consistency Loss (Поведение)
        with torch.no_grad():
            target = frozen_model(dream_input)
        current = model(dream_input)
        loss_consistency = nn.functional.mse_loss(current, target)
        
        # B. Stability Loss (Устойчивость к шуму)
        dream_noise = torch.randn_like(dream_input) * 0.1
        perturbed = model(dream_input + dream_noise)
        loss_stability = nn.functional.mse_loss(current, perturbed)
        
        # C. Drift Loss (Штраф за изменение весов) !!! НОВОЕ !!!
        loss_drift = 0
        for p_curr, p_init in zip(model.parameters(), initial_weights):
            loss_drift += torch.sum((p_curr - p_init) ** 2)
        
        # ИТОГОВАЯ ФУНКЦИЯ ПОТЕРЬ
        # Баланс: сильный якорь поведения, умеренная стабильность, слабый дрейф
        total_loss = (consistency_weight * loss_consistency) + (stability_weight * loss_stability) + (drift_weight * loss_drift)
        
        losses_consistency.append(loss_consistency.item())
        losses_stability.append(loss_stability.item())
        losses_drift.append(loss_drift.item())
        losses_total.append(total_loss.item())
        
        total_loss.backward()
        optimizer.step()
        
        # Показываем прогресс каждые 200 итераций
        if i % 200 == 0:
            print(f"Iter {i}: Total={total_loss.item():.5f} (Consistency={loss_consistency.item():.5f}, "
                  f"Stability={loss_stability.item():.5f}, Drift={loss_drift.item():.5f})")
    
    print("\n[WAKE] Waking up. Structure consolidated.")
    print(f"Final losses - Total: {losses_total[-1]:.6f}, Consistency: {losses_consistency[-1]:.6f}, "
          f"Stability: {losses_stability[-1]:.6f}, Drift: {losses_drift[-1]:.6f}")
    
    # Статистика по изменениям
    print(f"\nLoss statistics:")
    print(f"  Total loss: min={min(losses_total):.6f}, max={max(losses_total):.6f}, "
          f"mean={np.mean(losses_total):.6f}, final={losses_total[-1]:.6f}")
    print(f"  Stability loss: min={min(losses_stability):.6f}, max={max(losses_stability):.6f}, "
          f"mean={np.mean(losses_stability):.6f}, final={losses_stability[-1]:.6f}")
    print(f"  Consistency loss: min={min(losses_consistency):.6f}, max={max(losses_consistency):.6f}, "
          f"mean={np.mean(losses_consistency):.6f}, final={losses_consistency[-1]:.6f}")
    print(f"  Drift loss: min={min(losses_drift):.6f}, max={max(losses_drift):.6f}, "
          f"mean={np.mean(losses_drift):.6f}, final={losses_drift[-1]:.6f}")
    
    return model

# --- ЗАПУСК ЭКСПЕРИМЕНТА ---

if __name__ == "__main__":
    print("=" * 70)
    print("Simple Sleep Demo - Elastic Weight Consolidation")
    print("NOTE: SimpleNet (~5000 params) is a toy model for demonstration.")
    print("Real stability effects require larger models (millions+ parameters).")
    print("=" * 70)
    print()
    
    # A. Подготовка
    input_dim = 10
    data_size = 500
    X = torch.randn(data_size, input_dim)
    Y = (X.sum(dim=1) > 0).long()  # Простая задача классификации

    model = SimpleNet(input_dim=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # B. Обучение (День)
    print("--- Phase 1: Learning (Day) ---")
    print("Training progress: ", end="", flush=True)
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        
        # Показываем прогресс каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                preds = model(X).argmax(dim=1)
                acc = (preds == Y).float().mean().item()
            print(f"{epoch+1}% (loss: {loss.item():.4f}, acc: {acc:.2f}) ", end="", flush=True)
    print("\nLearning complete.")
    
    # Оценка после обучения
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        acc = (preds == Y).float().mean().item()
        out = model(X)
        loss_val = criterion(out, Y).item()
        
        # Статистика по весам
        total_params = sum(p.numel() for p in model.parameters())
        weight_mean = torch.cat([p.flatten() for p in model.parameters()]).mean().item()
        weight_std = torch.cat([p.flatten() for p in model.parameters()]).std().item()
        weight_min = torch.cat([p.flatten() for p in model.parameters()]).min().item()
        weight_max = torch.cat([p.flatten() for p in model.parameters()]).max().item()
        
        print(f"\nFinal training metrics:")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Loss: {loss_val:.6f}")
        print(f"  Model statistics:")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Weight mean: {weight_mean:.6f}")
        print(f"    Weight std: {weight_std:.6f}")
        print(f"    Weight range: [{weight_min:.6f}, {weight_max:.6f}]")

    # C. Клонирование: Одна модель пойдет спать, другая нет
    model_insomniac = copy.deepcopy(model) # Не спит
    model_dreamer = copy.deepcopy(model)   # Пойдет спать
    
    # Сохраняем веса до сна для сравнения
    weights_before_sleep = [p.clone().detach() for p in model_dreamer.parameters()]

    # D. Сон (Без данных!)
    # Мы не показываем model_dreamer ни X, ни Y. Только шум.
    print("\n--- Phase 2: Sleep (Tuned Elastic Weight Consolidation) ---")
    # Ослабленный drift (10.0 вместо 100.0) дает свободу найти широкое плато
    # Усиленный consistency (2.0) сохраняет поведение
    sleep_cycle(model_dreamer, iterations=2000, consistency_weight=2.0, stability_weight=1.0, drift_weight=10.0)
    
    # Анализ изменений весов после сна
    with torch.no_grad():
        weight_changes = []
        for p_before, p_after in zip(weights_before_sleep, model_dreamer.parameters()):
            change = (p_after - p_before).abs()
            weight_changes.append(change.flatten())
        all_changes = torch.cat(weight_changes)
        
        print(f"\nWeight change statistics after sleep:")
        print(f"  Mean absolute change: {all_changes.mean().item():.8f}")
        print(f"  Max absolute change: {all_changes.max().item():.8f}")
        print(f"  Std of changes: {all_changes.std().item():.8f}")
        print(f"  Total weight drift: {all_changes.sum().item():.6f}")
        
        # Сравнение весов моделей
        insomniac_weights = torch.cat([p.flatten() for p in model_insomniac.parameters()])
        dreamer_weights = torch.cat([p.flatten() for p in model_dreamer.parameters()])
        weight_diff = (dreamer_weights - insomniac_weights).abs()
        print(f"\nWeight difference (Dreamer vs Insomniac):")
        print(f"  Mean difference: {weight_diff.mean().item():.8f}")
        print(f"  Max difference: {weight_diff.max().item():.8f}")
        print(f"  Total difference: {weight_diff.sum().item():.6f}")

    # E. Стресс-тест (Атака на веса)
    print("\n--- Phase 3: Robustness Test (Damage) ---")

    def damage_weights(m, noise_level=0.1):
        with torch.no_grad():
            for param in m.parameters():
                param.add_(torch.randn_like(param) * noise_level)

    # Наносим одинаковый урон обеим моделям
    # КРИТИЧЕСКОЕ повреждение! Insomniac должен упасть до случайного угадывания (~50%)
    noise_lvl = 1.5  # Увеличено с 0.8 до 1.5 для более жесткого теста
    print(f"Applying CRITICAL damage (noise_level={noise_lvl})...")
    print(f"  Expected: Insomniac should drop to ~50% or below (random guessing)")
    damage_weights(model_insomniac, noise_lvl)
    damage_weights(model_dreamer, noise_lvl)

    # F. Сравнение
    def evaluate(m, name):
        with torch.no_grad():
            preds = m(X).argmax(dim=1)
            acc = (preds == Y).float().mean().item()
            # Также вычисляем loss для полноты картины
            out = m(X)
            loss_val = criterion(out, Y).item()
            
            # Дополнительные метрики
            correct = (preds == Y).sum().item()
            total = Y.numel()
            confidences = torch.softmax(out, dim=1)
            avg_confidence = confidences.max(dim=1)[0].mean().item()
            
            # Статистика по весам после повреждения
            weights = torch.cat([p.flatten() for p in m.parameters()])
            weight_mean = weights.mean().item()
            weight_std = weights.std().item()
            weight_norm = weights.norm().item()
            
            print(f"Model {name}:")
            print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%) | Correct: {correct}/{total}")
            print(f"  Loss: {loss_val:.6f}")
            print(f"  Average confidence: {avg_confidence:.4f}")
            print(f"  Weight statistics:")
            print(f"    Mean: {weight_mean:.6f}, Std: {weight_std:.6f}, Norm: {weight_norm:.6f}")
            return acc, loss_val, weight_norm

    print(f"\nAfter applying {noise_lvl} noise to weights:")
    acc_insomniac, loss_insomniac, norm_insomniac = evaluate(model_insomniac, "Insomniac (No Sleep)")
    acc_dreamer, loss_dreamer, norm_dreamer = evaluate(model_dreamer, "Dreamer (Sleep)")
    
    print(f"\n--- Detailed Summary ---")
    print(f"Accuracy improvement: {acc_dreamer - acc_insomniac:+.4f} ({(acc_dreamer - acc_insomniac)*100:+.2f}%)")
    print(f"  Insomniac: {acc_insomniac:.4f} ({acc_insomniac*100:.2f}%)")
    print(f"  Dreamer:   {acc_dreamer:.4f} ({acc_dreamer*100:.2f}%)")
    print(f"  Relative improvement: {((acc_dreamer / acc_insomniac - 1) * 100):+.2f}%")
    
    print(f"\nLoss comparison:")
    print(f"  Insomniac: {loss_insomniac:.6f}")
    print(f"  Dreamer:   {loss_dreamer:.6f}")
    print(f"  Loss reduction: {((loss_insomniac - loss_dreamer) / loss_insomniac * 100):+.2f}%")
    
    print(f"\nWeight norm comparison:")
    print(f"  Insomniac: {norm_insomniac:.6f}")
    print(f"  Dreamer:   {norm_dreamer:.6f}")
    print(f"  Norm difference: {norm_dreamer - norm_insomniac:+.6f}")
    
    if acc_dreamer > acc_insomniac:
        improvement_pct = ((acc_dreamer - acc_insomniac) / (1.0 - acc_insomniac) * 100) if acc_insomniac < 1.0 else 0
        print(f"\n[SUCCESS] Sleep helped! Dreamer is more robust.")
        print(f"  Recovery from damage: {improvement_pct:.2f}% of lost accuracy")
    else:
        print(f"\n[FAILURE] Sleep did not help in this run. May need parameter tuning.")
        print(f"  Performance degradation: {((acc_insomniac - acc_dreamer) / acc_insomniac * 100):.2f}%")


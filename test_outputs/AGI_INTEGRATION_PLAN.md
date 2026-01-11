# AGI Integration Plan: Минимальные Замыкания

## Проблемы текущей реализации

1. **World Model** - объявлен, но не используется (нет loss, нет реальных действий)
2. **Internal Goals** - не влияют на policy
3. **Own Concepts** - просто автоэнкодер, не используется в управлении
4. **Autobiographical Memory** - просто лог, не влияет на решения
5. **Self Model** - нет обучающих сигналов
6. **Нет агентности** - нет действий, меняющих мир

## Реализованные изменения

### 1. Action Loop (AttentionAction)
- ✅ Добавлен `AttentionAction` - выбор patch для фокуса (2x2 grid = 4 patches)
- ✅ Действие применяется к изображению: crop + zoom выбранного patch
- ✅ `WorldModel.action_dim = 4` (patches), а не `output_size` (классы)

### 2. World Model с реальными действиями
- ✅ `predict_next(features, action_onehot)` - предсказывает следующее состояние
- ✅ `predict_best_action(features, goal_features)` - предсказывает лучшее действие
- ✅ Возвращает `z_mean, z_logvar` для KL loss

### 3. Goal-conditioned Control
- ✅ `InternalGoals.get_goal_policy_modifier()` - влияет на recursion/replay/temperature
- ✅ Цели теперь влияют на policy через модификаторы

### 4. Memory → Decision
- ✅ `AutobiographicalMemory.get_recall_policy_modifier()` - влияет на policy на основе recall
- ✅ Анализирует исходы похожих эпизодов и модифицирует recursion/replay

### 5. Self-Model Supervision
- ✅ `SelfModel.update_ema_targets()` - обновляет EMA targets для self-supervised обучения
- ✅ `SelfModel.get_targets()` - возвращает targets (capabilities, confidence, weakness)

### 6. Concepts в управлении
- ✅ `OwnConcepts.get_concept_based_routing()` - генерирует routing signal на основе концептов

## Что нужно добавить в Training Loop

### 1. World Model Loss
```python
# В Phase2 training loop:
if agent.use_world_model:
    # Выбираем действие
    action_idx, action_logits = agent.select_action(features, goal_features)
    
    # Применяем действие к изображению
    x_next = agent.apply_action_to_image(data_real, action_idx)
    features_next, _ = agent.shared_backbone(x_next), None
    
    # Предсказываем следующее состояние
    next_features_pred, z_mean, z_logvar, next_z_mean, next_z_logvar = agent.predict_next_state(features, action_idx)
    
    # World Model Loss
    loss_wm_recon = F.mse_loss(next_features_pred, features_next.detach())
    loss_wm_kl = -0.5 * torch.sum(1 + next_z_logvar - next_z_mean.pow(2) - next_z_logvar.exp()) / features.size(0)
    loss_world_model = loss_wm_recon + 0.1 * loss_wm_kl
    
    total_loss = total_loss + 0.1 * loss_world_model
```

### 2. Goal-conditioned Policy
```python
if agent.use_internal_goals:
    goal = agent.generate_internal_goal(features)
    policy_mod = agent.internal_goals.get_goal_policy_modifier(features, goal)
    
    # Модифицируем actions от Complexity Controller
    if actions is not None:
        actions["n_recursions"] = max(1, min(3, int(actions["n_recursions"] + policy_mod[0].mean().item())))
        actions["replay_ratio"] = max(0.1, min(0.4, actions["replay_ratio"] + policy_mod[1].mean().item() * 0.1))
        actions["gate_temperature"] = max(0.7, min(2.0, actions["gate_temperature"] + policy_mod[2].mean().item() * 0.2))
```

### 3. Memory → Decision
```python
if agent.use_autobiographical_memory and len(agent.heads) > 0:
    # Вспоминаем похожие эпизоды при высокой сложности
    if complexity > 0.5:
        similar_memories = agent.recall_similar_experiences(features[0], k=5)
        if similar_memories:
            memory_mod = agent.autobiographical_memory.get_recall_policy_modifier(similar_memories)
            
            # Модифицируем actions
            if actions is not None:
                actions["n_recursions"] = max(1, min(3, int(actions["n_recursions"] + memory_mod["recursion_boost"])))
                actions["replay_ratio"] = max(0.1, min(0.4, actions["replay_ratio"] + memory_mod["replay_boost"]))
    
    # Записываем эпизод
    agent.record_autobiographical_memory(
        step=step,
        state_features=features[0],
        action=action_idx[0] if agent.use_world_model else None,
        outcome={"loss": float(loss_new.item()), "surprise": float(surprise.item()) if surprise else 0.0},
        reward_signal=1.0 - float(loss_new.item()) / 2.0,  # нормализованная награда
        context={"complexity": complexity, "entropy": entropy_test}
    )
```

### 4. Self-Model Supervision
```python
if agent.use_self_model:
    # Вычисляем реальные метрики
    capabilities_real = torch.zeros(len(agent.heads))
    for i, head in enumerate(agent.heads):
        # Используем accuracy по head (упрощённо)
        # В реальности нужно отслеживать accuracy по каждому head
        capabilities_real[i] = 0.8  # placeholder
    
    confidence_real = 1.0 - float(loss_new.item()) / 2.0  # нормализованная уверенность
    weakness_real = min(1.0, (surprise.item() if surprise else 0.0) + pain_value + entropy_test / 2.3)
    
    # Обновляем EMA targets
    agent.self_model.update_ema_targets(capabilities_real, confidence_real, weakness_real)
    
    # Self-Model Loss
    capabilities_pred = agent.self_model.predict_capabilities(features)
    confidence_pred = agent.self_model.estimate_confidence(features)
    weakness_pred = agent.self_model.detect_weakness(features)
    
    targets = agent.self_model.get_targets()
    loss_self_cap = F.mse_loss(capabilities_pred.mean(dim=0), targets["capabilities"].to(features.device))
    loss_self_conf = F.mse_loss(confidence_pred.mean(), torch.tensor(targets["confidence"], device=features.device))
    loss_self_weak = F.mse_loss(weakness_pred.mean(), torch.tensor(targets["weakness"], device=features.device))
    
    loss_self_model = loss_self_cap + loss_self_conf + loss_self_weak
    total_loss = total_loss + 0.05 * loss_self_model
```

### 5. Concepts в управлении
```python
if agent.use_own_concepts:
    concept_activations, concept_importance = agent.extract_own_concepts(features)
    routing_signal = agent.own_concepts.get_concept_based_routing(concept_activations, concept_importance)
    
    # Модифицируем routing gate temperature на основе концептов
    if actions is not None and agent.use_soft_routing:
        concept_temp_mod = routing_signal.mean().item() * 0.3  # [-0.3..0.3]
        actions["gate_temperature"] = max(0.7, min(2.0, actions["gate_temperature"] + concept_temp_mod))
    
    # Concept reconstruction loss
    reconstructed = agent.own_concepts.reconstruct_from_concepts(concept_activations)
    loss_concept_recon = F.mse_loss(reconstructed, features.detach())
    total_loss = total_loss + 0.02 * loss_concept_recon
```

### 6. Complexity Controller с World Model Error
```python
# Вместо surp_approx = 0.5 * entropy_test
if agent.use_world_model and len(agent.heads) > 0:
    # Используем prediction error от World Model как сигнал сложности
    if hasattr(agent, '_last_wm_error'):
        wm_error_signal = agent._last_wm_error
    else:
        wm_error_signal = entropy_test * 0.5  # fallback
    
    complexity = agent.complexity_controller.compute_complexity(
        surprise=wm_error_signal,
        pain=pain_value,
        entropy=entropy_test,
        unknown_rate=unknown_rate_test
    )
else:
    # Старый способ
    surp_approx = 0.5 * entropy_test if entropy_test > 0 else 0.0
    complexity = agent.complexity_controller.compute_complexity(...)
```

## Порядок интеграции

1. **Сначала**: World Model Loss (минимальное замыкание)
2. **Затем**: Goal-conditioned Control
3. **Потом**: Memory → Decision
4. **Далее**: Self-Model Supervision
5. **Наконец**: Concepts в управлении

## Тестирование

После каждого шага проверять:
- Loss не взрывается
- Компоненты действительно влияют на поведение (логи показывают изменения)
- Производительность не деградирует

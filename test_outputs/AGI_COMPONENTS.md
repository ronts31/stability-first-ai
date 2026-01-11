# AGI Components Architecture

## Обзор

Добавлены 5 ключевых компонентов для приближения к AGI:

1. **World Model** - предсказание будущих состояний и причинность
2. **Internal Goals** - внутренние цели, независимые от внешних наград
3. **Own Concepts** - собственные концепты, генерируемые системой
4. **Autobiographical Memory** - автобиографическая память о собственных действиях
5. **Self Model** - явная модель самого себя и своих способностей

## 1. World Model (Модель Мира)

**Назначение:** Предсказывает будущие состояния и причинно-следственные связи.

**Архитектура:**
- Encoder: features → latent space
- Transition: (latent, action) → next_latent
- Decoder: latent → predicted_features
- Causality Head: предсказывает последствия действий

**Использование:**
```python
next_features_pred, causality_pred = agent.use_world_model_prediction(features, action_logits)
```

**Обучение:**
- Reconstruction loss: предсказанные features vs реальные
- KL divergence: регуляризация latent space
- Causality loss: предсказанные последствия vs реальные

## 2. Internal Goals (Внутренние Цели)

**Назначение:** Генерирует собственные цели на основе curiosity, novelty, и внутренней мотивации.

**Архитектура:**
- Goal Generator: features → goal vector
- Goal Evaluator: (features, goal) → achievement_score
- Novelty Detector: features → novelty_signal

**Использование:**
```python
goal = agent.generate_internal_goal(features)
achievement = agent.internal_goals.evaluate_goal_achievement(features, goal)
novelty = agent.internal_goals.compute_novelty(features)
```

**Обучение:**
- Goal generation loss: цели должны быть достижимыми
- Achievement prediction: точность оценки достижения целей
- Novelty signal: корреляция с реальной новизной

## 3. Own Concepts (Собственные Концепты)

**Назначение:** Генерирует и использует внутренние концепты, не заданные извне.

**Архитектура:**
- Concept Encoder: features → concept_activations
- Concept Bank: хранилище концептов (trainable embeddings)
- Concept Decoder: concepts → reconstructed_features
- Importance Net: важность каждого концепта

**Использование:**
```python
concept_activations, concept_importance = agent.extract_own_concepts(features)
reconstructed = agent.own_concepts.reconstruct_from_concepts(concept_activations)
new_concept_found = agent.own_concepts.discover_new_concept(features)
```

**Обучение:**
- Reconstruction loss: восстановление features из концептов
- Concept discovery: автоматическое обнаружение новых концептов
- Importance learning: важность концептов для разных задач

## 4. Autobiographical Memory (Автобиографическая Память)

**Назначение:** Записывает собственные действия, решения и опыт с контекстом.

**Структура записи:**
- step: номер шага
- state: состояние (features)
- action: действие
- outcome: результат действия
- reward: сигнал награды
- context: дополнительный контекст

**Использование:**
```python
agent.record_autobiographical_memory(step, state_features, action, outcome, reward_signal, context)
similar_memories = agent.recall_similar_experiences(query_features, k=10)
stats = agent.autobiographical_memory.get_statistics()
```

**Особенности:**
- Автоматическое ограничение размера (max_memories)
- Поиск похожих эпизодов по cosine similarity
- Сохранение контекста для понимания "почему"

## 5. Self Model (Модель Себя)

**Назначение:** Модель самого себя и своих способностей.

**Компоненты:**
- Capability Predictor: предсказывает способности на разных задачах
- Confidence Estimator: оценивает уверенность в предсказаниях
- Weakness Detector: обнаруживает слабые места
- Self-Awareness: мета-оценка собственного состояния

**Использование:**
```python
capabilities, confidence, weakness = agent.self_assess_capabilities(features)
assessment = agent.self_model.self_assess(features, head_capabilities)
```

**Обучение:**
- Capability prediction: точность предсказания способностей
- Confidence calibration: калибровка уверенности
- Weakness detection: обнаружение реальных слабостей

## Интеграция в Training Loop

Все компоненты интегрированы в `RecursiveAgent` и могут быть включены через флаги:

```python
agent = RecursiveAgent(
    use_curiosity=True,
    use_subjective_time=True,
    use_vae_dreams=True,
    use_world_model=True,          # NEW
    use_internal_goals=True,        # NEW
    use_own_concepts=True,          # NEW
    use_autobiographical_memory=True,  # NEW
    use_self_model=True            # NEW
)
```

## Следующие Шаги

1. **Обучение компонентов:** Добавить loss функции для каждого компонента
2. **Интеграция в decision-making:** Использовать компоненты для принятия решений
3. **Взаимодействие компонентов:** Связать компоненты друг с другом
4. **Эксперименты:** Тестирование на различных задачах

## Примечания

- Все компоненты опциональны и могут быть включены/выключены независимо
- Компоненты используют общий `feature_dim=512` для совместимости
- Автобиографическая память не требует обучения (это структура данных)
- World Model, Internal Goals, Own Concepts, Self Model требуют обучения

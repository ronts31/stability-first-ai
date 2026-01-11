# Быстрый запуск на Runpod B200

## Одна команда для запуска (B200-оптимизированная версия):

```bash
cd /workspace && git clone https://github.com/vitali-sialedchyk/stability-first-ai.git 2>/dev/null || (cd stability-first-ai && git pull) && cd stability-first-ai/test_outputs && pip install -q git+https://github.com/openai/CLIP.git matplotlib numpy pillow torch torchvision && python test_recursive_cifar10_b200.py
```

## Одна команда для запуска (стандартная версия, БЕЗ Complexity Controller):

```bash
cd /workspace && git clone https://github.com/vitali-sialedchyk/stability-first-ai.git 2>/dev/null || (cd stability-first-ai && git pull) && cd stability-first-ai/test_outputs && pip install -q git+https://github.com/openai/CLIP.git matplotlib numpy pillow torch torchvision && python test_recursive_cifar10.py
```

**⚠️ ВАЖНО:** Для использования Complexity Controller (управление временем и рекурсией) используйте `test_recursive_cifar10_b200.py`!

## Или по шагам:

```bash
# 1. Клонируем/обновляем репозиторий
cd /workspace
git clone https://github.com/vitali-sialedchyk/stability-first-ai.git || (cd stability-first-ai && git pull)

# 2. Переходим в папку с тестом
cd stability-first-ai/test_outputs

# 3. Устанавливаем зависимости
pip install git+https://github.com/openai/CLIP.git matplotlib numpy pillow

# 4. Запускаем тест
python test_recursive_cifar10.py
```

## Что интегрировано (максимальная интеграция всех механизмов):

### В `test_recursive_cifar10_b200.py` (РЕКОМЕНДУЕТСЯ):
- ✅ **Complexity Controller** - единый метаконтроллер для управления временем и рекурсией
- ✅ **Внутренняя рекурсия** - динамический compute loop (Pass0..PassN)
- ✅ **Complexity Budget Accounting** - закон сохранения сложности
- ✅ **Gate Temperature Control** - управление routing через сложность
- ✅ **Memory Scheduler** - управление replay через сложность
- ✅ **Subjective Time Critic** - автоматическая регуляция пластичности на основе Surprise
- ✅ **Fractal Time** - разные уровни защиты для разных слоев backbone
- ✅ **Adaptive Time/Pain** - динамический lambda на основе конфликта градиентов
- ✅ **VAE Dream Generator** - реалистичные сны вместо белого шума
- ✅ **Replay Buffer** - защита памяти через replay loss
- ✅ **Head-only Recovery** - восстановление забытых задач
- ✅ **Lazarus v3** - Consistency Anchor + Stability Loss + Entropy Floor
- ✅ **CNN архитектура** вместо MLP
- ✅ **Аугментации данных** (RandomCrop, RandomHorizontalFlip)
- ✅ **Батчевый CLIP teacher** (быстрее и эффективнее)
- ✅ **Улучшенные промпты** ("a photo of..." вместо "a...")

### В `test_recursive_cifar10.py` (старая версия):
- ✅ Все вышеперечисленное, **КРОМЕ Complexity Controller**

## Ожидаемые результаты:

- Животные: 46% → **65-75%** (ожидаемый прирост с полной интеграцией)
- Cat/Dog: особенно заметное улучшение
- Общая точность: **+15-25%**
- Стабильность памяти: **+10-15%** (благодаря Subjective Time + Replay)

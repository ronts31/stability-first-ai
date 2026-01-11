# Быстрый запуск на Runpod B200

## Одна команда для запуска:

```bash
cd /workspace && git clone https://github.com/vitali-sialedchyk/stability-first-ai.git 2>/dev/null || (cd stability-first-ai && git pull) && cd stability-first-ai/test_outputs && pip install -q git+https://github.com/openai/CLIP.git matplotlib numpy pillow && python test_recursive_cifar10.py
```

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

## Что изменилось:

- ✅ **CNN архитектура** вместо MLP (ожидаемый прирост +10-20%)
- ✅ **Аугментации данных** (RandomCrop, RandomHorizontalFlip)
- ✅ **Батчевый CLIP teacher** (быстрее и эффективнее)
- ✅ **Улучшенные промпты** ("a photo of..." вместо "a...")

## Ожидаемые результаты:

- Животные: 46% → **60-70%** (ожидаемый прирост)
- Cat/Dog: особенно заметное улучшение
- Общая точность: **+10-15%**

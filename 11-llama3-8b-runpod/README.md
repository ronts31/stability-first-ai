# LLaMA‑3‑8B: Полный тестовый suite для Runpod (B200, PyTorch 2.8+)

Этот модуль — **полноценный suite** для проверки **всех теорий TemporalLoRA** на большой модели **LLaMA‑3‑8B** на GPU **B200** в поде Runpod.

## Образ Runpod

- `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0)

## Что входит

### Основные компоненты:

1. **`temporal_lora_llama3.py`** — адаптированная версия TemporalLoRAModel для LLaMA-3-8B:
   - Загрузка LLaMA-3-8B с поддержкой BF16 и device_map="auto"
   - LoRA адаптеры для разных временных эпох
   - Time Mixer для динамического переключения между доменами
   - Обучение адаптеров с Active Sleep
   - Калибровка Time Mixer

2. **`test_hysteresis_llama3.py`** — тесты гистерезиса (кристаллизация времени):
   - Тест A→B→A (чистое переключение)
   - Тест A→Mix→A (смешанный сегмент)
   - Метрики: switch-lag, return-gap, switching asymmetry
   - Визуализация результатов

3. **`test_fatigue_llama3.py`** — тесты fatigue (глубокая кристаллизация):
   - Sweep тест с разными длинами Python блоков
   - Метрики: deep_crystallization_ratio, relax_time_0.99, tail_area_32
   - Корреляционный анализ
   - Визуализация результатов

4. **`main_full_suite.py`** — **ГЛАВНЫЙ СКРИПТ** для запуска всего:
   - Проверка среды (PyTorch 2.8+, CUDA, BF16)
   - Загрузка модели
   - Обучение адаптеров
   - Калибровка Time Mixer
   - Запуск всех тестов
   - Сохранение результатов

5. **`llama3_8b_full_suite.py`** — базовые технические проверки (опционально)

## Установка зависимостей

В поде Runpod:

```bash
cd /workspace  # или путь к корню репозитория
pip install -r 11-llama3-8b-runpod/requirements.txt
```

> **Примечание:** для LLaMA‑3‑8B нужны права/токен Hugging Face. Задайте переменную `HF_TOKEN` или выполните `huggingface-cli login`.

## Запуск полного suite

### Базовый запуск (все тесты):

```bash
cd /workspace
python 11-llama3-8b-runpod/main_full_suite.py
```

### С параметрами:

```bash
# Использовать другую модель
MODEL_NAME="meta-llama/Meta-Llama-3-8B" \
python 11-llama3-8b-runpod/main_full_suite.py

# Быстрый режим (меньше данных, меньше эпох)
FAST_MODE="True" \
python 11-llama3-8b-runpod/main_full_suite.py
```

## Что проверяется

### 1. Технические проверки:
- ✅ Версия PyTorch ≥ 2.8.0
- ✅ CUDA доступность и устройство (B200)
- ✅ Поддержка BF16
- ✅ Память GPU
- ✅ Загрузка модели LLaMA-3-8B

### 2. Обучение:
- ✅ Обучение Shakespeare адаптера
- ✅ Обучение Python адаптера с Active Sleep
- ✅ Калибровка Time Mixer (контрастная калибровка)

### 3. Тесты гистерезиса (Time Crystallization):
- ✅ **Switch-lag**: инерция переключения между доменами
- ✅ **Return-gap**: память траектории (насколько второй "A" отличается от первого)
- ✅ **Switching asymmetry**: асимметрия переключения A→B vs B→A
- ✅ **Mix segment analysis**: анализ смешанных сегментов

### 4. Тесты fatigue (Deep Crystallization):
- ✅ **Deep crystallization ratio**: доля токенов с весом > 0.95
- ✅ **Relax-time 0.99**: время релаксации до веса 0.99
- ✅ **Tail area 32**: площадь "неопределённости" в первых 32 токенах
- ✅ **Корреляционный анализ**: зависимость от длины пребывания в домене

## Результаты

После выполнения suite создаются файлы:

- **`temporal_lora_checkpoint_llama3.pt`** — чекпоинт обученной модели
- **`hysteresis_results_llama3.json`** — результаты тестов гистерезиса
- **`hysteresis_analysis_llama3.png`** — визуализация гистерезиса
- **`fatigue_results_llama3.json`** — результаты тестов fatigue
- **`fatigue_analysis_llama3.png`** — визуализация fatigue
- **`full_suite_results_llama3.json`** — **полная сводка всех результатов**

## Структура результатов

### `full_suite_results_llama3.json` содержит:

```json
{
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "torch_version": "2.8.0+cu128",
  "timing": {
    "model_load_sec": 45.2,
    "phase3_training_sec": 120.5,
    "phase4_calibration_sec": 30.1,
    "phase5_hysteresis_sec": 15.3,
    "phase6_fatigue_sec": 45.7,
    "total_sec": 256.8
  },
  "hysteresis": {
    "test_aba": {
      "switch_lag_AB": 0,
      "switch_lag_BA": 9,
      "switching_asymmetry": 9,
      "return_gap_cosine_distance": 0.0000
    },
    "test_amixba": {
      "switch_lag_mixA": 16,
      "return_gap_cosine_distance": 0.1909,
      "avg_mix_entropy": 0.679
    }
  },
  "fatigue": {
    "correlations": {
      "deep_crystallization": 0.7381,
      "switch_lag": 0.1234
    },
    "summary": {
      "num_tests": 5,
      "avg_deep_crystallization": 0.65
    }
  }
}
```

## Интерпретация результатов

### Гистерезис (Time Crystallization):

- **Switch-lag A→B = 0**: быстрое переключение при входе в домен
- **Switch-lag B→A = 9**: инерция при выходе из домена (кристаллизация)
- **Return-gap > 0.1**: память траектории существует (второй "A" отличается от первого)
- **Return-gap после Mix > Return-gap после чистого переключения**: смешанные сегменты создают более сильную память

### Fatigue (Deep Crystallization):

- **Deep crystallization correlation > 0.7**: подтверждение гипотезы — чем дольше пребывание в домене, тем сильнее кристаллизация
- **Switch-lag correlation низкая**: переключение не усложняется с длиной (на пороге 0.9)
- **Tail area 32 растёт**: увеличивается "неопределённость" при возврате после длительного пребывания

## Научный вклад

Этот suite проверяет следующие гипотезы:

1. ✅ **Router имеет временную динамику** — не просто классификатор
2. ✅ **Trajectory memory существует** — роутер "помнит" путь через смешанные сегменты
3. ✅ **Domain confidence saturates** — длительное пребывание → больше экстремальных весов
4. ✅ **Switching asymmetry** — инерция при выходе из домена
5. ✅ **Deep crystallization** — насыщение уверенности домена с увеличением времени пребывания

## Интеграция с репозиторием

Папка `11-llama3-8b-runpod` полностью автономна и не влияет на существующие эксперименты с GPT-2 (`02-temporal-lora-gpt2`).

## Troubleshooting

### Ошибка загрузки модели:
- Проверьте `HF_TOKEN` или выполните `huggingface-cli login`
- Убедитесь, что модель доступна на Hugging Face

### Out of Memory:
- Уменьшите `batch_size` в коде
- Используйте `FAST_MODE="True"`
- Уменьшите `max_length` в тестах

### Медленная работа:
- Используйте `FAST_MODE="True"` для быстрых проверок
- Уменьшите количество эпох обучения
- Уменьшите количество тестовых длин в fatigue sweep

## Лицензия

См. основной LICENSE репозитория.

# Инструкция по выгрузке результатов из Runpod

## Вариант 1: Через SSH (SCP) - Рекомендуется

### Настройка SSH ключа (если еще не настроен)

1. Сгенерируйте SSH ключ (если еще нет):
```bash
ssh-keygen -t ed25519 -C "info@agdgroup.pl"
```

2. Скопируйте публичный ключ в Runpod:
   - Откройте https://console.runpod.io/pods?id=y68lflk0zivqns
   - Вставьте публичный ключ в настройках SSH
   - Публичный ключ находится в: `~/.ssh/id_ed25519.pub`

### Скачивание файлов через SCP

```bash
# Перейдите в корень репозитория
cd /path/to/stability-first-ai

# Создайте директорию для результатов
mkdir -p 11-llama3-8b-runpod/results

# Скачайте JSON результаты (маленькие файлы)
scp -P 11773 root@66.92.198.130:/workspace/stability-first-ai/11-llama3-8b-runpod/full_suite_results_llama3.json \
    11-llama3-8b-runpod/results/

scp -P 11773 root@66.92.198.130:/workspace/stability-first-ai/11-llama3-8b-runpod/hysteresis_results_llama3.json \
    11-llama3-8b-runpod/results/

scp -P 11773 root@66.92.198.130:/workspace/stability-first-ai/11-llama3-8b-runpod/fatigue_results_llama3.json \
    11-llama3-8b-runpod/results/

# Скачайте изображения
scp -P 11773 root@66.92.198.130:/workspace/stability-first-ai/11-llama3-8b-runpod/fatigue_analysis_llama3.png \
    11-llama3-8b-runpod/results/

scp -P 11773 root@66.92.198.130:/workspace/stability-first-ai/11-llama3-8b-runpod/hysteresis_analysis_llama3.png \
    11-llama3-8b-runpod/results/

# Опционально: скачайте чекпоинт модели (большой файл, ~100MB+)
scp -P 11773 root@66.92.198.130:/workspace/stability-first-ai/11-llama3-8b-runpod/temporal_lora_checkpoint_llama3.pt \
    11-llama3-8b-runpod/results/
```

## Вариант 2: Через веб-терминал Runpod

1. Откройте веб-терминал в Runpod: https://console.runpod.io/pods?id=y68lflk0zivqns
2. **Сначала найдите файлы результатов:**

```bash
# Перейдите в корень репозитория
cd /workspace/stability-first-ai

# Найдите все файлы результатов
find . -name "full_suite_results_llama3.json" -o \
       -name "hysteresis_results_llama3.json" -o \
       -name "fatigue_results_llama3.json" -o \
       -name "fatigue_analysis_llama3.png" -o \
       -name "hysteresis_analysis_llama3.png" -o \
       -name "temporal_lora_checkpoint_llama3.pt"

# Или просто проверьте корневую директорию
ls -lh *.json *.png *.pt 2>/dev/null

# Или проверьте в поддиректории
ls -lh 11-llama3-8b-runpod/*.json 11-llama3-8b-runpod/*.png 11-llama3-8b-runpod/*.pt 2>/dev/null
```

3. **Создайте архив с результатами:**

```bash
# Если файлы в корне репозитория
cd /workspace/stability-first-ai
tar -czf results.tar.gz \
    full_suite_results_llama3.json \
    hysteresis_results_llama3.json \
    fatigue_results_llama3.json \
    fatigue_analysis_llama3.png \
    hysteresis_analysis_llama3.png 2>/dev/null

# Или если файлы в поддиректории
cd /workspace/stability-first-ai/11-llama3-8b-runpod
tar -czf ../results.tar.gz \
    full_suite_results_llama3.json \
    hysteresis_results_llama3.json \
    fatigue_results_llama3.json \
    fatigue_analysis_llama3.png \
    hysteresis_analysis_llama3.png 2>/dev/null

# Проверьте размер
ls -lh results.tar.gz
```

4. **Скопируйте содержимое JSON файлов** (если архив не работает):

```bash
# Покажите содержимое файлов для копирования
cat full_suite_results_llama3.json
# Скопируйте весь вывод и создайте файл локально

# Или используйте base64 для передачи бинарных файлов
base64 fatigue_analysis_llama3.png
# Скопируйте вывод, затем локально: base64 -d > fatigue_analysis_llama3.png
```

## Вариант 3: Через HTTP сервис (если настроен)

Если в Runpod настроен HTTP сервис на порту 8888:

```bash
# На Runpod pod создайте простой HTTP сервер
cd /workspace/stability-first-ai/11-llama3-8b-runpod
python3 -m http.server 8888
```

Затем скачайте файлы через браузер:
- http://your-pod-ip:8888/full_suite_results_llama3.json
- http://your-pod-ip:8888/hysteresis_results_llama3.json
- и т.д.

## После скачивания - добавление в Git

```bash
# Перейдите в корень репозитория
cd /path/to/stability-first-ai

# Добавьте результаты в git
git add 11-llama3-8b-runpod/results/*.json
git add 11-llama3-8b-runpod/results/*.png

# Опционально: добавьте чекпоинт (если скачали)
# git add 11-llama3-8b-runpod/results/temporal_lora_checkpoint_llama3.pt

# Закоммитьте
git commit -m "Add LLaMA-3-8B test results from Runpod

- Full suite results (Mistral-7B-Instruct)
- Hysteresis tests results
- Fatigue tests results
- Visualization plots"

# Запушьте в репозиторий
git push origin main
```

## Список файлов для скачивания

**Обязательные (маленькие файлы):**
- `full_suite_results_llama3.json` - полная сводка
- `hysteresis_results_llama3.json` - результаты гистерезиса
- `fatigue_results_llama3.json` - результаты fatigue
- `fatigue_analysis_llama3.png` - график fatigue
- `hysteresis_analysis_llama3.png` - график гистерезиса (если создан)

**Опциональные (большие файлы):**
- `temporal_lora_checkpoint_llama3.pt` - чекпоинт модели (~100MB+)

## Примечание

Чекпоинт модели (`temporal_lora_checkpoint_llama3.pt`) может быть очень большим (100MB+). 
Рекомендуется:
- Либо использовать Git LFS для больших файлов
- Либо не коммитить чекпоинт, а хранить его отдельно
- Либо добавить в `.gitignore`


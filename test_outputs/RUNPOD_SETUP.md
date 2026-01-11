# Запуск на Runpod B200

## Шаг 1: Подключение к Pod

После создания pod на Runpod, подключитесь через:
- **Jupyter Lab** (рекомендуется) или
- **SSH** (если настроен)

## Шаг 2: Установка зависимостей

В терминале Jupyter или SSH выполните:

```bash
# Переходим в рабочую директорию
cd /workspace

# Клонируем репозиторий (если еще не склонирован)
# git clone <your-repo-url>
# cd <repo-name>

# Устанавливаем зависимости
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/openai/CLIP.git
pip install matplotlib numpy pillow

# Проверяем версию PyTorch (должна быть >= 2.8.0)
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## Шаг 3: Копирование файлов

Если файлы еще не на сервере, скопируйте:
- `test_outputs/test_recursive_cifar10.py`
- Структуру проекта (если нужна)

## Шаг 4: Запуск теста

```bash
cd /workspace
cd test_outputs  # или путь к вашей директории

# Запускаем тест
python test_recursive_cifar10.py
```

## Шаг 5: Мониторинг

Тест будет выводить в реальном времени:
- Информацию о GPU (B200)
- Прогресс обучения
- Метрики (Loss, Accuracy, UnknownRate)
- Логи CLIP и конфликтов
- Статистику по слоям

## Ожидаемый вывод

```
Running on: cuda
CUDA device: NVIDIA B200
GPU memory: XX.XX GB
BF16 supported: True

[CURIOSITY] Loading World Knowledge (CLIP)...
[CURIOSITY] CLIP loaded successfully!
[INFO] Curiosity Module (CLIP) enabled - agent can query world knowledge!

--- PHASE 1: URBAN ENVIRONMENT (Learning Machines: [0, 1, 8, 9]) ---
Step 0: Loss 5.08 | Acc Machines: 6.5%
...
```

## Примечания

- **B200 требует PyTorch 2.8+** (уже установлен в образе)
- CLIP будет работать быстрее на GPU
- Тест может занять 10-30 минут в зависимости от данных
- Результаты сохраняются в `cifar10_drone_result.png`

## Troubleshooting

Если CLIP не устанавливается:
```bash
pip install --upgrade pip
pip install git+https://github.com/openai/CLIP.git --no-cache-dir
```

Если проблемы с CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

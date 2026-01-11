# Инструкции по клонированию на Runpod

## На Runpod выполните:

```bash
cd /workspace

# Клонируем репозиторий
git clone https://github.com/vitali-sialedchyk/stability-first-ai.git

# Переходим в репозиторий
cd stability-first-ai

# Переходим в папку с тестом
cd test_outputs

# Устанавливаем зависимости (если еще не установлены)
pip install git+https://github.com/openai/CLIP.git
pip install matplotlib numpy pillow

# Запускаем тест
python test_recursive_cifar10.py
```

## Или одной командой:

```bash
cd /workspace && git clone https://github.com/vitali-sialedchyk/stability-first-ai.git && cd stability-first-ai/test_outputs && pip install git+https://github.com/openai/CLIP.git matplotlib numpy pillow && python test_recursive_cifar10.py
```

## Если файл test_recursive_cifar10.py не в репозитории:

Файл может быть в `.gitignore`. В этом случае:

1. Скопируйте содержимое файла `test_outputs/test_recursive_cifar10.py` 
2. Создайте файл на Runpod:
```bash
cd /workspace/stability-first-ai
mkdir -p test_outputs
cd test_outputs
nano test_recursive_cifar10.py
# Вставьте код и сохраните (Ctrl+O, Enter, Ctrl+X)
```

3. Запустите тест:
```bash
python test_recursive_cifar10.py
```

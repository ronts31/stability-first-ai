#!/bin/bash
# Быстрый запуск на Runpod B200

echo "=== Проверка окружения ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== Установка зависимостей ==="
pip install -q git+https://github.com/openai/CLIP.git
pip install -q matplotlib numpy pillow

echo ""
echo "=== Запуск теста ==="
cd /workspace/test_outputs || cd test_outputs
python test_recursive_cifar10.py

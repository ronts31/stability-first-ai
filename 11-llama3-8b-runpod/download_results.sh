#!/bin/bash
# Скрипт для скачивания результатов из Runpod pod
# Использование: ./download_results.sh

# Настройки
POD_IP="66.92.198.130"
POD_PORT="11773"
POD_USER="root"
REMOTE_DIR="/workspace/stability-first-ai/11-llama3-8b-runpod"
LOCAL_DIR="./11-llama3-8b-runpod/results"

# Создаем локальную директорию для результатов
mkdir -p "$LOCAL_DIR"

echo "Скачивание результатов из Runpod pod..."
echo "Pod: $POD_USER@$POD_IP:$POD_PORT"
echo ""

# Список файлов для скачивания
FILES=(
    "full_suite_results_llama3.json"
    "hysteresis_results_llama3.json"
    "fatigue_results_llama3.json"
    "fatigue_analysis_llama3.png"
    "hysteresis_analysis_llama3.png"
)

# Скачиваем каждый файл
for file in "${FILES[@]}"; do
    echo "Скачивание $file..."
    scp -P $POD_PORT -o StrictHostKeyChecking=no \
        $POD_USER@$POD_IP:$REMOTE_DIR/$file \
        "$LOCAL_DIR/$file" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "  ✓ $file скачан"
    else
        echo "  ✗ Ошибка при скачивании $file"
    fi
done

# Чекпоинт модели (большой файл, опционально)
echo ""
read -p "Скачать чекпоинт модели? (temporal_lora_checkpoint_llama3.pt) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Скачивание чекпоинта (это может занять время)..."
    scp -P $POD_PORT -o StrictHostKeyChecking=no \
        $POD_USER@$POD_IP:$REMOTE_DIR/temporal_lora_checkpoint_llama3.pt \
        "$LOCAL_DIR/temporal_lora_checkpoint_llama3.pt" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Чекпоинт скачан"
    else
        echo "  ✗ Ошибка при скачивании чекпоинта"
    fi
fi

echo ""
echo "Готово! Файлы находятся в: $LOCAL_DIR"


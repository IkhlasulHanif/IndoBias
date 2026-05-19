#!/bin/bash

# IndoBias - Run ALL groups with official Qwen family models

echo ""
echo "IndoBias: Run ALL Groups (Qwen)"
echo "================================"
echo ""

# Check Python
python3 --version || exit 1
echo ""

# Show groups
echo "Groups to process:"
if [ -d "data/groups" ]; then
    for csv in data/groups/*.csv; do
        [ -f "$csv" ] && basename "$csv" .csv | sed 's/^/  • /'
    done
else
    echo "Error: data/groups not found"
    exit 1
fi

# Official checkpoints: Qwen3 has 8B, Qwen3.5 has 9B
QWEN_MODELS=(
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3.5-9B"
)

echo ""
echo "Models:"
for model in "${QWEN_MODELS[@]}"; do
    echo "  • $model"
done
read -p "Process ALL groups? [y/N]: " confirm
[[ ! $confirm =~ ^[yY]$ ]] && echo "Cancelled" && exit 0

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run pipeline
echo ""
echo "Starting pipeline..."
echo ""

for model in "${QWEN_MODELS[@]}"; do
    echo "Running model: $model"
    python3 main.py --model-id "$model"
    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ Pipeline failed for $model"
        exit 1
    fi
done

echo ""
echo "✓ Pipeline completed!"
if [ -d "output/models" ]; then
    echo "Recent result directories:"
    ls -t output/models | head -2 | sed 's/^/  • output\/models\//'
fi

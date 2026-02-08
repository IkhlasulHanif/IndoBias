#!/bin/bash

# IndoBias - Run ALL groups pipeline

echo ""
echo "IndoBias: Run ALL Groups"
echo "========================="
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

# Confirm
echo ""
echo "Model: Qwen/Qwen3-8B"
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

python3 main.py --model-id Qwen/Qwen3-8B

# Done
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Pipeline completed!"
    if [ -d "output/models" ]; then
        latest=$(ls -t output/models | head -1)
        echo "Results: output/models/$latest"
    fi
else
    echo ""
    echo "✗ Pipeline failed"
    exit 1
fi

#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_RESULTS_DIR="$REPO_ROOT/results/raw"

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
if [ -d "$SCRIPT_DIR/data/groups" ]; then
    for csv in "$SCRIPT_DIR"/data/groups/*.csv; do
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
if [ -d "$REPO_ROOT/.venv" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Run prompt generation pipeline from the repo root so package imports resolve.
echo ""
echo "Starting pipeline..."
echo ""

cd "$REPO_ROOT"

for model in "${QWEN_MODELS[@]}"; do
    echo "Running model: $model"
    python3 -m indobias_qa.commands generate --model-name "$model"
done

echo ""
echo "✓ Pipeline completed!"
if [ -d "$RAW_RESULTS_DIR" ]; then
    echo "Results written under:"
    echo "  • $RAW_RESULTS_DIR"
fi

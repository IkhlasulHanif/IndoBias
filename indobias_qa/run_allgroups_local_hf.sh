#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_RESULTS_DIR="$REPO_ROOT/results/raw"

echo ""
echo "IndoBias: Run ALL Groups (Local HF Models)"
echo "================================"
echo ""

python3 --version || exit 1
echo ""

echo "Groups to process:"
if [ -d "$SCRIPT_DIR/data/groups" ]; then
    for csv in "$SCRIPT_DIR"/data/groups/*.csv; do
        [ -f "$csv" ] && basename "$csv" .csv | sed 's/^/  • /'
    done
else
    echo "Error: data/groups not found"
    exit 1
fi

LOCAL_MODELS=(
    "aisingapore/Gemma-SEA-LION-v3-9B-IT"
    "aisingapore/Apertus-SEA-LION-v4-8B-IT"
)

echo ""
echo "Models:"
for model in "${LOCAL_MODELS[@]}"; do
    echo "  • $model"
done
read -p "Process ALL groups? [y/N]: " confirm
[[ ! $confirm =~ ^[yY]$ ]] && echo "Cancelled" && exit 0

if [ -d "$REPO_ROOT/.venv" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

echo ""
echo "Starting pipeline..."
echo ""

cd "$REPO_ROOT"

for model in "${LOCAL_MODELS[@]}"; do
    echo "Running model: $model"
    python3 -m indobias_qa.model_runner --model-name "$model" --max-new-tokens 128 --batch-size 3
done

echo ""
echo "✓ Pipeline completed!"
if [ -d "$RAW_RESULTS_DIR" ]; then
    echo "Results written under:"
    echo "  • $RAW_RESULTS_DIR"
fi

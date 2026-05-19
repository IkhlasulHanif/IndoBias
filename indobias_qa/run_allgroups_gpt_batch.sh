#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo ""
echo "IndoBias: OpenAI Batch ALL Groups"
echo "================================="
echo ""

python3 --version || exit 1
echo ""

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY is not set."
    exit 1
fi

GPT_MODELS=(
    "gpt-4.1-mini"
    "gpt-5-mini"
)

echo "Models:"
for model in "${GPT_MODELS[@]}"; do
    echo "  • $model"
done
echo ""
echo "This prepares requests and submits OpenAI Batch jobs."
read -p "Continue? [y/N]: " confirm
[[ ! $confirm =~ ^[yY]$ ]] && echo "Cancelled" && exit 0

if [ -d "$REPO_ROOT/.venv" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

cd "$REPO_ROOT"

python3 -m indobias_qa.commands openai-batch-prepare-submit $(printf -- '--model %q ' "${GPT_MODELS[@]}")

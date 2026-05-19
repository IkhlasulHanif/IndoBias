#!/bin/bash

# IndoBias Test Pipeline - Quick testing script

echo ""
echo "IndoBias Test Pipeline"
echo "======================"
echo ""

# Check Python
python3 --version || exit 1
echo ""

# Menu
echo "Choose test:"
echo "1) Quick (ethnicity_indonesia + gender)"
echo "2) Medium (ethnicity + gender + religion)"
echo "3) Custom groups"
echo "4) List groups"
echo "5) Exit"
echo ""
read -p "Choice [1-5]: " choice

# Set groups and model based on choice
case $choice in
    1)
        MY_GROUPS="ethnicity_indonesia gender"
        QWEN_MODELS=(
            "Qwen/Qwen3-8B"
            "Qwen/Qwen3.5-9B"
        )
        ;;
    2)
        MY_GROUPS="ethnicity_indonesia gender religion"
        QWEN_MODELS=(
            "Qwen/Qwen3-8B"
            "Qwen/Qwen3.5-9B"
        )
        ;;
    3)
        python3 main.py --list-groups
        echo ""
        read -p "Enter groups (space-separated): " MY_GROUPS
        QWEN_MODELS=(
            "Qwen/Qwen3-8B"
            "Qwen/Qwen3.5-9B"
        )
        ;;
    4)
        python3 main.py --list-groups
        exit 0
        ;;
    5)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run pipeline
echo ""
echo "Running: Groups=$MY_GROUPS"
echo "Models:"
for model in "${QWEN_MODELS[@]}"; do
    echo "  • $model"
done
echo ""

for model in "${QWEN_MODELS[@]}"; do
    echo "Running model: $model"
    python3 main.py --groups ${MY_GROUPS} --model-id "$model"
    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ Test failed for $model"
        exit 1
    fi
done

# Done
echo ""
echo "✓ Test completed!"
if [ -d "output/models" ]; then
    echo "Recent result directories:"
    ls -t output/models | head -2 | sed 's/^/  • output\/models\//'
fi

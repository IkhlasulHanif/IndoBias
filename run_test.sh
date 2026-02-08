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
        MY_MODEL="Qwen/Qwen3-0.6B"
        ;;
    2)
        MY_GROUPS="ethnicity_indonesia gender religion"
        MY_MODEL="Qwen/Qwen3-0.6B"
        ;;
    3)
        python3 main.py --list-groups
        echo ""
        read -p "Enter groups (space-separated): " MY_GROUPS
        MY_MODEL="Qwen/Qwen3-0.6B"
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
echo "Running: Groups=$MY_GROUPS, Model=$MY_MODEL"
echo ""

python3 main.py --groups ${MY_GROUPS} --model-id "${MY_MODEL}"

# Done
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Test completed!"
    if [ -d "output/models" ]; then
        latest=$(ls -t output/models | head -1)
        echo "Results: output/models/$latest"
    fi
else
    echo ""
    echo "✗ Test failed"
    exit 1
fi

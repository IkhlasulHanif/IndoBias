# Calculate PPL
To install dependencies, use:
```
pip install -r requirements.txt
```

To run, use:
```
python calculate_ppl.py
```

All configs are available on `config.py`

# Pretraining
To install dependencies, use:
```
uv pip install -r requirements.txt
```

To run, use:
```
python pretraining.py --model [MODEL_NAME] \
                      --dataset1 [DATASET1_NAME] \
                      --variant1 [VARIANT1] \
                      --dataset2 [DATASET2_NAME] \
                      --variant2 [VARIANT2] \
                      --dataset3 [DATASET3_NAME] \
                      --variant3 [VARIANT3]
```

Example:
```
python pretraining.py --model "indolem/indobert-base-uncased" \
                      --dataset1 "wikimedia/wikipedia" \
                      --variant1 "20231101.id" \
                      --dataset2 "wikimedia/wikipedia" \
                      --variant2 "20231101.jv" \
                      --dataset3 "wikimedia/wikipedia" \
                      --variant3 "20231101.su"
```
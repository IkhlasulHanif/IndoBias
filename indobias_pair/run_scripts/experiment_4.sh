python pretraining.py --model "indolem/indobert-base-uncased" \
                      --dataset1 "wikimedia/wikipedia" \
                      --variant1 "20231101.id" \
                      --dataset2 "wikimedia/wikipedia" \
                      --variant2 "20231101.jv" \
                      --dataset3 "wikimedia/wikipedia" \
                      --variant3 "20231101.su" \
                      --prob1 8 \
                      --prob2 1 \
                      --prob3 1
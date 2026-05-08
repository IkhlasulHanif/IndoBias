python pretraining.py --model "indolem/indobert-base-uncased" \
                      --dataset1 "SEACrowd/cc100" \
                      --variant1 "cc100_ind_source" \
                      --dataset2 "SEACrowd/cc100" \
                      --variant2 "cc100_jav_source" \
                      --dataset3 "SEACrowd/cc100" \
                      --variant3 "cc100_sun_source" \
                      --prob1 8 \
                      --prob2 1 \
                      --prob3 1
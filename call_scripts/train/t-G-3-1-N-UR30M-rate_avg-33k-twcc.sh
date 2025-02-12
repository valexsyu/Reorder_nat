#!/bin/bash
# source $HOME/.bashrc 
# conda activate base
source call_scripts/train/pair_experiment.sh
experiment=t-G-3-1-N-UR30M-rate_avg-33k

pair_experiment_wmt16_8_2048_rate_avg_33k_twcc $experiment       

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/generate_nat.sh -b 10 --twcc --data-subset test --ck-types top --avg-speed 1 \
                    -e $experiment   \

            
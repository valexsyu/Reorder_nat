#!/bin/bash
# source $HOME/.bashrc 
# conda activate base
source call_scripts/train/pair_experiment.sh
experiment=m-B-3-1-N-UR30M-rate_avg-33k-w1

pair_experiment_iwslt14_8_1536_rate_avg_33k_w1_twcc $experiment       

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/generate_nat.sh -b 10 --twcc --data-subset test --ck-types top --avg-speed 1 \
                    -e $experiment   \

            
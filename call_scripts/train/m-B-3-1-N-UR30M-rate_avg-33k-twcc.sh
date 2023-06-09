#!/bin/bash
# source $HOME/.bashrc 
# conda activate base
source call_scripts/train/pair_experiment.sh
experiment=m-B-3-1-N-UR30M-rate_avg-33k

pair_experiment_iwslt14_4_2048_rate_avg_33k_twcc $experiment       

CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/generate_nat.sh -b 1 --twcc --data-subset test --ck-types top --avg-speed 1 \
                    -e $experiment   \

            
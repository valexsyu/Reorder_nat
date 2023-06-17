#!/bin/bash
# source $HOME/.bashrc 
# conda activate base
source call_scripts/train/pair_experiment.sh
experiment=s-F-3-1-N-UR30M-rate_sel-33k

pair_experiment_wmt14_8_1638_rate_sel_33k_twcc $experiment       

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/generate_nat.sh -b 10 --twcc --data-subset test --ck-types top --avg-speed 1 \
                    -e $experiment   \

            
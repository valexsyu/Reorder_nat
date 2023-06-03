#!/bin/bash
# source $HOME/.bashrc 
# conda activate base
source call_scripts/train/pair_experiment.sh
experiment=a-6-3-1-N-UF30T

pair_experiment_wmt14_4_3276_100k_twcc $experiment       

CUDA_VISIBLE_DEVICES=0,1,2,3 bash call_scripts/generate_nat.sh -b 1 --twcc --data-subset test --ck-types top --avg-speed 1 --no-atten-mask \
                    -e $experiment   \

            
#!/bin/bash
# source $HOME/.bashrc 
# conda activate base
source call_scripts/train/pair_experiment.sh
experiment=3-5-4-1-H12-UR30M

pair_experiment_iwslt14_2_3072_50k_twcc $experiment       

CUDA_VISIBLE_DEVICES=0,1 bash call_scripts/generate_nat.sh -b 1 --twcc --data-subset test --ck-types top --avg-speed 1 \
                    -e $experiment   \

            
#!/bin/bash
# source $HOME/.bashrc 
# conda activate base
source call_scripts/train/pair_experiment.sh
experiment=2-2-3-1-H12-UR40M-50k-wo_softmax

pair_experiment_iwslt14_8_1536_50k_twcc $experiment       

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/generate_nat.sh -b 10 --twcc --data-subset test --ck-types top --avg-speed 1 \
#                     -e $experiment   \
#!/bin/bash
# source $HOME/.bashrc 
# conda activate base
source call_scripts/train/pair_experiment.sh
experiment=r-E-3-1-N-UR20M-100k_300k


mkdir checkpoints/r-E-3-1-N-UR20M-100k_300k
cp -r checkpoints/r-E-3-1-N-UR20M/checkpoint_last.pt checkpoints/r-E-3-1-N-UR20M-100k_300k/
pair_experiment_wmt14_8_4095_100k_300k_twcc $experiment        

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/generate_nat.sh -b 10 --twcc --data-subset test --ck-types top --avg-speed 1 \
                    -e $experiment   \

            
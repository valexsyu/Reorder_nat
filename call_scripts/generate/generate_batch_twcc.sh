# source $HOME/.bashrc 
# conda activate base
CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat_diffbatch.sh -b 50 --data-subset test-valid \
    --ck-types top --no-atten-mask --avg-ck-turnoff --twcc --load-exist-bleu \
-e 2-2-1-1-H12-UR15M \
-e 2-2-1-1-H12-UR20M \
-e 2-2-1-1-H12-UR22M \
-e 2-2-1-1-H12-UR25M \
-e 2-2-1-1-H12-UR30M \
-e 2-2-1-1-H12-UR33M \
-e 2-2-1-1-H12-UR40M \
-e 2-2-1-1-H12-UR45M \
-e 2-2-1-1-H12-UR50M \

experiment_1=e-7-1-1-H12-UR40M
experiment_2=e-7-1-1-N-UR40M
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment_1 --fp16 -g 8 --twcc \
                               --save-interval-updates 70000 --max-update 70000 --lm-start-step 75000 \
                               --max-tokens 2048 -b 65536 --no-atten-mask 
mkdir checkpoints/$experiment_1/top5_70000steps
cp checkpoints/$experiment_1/checkpoint.best_bleu_*  checkpoints/$experiment_1/top5_70000steps
mkdir checkpoints/$experiment_2/
cp checkpoints/$experiment_1/top5_70000steps/* checkpoints/$experiment_2/
cp checkpoints/$experiment_1/checkpoint_last.pt checkpoints/$experiment_2/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment_1 --fp16 -g 8 --twcc \
                               --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 \
                               --max-tokens 2048 -b 65536 --no-atten-mask 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash call_scripts/train_nat.sh -e $experiment_2 --fp16 -g 8 --twcc \
#                                --save-interval-updates 70000 --max-update 100000 --lm-start-step 75000 \
#                                --max-tokens 2048 -b 65536 --no-atten-mask 

# CUDA_VISIBLE_DEVICES=0 bash call_scripts/generate_nat.sh -b 1 --twcc --data-subset test --ck-types top --avg-speed 1 --no-atten-mask \
#                     -e $experiment_1 \
#                     -e $experiment_2 \


